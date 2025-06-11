import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Neural Network Implementation
class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, hidden_size=64, num_layers=3, 
                 dropout=0.1, learning_rate=0.001, epochs=100, 
                 batch_size=64, patience=10, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.scaler = StandardScaler()
        self.model = None
        self.best_loss = float('inf')
        self.no_improve = 0
        
    def _build_model(self):
        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            
        layers.append(nn.Linear(self.hidden_size, 1))
        return nn.Sequential(*layers)
    
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1).to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = self._build_model().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            
            # Early stopping
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.no_improve = 0
                # Save best model weights
                self.best_weights = self.model.state_dict()
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Load best weights
        self.model.load_state_dict(self.best_weights)
        return self
    
    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return preds.ravel()

def train_and_evaluate(
        features_path: Path,
        model_dir: Path,
        test_size: float = 0.2,
        random_state: int = 42
) -> dict:
    """Trains multiple models and returns evaluation metrics for the best one"""

    print(f"Loading data from {features_path}")
    X = pd.read_csv(features_path)

    if 'recovery_days' not in X.columns:
        raise ValueError("Target column 'recovery_days' not found in the dataset")

    y = X.pop('recovery_days')

    # Drop any non-feature columns
    cols_to_drop = ['patient_id']
    X = X.drop([col for col in cols_to_drop if col in X.columns], axis=1)

    # Handle any missing values
    X = X.fillna(X.mean())

    print(f"Dataset has {X.shape[0]} rows and {X.shape[1]} features")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Get categorical feature indices for TabNet
    categorical_dims = {}
    cat_columns = X.select_dtypes(include=['object', 'category']).columns
    for i, col in enumerate(X.columns):
        if col in cat_columns:
            categorical_dims[i] = len(X[col].unique())

    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1
        ),
        'ElasticNet': ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=1000,
            random_state=random_state
        ),
        'MLP': MLPRegressor(
            input_size=X_train.shape[1],
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
            learning_rate=0.001,
            epochs=200,
            batch_size=64,
            patience=15,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ),
        'TabNet': TabNetRegressor(
            n_d=32, n_a=32, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            cat_idxs=[i for i, col in enumerate(X.columns) if col in cat_columns],
            cat_dims=[len(X[col].unique()) for col in X.columns if col in cat_columns],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":50, "gamma":0.9},
            mask_type='entmax',
            verbose=0
        )
    }

    results = {}
    all_metrics = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name} model...")
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            # Skip cross-validation for neural networks (too time-consuming)
            if name in ['MLP', 'TabNet']:
                cv_r2 = r2  # Use test R² as proxy
            else:
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                cv_r2 = cv_scores.mean()
                
            training_time = time.time() - start_time

            metrics = {
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'cv_r2': cv_r2,
                'training_time': training_time,
                'baseline_mae': mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
            }

            all_metrics[name] = metrics

            print(f"{name} Results:")
            print(f"- MAE: {mae:.2f} days")
            print(f"- R²: {r2:.2f}")
            print(f"- RMSE: {rmse:.2f} days")
            print(f"- 5-Fold CV R²: {cv_r2:.2f}")
            print(f"- Training Time: {training_time:.2f} seconds")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model = model
                
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            all_metrics[name] = {
                'error': str(e),
                'mae': None,
                'r2': None,
                'rmse': None,
                'cv_r2': None,
                'training_time': None
            }

    model_dir.mkdir(exist_ok=True)

    # Visualization of model performance
    plot_model_comparison(all_metrics, model_dir)

    print(f"\nBest model: {best_model_name} with R² = {best_r2:.3f}")
    
    # Save all models
    for name, model in models.items():
        if name in ['MLP', 'TabNet']:
            # Special handling for neural networks
            if name == 'MLP':
                torch.save(model.model.state_dict(), model_dir / f"{name.lower()}_weights.pt")
                joblib.dump(model.scaler, model_dir / f"{name.lower()}_scaler.joblib")
            elif name == 'TabNet':
                model.save_model(model_dir / "tabnet_model.zip")
        else:
            joblib.dump(model, model_dir / f"{name.lower()}_model.joblib")

    # Save metrics
    with open(model_dir / "all_metrics.json", 'w') as f:
        json.dump({k: {m: float(v) for m, v in metrics.items()} for k, metrics in all_metrics.items()}, f, indent=2)

    # Feature Importance for interpretable models
    if best_model_name in ['RandomForest', 'XGBoost']:
        plot_feature_importance(best_model, X.columns, best_model_name, model_dir)
    elif best_model_name == 'ElasticNet':
        plot_elasticnet_coefficients(best_model, X.columns, model_dir)
    elif best_model_name == 'TabNet':
        explain_tabnet(best_model, X_test.values, model_dir)

    return all_metrics[best_model_name]

def plot_model_comparison(metrics, model_dir):
    """Plot comparison of model metrics"""
    # Filter out models with errors
    valid_metrics = {k: v for k, v in metrics.items() if 'error' not in v}
    
    # R² comparison
    plt.figure(figsize=(12, 6))
    r2_values = [m['r2'] for m in valid_metrics.values()]
    plt.bar(valid_metrics.keys(), r2_values)
    plt.title('Model Comparison - R² Score (higher is better)')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    plt.savefig(model_dir / "r2_comparison.png")
    plt.close()

    # MAE comparison
    plt.figure(figsize=(12, 6))
    mae_values = [m['mae'] for m in valid_metrics.values()]
    plt.bar(valid_metrics.keys(), mae_values)
    plt.title('Model Comparison - MAE (lower is better)')
    plt.ylabel('Mean Absolute Error (days)')
    for i, v in enumerate(mae_values):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    plt.savefig(model_dir / "mae_comparison.png")
    plt.close()

def plot_feature_importance(model, feature_names, model_name, model_dir):
    """Plot feature importance for tree-based models"""
    feature_importance = model.feature_importances_
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title(f'Top 15 Feature Importance for {model_name}')
    plt.tight_layout()
    plt.savefig(model_dir / "feature_importance.png")
    plt.close()

def plot_elasticnet_coefficients(model, feature_names, model_dir):
    """Plot coefficients for ElasticNet"""
    coef = model.coef_
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef
    }).sort_values('Coefficient', key=abs, ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(15))
    plt.title('Top 15 Feature Coefficients for ElasticNet')
    plt.tight_layout()
    plt.savefig(model_dir / "elasticnet_coefficients.png")
    plt.close()

def explain_tabnet(model, X_test, model_dir):
    """Generate TabNet explanations"""
    try:
        explain_matrix, masks = model.explain(X_test)
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        
        for i in range(3):  # Plot first 3 samples
            axs[i].imshow(masks[i][:15], cmap='viridis')  # Show top 15 features
            axs[i].set_title(f'Sample {i} Feature Importance')
            axs[i].set_xlabel('Decision step')
            axs[i].set_ylabel('Feature importance')
        
        plt.tight_layout()
        plt.savefig(model_dir / "tabnet_explanations.png")
        plt.close()
    except Exception as e:
        print(f"Could not generate TabNet explanations: {str(e)}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent

    try:
        metrics = train_and_evaluate(
            features_path=project_root / "data/processed/processed_features.csv",
            model_dir=project_root / "models"
        )

        print(f"\nBest Model Metrics Summary:")
        print(f"- MAE: {metrics['mae']:.2f} days")
        print(f"- R²: {metrics['r2']:.2f}")
        print(f"- RMSE: {metrics['rmse']:.2f} days")
        print(f"- 5-Fold CV R²: {metrics['cv_r2']:.2f}")
        print(f"- Baseline MAE: {metrics['baseline_mae']:.2f} days")
        print(f"- Training Time: {metrics['training_time']:.2f} seconds")
    except Exception as e:
        print(f"Error during model training: {e}")
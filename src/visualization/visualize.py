import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(data_path: str, output_path: str) -> None:
    """
    Generate EDA visualizations
    
    Args:
        data_path: Path to processed data
        output_path: Directory to save plots
    """
    df = pd.read_csv(data_path)
    
    # Numeric features
    plt.figure(figsize=(12, 6))
    sns.histplot(df['recovery_days'], kde=True)
    plt.title('Recovery Days Distribution')
    plt.savefig(f"{output_path}/recovery_dist.png")
    plt.close()
    
    # Categorical features
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='diagnosis', y='recovery_days', data=df)
    plt.xticks(rotation=45)
    plt.title('Recovery Days by Diagnosis')
    plt.savefig(f"{output_path}/diagnosis_boxplot.png")
    plt.close()

if __name__ == "__main__":
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    plot_feature_distributions(
        data_path=f"{project_dir}/data/processed/features.csv",
        output_path=f"{project_dir}/reports/figures"
    )
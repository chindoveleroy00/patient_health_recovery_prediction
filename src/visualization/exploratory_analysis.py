import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np
from scipy import stats

# Updated for NumPy 2.0 compatibility
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def generate_eda_report(input_path: Path, output_dir: Path):
    """Generates exploratory data analysis visualizations"""
    df = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Basic statistics
    stats_dict = {
        'total_patients': len(df),
        'avg_recovery_days': df['recovery_days'].mean(),
        'max_recovery_days': df['recovery_days'].max(),
        'min_recovery_days': df['recovery_days'].min(),
        'median_recovery_days': df['recovery_days'].median(),
        'age_distribution': {
            'min': df['age'].min(),
            'max': df['age'].max(),
            'mean': df['age'].mean(),
            'median': df['age'].median()
        },
        'gender_distribution': df['gender'].value_counts().to_dict(),
        'severity_distribution': df['severity'].value_counts().to_dict(),
        'ward_distribution': df['ward_type'].value_counts().to_dict(),
        'treatment_distribution': df['treatment_type'].value_counts().to_dict(),
        'complication_rate': df['complications'].mean(),
        'admission_type_distribution': df['admission_type'].value_counts().to_dict(),
        'medication_given_distribution': df['medication_given'].value_counts().to_dict(),
        'avg_duration_of_treatment': df['duration_of_treatment'].mean(),
        'avg_length_of_stay': df['length_of_stay'].mean(),
        'diagnosis_distribution': df['diagnosis'].value_counts().to_dict(),  # Added diagnosis distribution
        'admission_date_distribution': df['admission_date'].value_counts().to_dict()  # New admission_date distribution
    }

    # Convert for JSON
    serializable_stats = {
        key: {k: convert_to_serializable(v) for k, v in value.items()}
        if isinstance(value, dict) else convert_to_serializable(value)
        for key, value in stats_dict.items()
    }

    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(serializable_stats, f, indent=2)

    # === VISUALIZATIONS ===
    sns.set(style='whitegrid')

    # Recovery days distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['recovery_days'], kde=True)
    plt.title('Distribution of Recovery Days')
    plt.savefig(output_dir / 'recovery_days_distribution.png')

    # Recovery by severity
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='severity', y='recovery_days', data=df)
    plt.title('Recovery Time by Severity')
    plt.savefig(output_dir / 'recovery_by_severity.png')

    # Age vs recovery time (with trendline)
    plt.figure(figsize=(10, 6))
    sns.regplot(x='age', y='recovery_days', data=df, scatter_kws={'alpha': 0.5})
    plt.title('Age vs Recovery Time (with Trendline)')
    plt.savefig(output_dir / 'age_vs_recovery_trendline.png')

    # Treatment type comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='treatment_type', y='recovery_days', data=df)
    plt.title('Recovery Time by Treatment Type')
    plt.savefig(output_dir / 'recovery_by_treatment.png')

    # Complications impact
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='complications', y='recovery_days', data=df)
    plt.title('Impact of Complications on Recovery Time')
    plt.xticks([0, 1], ['No Complications', 'Complications'])
    plt.savefig(output_dir / 'complications_impact.png')

    # Gender comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='gender', y='recovery_days', data=df)
    plt.title('Recovery Time by Gender')
    plt.savefig(output_dir / 'gender_recovery.png')

    # Admission type comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='admission_type', y='recovery_days', data=df)
    plt.title('Recovery Time by Admission Type')
    plt.savefig(output_dir / 'recovery_by_admission_type.png')

    # Medication given vs recovery
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='medication_given', y='recovery_days', data=df)
    plt.title('Recovery Time by Medication Given')
    plt.savefig(output_dir / 'recovery_by_medication.png')

    # Duration of treatment vs recovery days (with trendline)
    plt.figure(figsize=(10, 6))
    sns.regplot(x='duration_of_treatment', y='recovery_days', data=df, scatter_kws={'alpha': 0.5})
    plt.title('Duration of Treatment vs Recovery Days (Trendline)')
    plt.savefig(output_dir / 'duration_vs_recovery_trendline.png')

    # Length of stay vs recovery days (with trendline)
    plt.figure(figsize=(10, 6))
    sns.regplot(x='length_of_stay', y='recovery_days', data=df, scatter_kws={'alpha': 0.5})
    plt.title('Length of Stay vs Recovery Days (Trendline)')
    plt.savefig(output_dir / 'length_of_stay_vs_recovery_trendline.png')

    # Diagnosis comparison (new visualization)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y='recovery_days', data=df)
    plt.title('Recovery Time by Diagnosis')
    plt.xticks(rotation=45)
    plt.savefig(output_dir / 'recovery_by_diagnosis.png')

    # Admission date comparison (new visualization)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='admission_date', data=df)
    plt.title('Admissions by Date')
    plt.xticks(rotation=45)
    plt.savefig(output_dir / 'admission_date_distribution.png')

    # === CORRELATION HEATMAP ===
    numeric_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(output_dir / 'correlation_heatmap.png')

    # === ANOVA TEST ===
    anova_results = {}
    categorical_features = ['gender', 'severity', 'treatment_type', 'admission_type', 'ward_type', 'medication_given', 'diagnosis', 'admission_date']  # Included admission_date

    for feature in categorical_features:
        groups = [group['recovery_days'].values for _, group in df.groupby(feature)]
        try:
            f_val, p_val = stats.f_oneway(*groups)
            anova_results[feature] = {
                'f_statistic': float(f_val),
                'p_value': float(p_val)
            }
        except Exception as e:
            anova_results[feature] = str(e)

    # Save ANOVA results to JSON
    with open(output_dir / 'anova_results.json', 'w') as f:
        json.dump(anova_results, f, indent=2)

    print(f"✅ EDA report generated in {output_dir}")

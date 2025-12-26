#!/usr/bin/env python3
"""
Heart Disease UCI Dataset - Exploratory Data Analysis (EDA)
Task 1: Data Acquisition & Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory for plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'screenshots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    """Load the heart disease dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'heart_disease.csv')
    df = pd.read_csv(data_path)
    return df

def data_overview(df):
    """Print basic dataset statistics."""
    print("=" * 60)
    print("HEART DISEASE UCI DATASET - OVERVIEW")
    print("=" * 60)
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn Names and Types:")
    print(df.dtypes)
    print(f"\nBasic Statistics:")
    print(df.describe())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nClass Distribution (Target):")
    print(df['target'].value_counts())
    print(f"\nClass Balance Ratio: {df['target'].value_counts()[1]/df['target'].value_counts()[0]:.2f}")

def plot_target_distribution(df):
    """Plot target variable distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    colors = ['#2ecc71', '#e74c3c']
    ax1 = axes[0]
    counts = df['target'].value_counts()
    bars = ax1.bar(['No Disease (0)', 'Disease (1)'], counts.values, color=colors, edgecolor='black')
    ax1.set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xlabel('Target Class', fontsize=12)
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                str(count), ha='center', fontsize=12, fontweight='bold')
    
    # Pie chart
    ax2 = axes[1]
    ax2.pie(counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%',
            colors=colors, explode=[0.02, 0.02], shadow=True, startangle=90)
    ax2.set_title('Class Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'target_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: target_distribution.png")

def plot_feature_histograms(df):
    """Plot histograms for numerical features."""
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        # Plot by target class
        for target, color, label in [(0, '#2ecc71', 'No Disease'), (1, '#e74c3c', 'Disease')]:
            data = df[df['target'] == target][col]
            ax.hist(data, bins=20, alpha=0.6, color=color, label=label, edgecolor='black')
        ax.set_title(f'{col.upper()} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
    
    # Remove empty subplot
    axes[5].axis('off')
    
    plt.suptitle('Numerical Features Distribution by Target Class', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: feature_histograms.png")

def plot_correlation_heatmap(df):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap.png")

def plot_categorical_features(df):
    """Plot categorical feature distributions."""
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    feature_names = {
        'sex': ['Female', 'Male'],
        'cp': ['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'],
        'fbs': ['< 120 mg/dl', '> 120 mg/dl'],
        'restecg': ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'],
        'exang': ['No', 'Yes'],
        'slope': ['Upsloping', 'Flat', 'Downsloping'],
        'ca': ['0', '1', '2', '3', '4'],
        'thal': ['Normal', 'Fixed Defect', 'Reversible Defect']
    }
    
    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        
        # Cross-tabulation
        ct = pd.crosstab(df[col], df['target'])
        ct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8)
        
        ax.set_title(f'{col.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(['No Disease', 'Disease'], loc='upper right')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Categorical Features Distribution by Target Class', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'categorical_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: categorical_features.png")

def plot_boxplots(df):
    """Plot boxplots for numerical features by target."""
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        df.boxplot(column=col, by='target', ax=ax, 
                   boxprops=dict(color='blue'),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(col.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Target', fontsize=10)
        ax.set_ylabel(col, fontsize=10)
    
    plt.suptitle('Numerical Features Box Plots by Target Class', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: boxplots.png")

def plot_age_analysis(df):
    """Detailed age analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Age distribution by target
    ax1 = axes[0]
    sns.kdeplot(data=df[df['target']==0]['age'], ax=ax1, color='#2ecc71', 
                fill=True, alpha=0.5, label='No Disease')
    sns.kdeplot(data=df[df['target']==1]['age'], ax=ax1, color='#e74c3c', 
                fill=True, alpha=0.5, label='Disease')
    ax1.set_title('Age Distribution (KDE)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Age', fontsize=10)
    ax1.legend()
    
    # Age groups analysis
    ax2 = axes[1]
    df['age_group'] = pd.cut(df['age'], bins=[29, 40, 50, 60, 70, 80], 
                             labels=['30-40', '41-50', '51-60', '61-70', '71-80'])
    age_disease = df.groupby('age_group')['target'].mean() * 100
    bars = ax2.bar(age_disease.index, age_disease.values, color='#3498db', edgecolor='black')
    ax2.set_title('Disease Rate by Age Group', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Age Group', fontsize=10)
    ax2.set_ylabel('Disease Rate (%)', fontsize=10)
    for bar, val in zip(bars, age_disease.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=10)
    
    # Age vs Max Heart Rate scatter
    ax3 = axes[2]
    scatter = ax3.scatter(df['age'], df['thalach'], c=df['target'], 
                         cmap='RdYlGn_r', alpha=0.6, edgecolors='black', s=50)
    ax3.set_title('Age vs Max Heart Rate', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Age', fontsize=10)
    ax3.set_ylabel('Max Heart Rate (thalach)', fontsize=10)
    plt.colorbar(scatter, ax=ax3, label='Target')
    
    df.drop('age_group', axis=1, inplace=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'age_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: age_analysis.png")

def plot_pair_plot(df):
    """Create pair plot for key features."""
    key_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
    
    g = sns.pairplot(df[key_features], hue='target', 
                    palette={0: '#2ecc71', 1: '#e74c3c'},
                    diag_kind='kde', corner=True,
                    plot_kws={'alpha': 0.6, 'edgecolor': 'black', 's': 40})
    g.fig.suptitle('Pair Plot - Key Features', fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(os.path.join(PLOTS_DIR, 'pair_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pair_plot.png")

def generate_eda_summary(df):
    """Generate and save EDA summary."""
    summary = []
    summary.append("=" * 60)
    summary.append("HEART DISEASE UCI DATASET - EDA SUMMARY")
    summary.append("=" * 60)
    summary.append(f"\n1. Dataset Overview:")
    summary.append(f"   - Total samples: {len(df)}")
    summary.append(f"   - Features: {len(df.columns) - 1}")
    summary.append(f"   - Target classes: 2 (Binary Classification)")
    
    summary.append(f"\n2. Class Distribution:")
    summary.append(f"   - No Disease (0): {df['target'].value_counts()[0]} ({df['target'].value_counts()[0]/len(df)*100:.1f}%)")
    summary.append(f"   - Disease (1): {df['target'].value_counts()[1]} ({df['target'].value_counts()[1]/len(df)*100:.1f}%)")
    
    summary.append(f"\n3. Missing Values: None (after preprocessing)")
    
    summary.append(f"\n4. Key Feature Statistics:")
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        summary.append(f"   - {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    
    summary.append(f"\n5. Key Correlations with Target:")
    corr_with_target = df.corr()['target'].drop('target').sort_values(key=abs, ascending=False)
    for feat, corr in corr_with_target.head(5).items():
        summary.append(f"   - {feat}: {corr:.3f}")
    
    summary_text = '\n'.join(summary)
    print(summary_text)
    
    with open(os.path.join(PLOTS_DIR, 'eda_summary.txt'), 'w') as f:
        f.write(summary_text)
    print("\nSaved: eda_summary.txt")

def main():
    """Run complete EDA pipeline."""
    print("Starting EDA Pipeline...")
    
    # Load data
    df = load_data()
    
    # Generate overview
    data_overview(df)
    
    # Generate all plots
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_target_distribution(df)
    plot_feature_histograms(df)
    plot_correlation_heatmap(df)
    plot_categorical_features(df)
    plot_boxplots(df)
    plot_age_analysis(df)
    plot_pair_plot(df)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    generate_eda_summary(df)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()

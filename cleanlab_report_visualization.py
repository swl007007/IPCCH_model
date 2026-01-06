"""
Cleanlab Results Visualization and Report Generation

This script creates visualizations and a comprehensive report from the cleanlab analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("GENERATING CLEANLAB LABEL QUALITY REPORT AND VISUALIZATIONS")
print("="*80)

# Load summary statistics
summary_df = pd.read_csv('cleanlab_results_summary.csv')
critical_samples = pd.read_csv('cleanlab_results_critical_samples.csv')

# Load detailed results for each phase
phase_results = {}
for phase in range(2, 6):
    phase_results[phase] = pd.read_csv(f'cleanlab_results_phase{phase}_label_quality.csv')

# Create PDF report
pdf_filename = 'cleanlab_label_quality_report.pdf'
with PdfPages(pdf_filename) as pdf:

    # Page 1: Summary Statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cleanlab Label Quality Analysis - Summary', fontsize=16, fontweight='bold')

    # 1.1 Quality scores by phase
    ax = axes[0, 0]
    x = summary_df['phase']
    y = summary_df['mean_quality_score']
    yerr = summary_df['std_quality_score']
    ax.bar(x, y, yerr=yerr, capsize=5, color=sns.color_palette("Blues_d", 4), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Phase', fontweight='bold')
    ax.set_ylabel('Mean Quality Score', fontweight='bold')
    ax.set_title('Average Label Quality Score by Phase')
    ax.set_ylim([0, 1.1])
    ax.set_xticks([2, 3, 4, 5])
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi, yi + 0.02, f'{yi:.3f}', ha='center', va='bottom', fontweight='bold')

    # 1.2 Distribution of quality categories
    ax = axes[0, 1]
    categories = ['High', 'Medium', 'Low']
    width = 0.2
    x_pos = np.arange(len(categories))
    for i, phase in enumerate(range(2, 6)):
        counts = [
            summary_df.loc[i, 'high_quality_count'],
            summary_df.loc[i, 'medium_quality_count'],
            summary_df.loc[i, 'low_quality_count']
        ]
        ax.bar(x_pos + i*width, counts, width, label=f'Phase {phase}', alpha=0.7)

    ax.set_xlabel('Quality Category', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Distribution of Label Quality Categories')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(categories)
    ax.legend()

    # 1.3 Prediction errors by phase
    ax = axes[1, 0]
    error_metrics = summary_df[['phase', 'mean_abs_error', 'median_abs_error', 'p90_abs_error']]
    x_pos = np.arange(len(error_metrics))
    width = 0.25
    ax.bar(x_pos - width, error_metrics['mean_abs_error'], width, label='Mean', alpha=0.7)
    ax.bar(x_pos, error_metrics['median_abs_error'], width, label='Median', alpha=0.7)
    ax.bar(x_pos + width, error_metrics['p90_abs_error'], width, label='90th Percentile', alpha=0.7)
    ax.set_xlabel('Phase', fontweight='bold')
    ax.set_ylabel('Absolute Error (%)', fontweight='bold')
    ax.set_title('Prediction Errors by Phase')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([2, 3, 4, 5])
    ax.legend()

    # 1.4 Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            f"Phase {int(row['phase'])}",
            f"{row['mean_quality_score']:.3f}",
            f"{row['mean_abs_error']:.3f}%",
            f"{row['low_quality_count']}"
        ])

    table = ax.table(cellText=table_data,
                    colLabels=['Phase', 'Avg Quality', 'Avg Error', 'Low Qual Count'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Summary Statistics', fontweight='bold', pad=20)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 2-5: Individual phase distributions
    for phase in range(2, 6):
        phase_data = phase_results[phase]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Phase {phase}_worse Label Quality Analysis', fontsize=16, fontweight='bold')

        # 2.1 Quality score distribution
        ax = axes[0, 0]
        ax.hist(phase_data['label_quality_score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(phase_data['label_quality_score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(phase_data['label_quality_score'].median(), color='green', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel('Label Quality Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'Distribution of Quality Scores (Phase {phase})')
        ax.legend()

        # 2.2 Quality by year
        ax = axes[0, 1]
        yearly_quality = phase_data.groupby('year')['label_quality_score'].mean()
        ax.plot(yearly_quality.index, yearly_quality.values, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.fill_between(yearly_quality.index, yearly_quality.values, alpha=0.3, color='steelblue')
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Mean Quality Score', fontweight='bold')
        ax.set_title(f'Label Quality Trend Over Time (Phase {phase})')
        ax.grid(True, alpha=0.3)

        # 2.3 Error distribution
        ax = axes[1, 0]
        ax.hist(phase_data['absolute_error'], bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax.set_xlabel('Absolute Error (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'Distribution of Prediction Errors (Phase {phase})')

        # 2.4 Quality category counts by year
        ax = axes[1, 1]
        quality_by_year = phase_data.groupby(['year', 'quality_category']).size().unstack(fill_value=0)
        quality_by_year.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title(f'Quality Categories by Year (Phase {phase})')
        ax.legend(title='Quality', labels=['High', 'Low', 'Medium'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # Page 6: Critical samples analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Critical Samples Analysis (Low Quality in Multiple Phases)', fontsize=16, fontweight='bold')

    # 6.1 Distribution of multi-phase issues
    ax = axes[0, 0]
    phase_count_dist = critical_samples['num_low_quality_phases'].value_counts().sort_index()
    ax.bar(phase_count_dist.index, phase_count_dist.values, color='crimson', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Phases with Low Quality', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Samples with Multi-Phase Quality Issues')
    for i, v in enumerate(phase_count_dist.values):
        ax.text(phase_count_dist.index[i], v + 5, str(v), ha='center', va='bottom', fontweight='bold')

    # 6.2 Critical samples by year
    ax = axes[0, 1]
    yearly_critical = critical_samples.groupby('year').size()
    ax.bar(yearly_critical.index, yearly_critical.values, color='orangered', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Number of Critical Samples', fontweight='bold')
    ax.set_title('Critical Samples Distribution by Year')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # 6.3 Top 10 regions with most critical samples
    ax = axes[1, 0]
    top_regions = critical_samples['area_id'].value_counts().head(10)
    ax.barh(range(len(top_regions)), top_regions.values, color='tomato', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_regions)))
    ax.set_yticklabels([f'Area {area_id}' for area_id in top_regions.index])
    ax.set_xlabel('Number of Critical Samples', fontweight='bold')
    ax.set_ylabel('Region (Area ID)', fontweight='bold')
    ax.set_title('Top 10 Regions with Most Critical Samples')
    ax.invert_yaxis()

    # 6.4 Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    total_critical = len(critical_samples)
    critical_4_phases = len(critical_samples[critical_samples['num_low_quality_phases'] == 4])
    critical_3_phases = len(critical_samples[critical_samples['num_low_quality_phases'] == 3])
    unique_regions = critical_samples['area_id'].nunique()

    stats_data = [
        ['Total Critical Samples', f'{total_critical}'],
        ['Samples with 4-phase issues', f'{critical_4_phases} ({critical_4_phases/total_critical*100:.1f}%)'],
        ['Samples with 3-phase issues', f'{critical_3_phases} ({critical_3_phases/total_critical*100:.1f}%)'],
        ['Unique Regions Affected', f'{unique_regions}'],
        ['% of Total Dataset', f'{total_critical/29621*100:.2f}%']
    ]

    table = ax.table(cellText=stats_data,
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Critical Samples Statistics', fontweight='bold', pad=20)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 7: Comparison across phases
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Phase Comparison', fontsize=16, fontweight='bold')

    # 7.1 Quality score boxplots
    ax = axes[0, 0]
    quality_data = [phase_results[p]['label_quality_score'] for p in range(2, 6)]
    bp = ax.boxplot(quality_data, labels=[f'Phase {p}' for p in range(2, 6)], patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("Set2", 4)):
        patch.set_facecolor(color)
    ax.set_xlabel('Phase', fontweight='bold')
    ax.set_ylabel('Quality Score', fontweight='bold')
    ax.set_title('Quality Score Distribution Across Phases')
    ax.grid(True, alpha=0.3)

    # 7.2 Error boxplots
    ax = axes[0, 1]
    error_data = [phase_results[p]['absolute_error'] for p in range(2, 6)]
    bp = ax.boxplot(error_data, labels=[f'Phase {p}' for p in range(2, 6)], patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("Set2", 4)):
        patch.set_facecolor(color)
    ax.set_xlabel('Phase', fontweight='bold')
    ax.set_ylabel('Absolute Error (%)', fontweight='bold')
    ax.set_title('Prediction Error Distribution Across Phases')
    ax.grid(True, alpha=0.3)

    # 7.3 Low quality samples over time
    ax = axes[1, 0]
    for phase in range(2, 6):
        low_quality_by_year = phase_results[phase][phase_results[phase]['quality_category'] == 'low'].groupby('year').size()
        ax.plot(low_quality_by_year.index, low_quality_by_year.values, marker='o', label=f'Phase {phase}', linewidth=2)
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Number of Low Quality Samples', fontweight='bold')
    ax.set_title('Low Quality Samples Trend by Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7.4 Percentage of low quality samples
    ax = axes[1, 1]
    low_quality_pct = []
    for phase in range(2, 6):
        total = len(phase_results[phase])
        low = len(phase_results[phase][phase_results[phase]['quality_category'] == 'low'])
        low_quality_pct.append(low / total * 100)

    ax.bar([2, 3, 4, 5], low_quality_pct, color=sns.color_palette("Reds_d", 4), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Phase', fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Percentage of Low Quality Labels by Phase')
    ax.set_xticks([2, 3, 4, 5])
    for i, (x, y) in enumerate(zip([2, 3, 4, 5], low_quality_pct)):
        ax.text(x, y + 0.1, f'{y:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"\nPDF report saved: {pdf_filename}")

# Generate text summary report
print("\n" + "="*80)
print("GENERATING TEXT SUMMARY REPORT")
print("="*80)

with open('cleanlab_label_quality_summary_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLEANLAB LABEL QUALITY ANALYSIS REPORT\n")
    f.write("IPC/CH Forecasting Model - Phase{}_worse Variables\n")
    f.write("="*80 + "\n\n")

    f.write("EXECUTIVE SUMMARY\n")
    f.write("-"*80 + "\n")
    f.write(f"Total samples analyzed: 29,621\n")
    f.write(f"Number of phase models: 4 (Phase 2, 3, 4, 5)\n")
    f.write(f"Critical samples (low quality in 3+ phases): {len(critical_samples)} ({len(critical_samples)/29621*100:.2f}%)\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-"*80 + "\n")
    f.write("1. Overall Label Quality:\n")
    for _, row in summary_df.iterrows():
        phase = int(row['phase'])
        f.write(f"   - Phase {phase}: Mean quality score = {row['mean_quality_score']:.3f}, "
                f"Mean error = {row['mean_abs_error']:.3f}%\n")

    f.write("\n2. Label Quality Distribution:\n")
    for _, row in summary_df.iterrows():
        phase = int(row['phase'])
        total = row['total_samples']
        high_pct = row['high_quality_count'] / total * 100
        medium_pct = row['medium_quality_count'] / total * 100
        low_pct = row['low_quality_count'] / total * 100
        f.write(f"   - Phase {phase}: High={high_pct:.1f}%, Medium={medium_pct:.1f}%, Low={low_pct:.1f}%\n")

    f.write("\n3. Temporal Patterns:\n")
    for phase in range(2, 6):
        yearly_avg = phase_results[phase].groupby('year')['label_quality_score'].mean()
        best_year = yearly_avg.idxmax()
        worst_year = yearly_avg.idxmin()
        f.write(f"   - Phase {phase}: Best year = {best_year} ({yearly_avg[best_year]:.3f}), "
                f"Worst year = {worst_year} ({yearly_avg[worst_year]:.3f})\n")

    f.write("\n4. Regional Patterns:\n")
    top_regions = critical_samples['area_id'].value_counts().head(5)
    f.write("   Top 5 regions with most critical samples:\n")
    for area_id, count in top_regions.items():
        f.write(f"   - Area {area_id}: {count} critical samples\n")

    f.write("\n5. Multi-Phase Quality Issues:\n")
    phase_count_dist = critical_samples['num_low_quality_phases'].value_counts().sort_index()
    for num_phases, count in phase_count_dist.items():
        f.write(f"   - {count} samples have low quality in {num_phases} phases\n")

    f.write("\n" + "="*80 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("="*80 + "\n")
    f.write("1. Review and validate the {0} critical samples with low quality in 3+ phases\n".format(len(critical_samples)))
    f.write("2. Focus data quality improvement efforts on:\n")
    for phase in range(2, 6):
        yearly_avg = phase_results[phase].groupby('year')['label_quality_score'].mean()
        worst_year = yearly_avg.idxmin()
        f.write(f"   - Phase {phase}: Year {worst_year} (quality score = {yearly_avg[worst_year]:.3f})\n")
    f.write("3. Investigate regions with consistently high error rates (see critical_samples.csv)\n")
    f.write("4. Consider using label quality scores as sample weights during model training\n")
    f.write("5. Implement automated data quality checks for new data collection\n")

    f.write("\n" + "="*80 + "\n")
    f.write("DETAILED STATISTICS BY PHASE\n")
    f.write("="*80 + "\n\n")

    for _, row in summary_df.iterrows():
        phase = int(row['phase'])
        f.write(f"PHASE {phase}_WORSE\n")
        f.write("-"*40 + "\n")
        f.write(f"Total samples: {int(row['total_samples']):,}\n")
        f.write(f"Mean quality score: {row['mean_quality_score']:.4f} ± {row['std_quality_score']:.4f}\n")
        f.write(f"High quality samples: {int(row['high_quality_count']):,} ({row['high_quality_count']/row['total_samples']*100:.1f}%)\n")
        f.write(f"Medium quality samples: {int(row['medium_quality_count']):,} ({row['medium_quality_count']/row['total_samples']*100:.1f}%)\n")
        f.write(f"Low quality samples: {int(row['low_quality_count']):,} ({row['low_quality_count']/row['total_samples']*100:.1f}%)\n")
        f.write(f"Mean absolute error: {row['mean_abs_error']:.4f}%\n")
        f.write(f"Median absolute error: {row['median_abs_error']:.4f}%\n")
        f.write(f"90th percentile error: {row['p90_abs_error']:.4f}%\n")
        f.write(f"Max absolute error: {row['max_abs_error']:.4f}%\n")
        f.write("\n")

print("Text summary report saved: cleanlab_label_quality_summary_report.txt")

print("\n" + "="*80)
print("REPORT GENERATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. cleanlab_label_quality_report.pdf - Visual report with charts and graphs")
print("  2. cleanlab_label_quality_summary_report.txt - Text summary with key findings")
print("\nAll cleanlab analysis files are ready for review!")

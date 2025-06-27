import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime
import re

def parse_markdown_table(md_content):
    """
    Parses markdown content to extract data from tables under specific categories.
    """
    categories = re.split(r'## Category: ', md_content)[1:]
    all_data = {}

    for category_block in categories:
        lines = category_block.strip().split('\n')
        category_name = lines[0].strip()
        
        table_lines = [line for line in lines if '|' in line and '---' not in line and 'System' not in line]
        
        data = []
        for line in table_lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 7:
                system, _, _, nodes, path_length, _, _ = parts
                data.append([system, nodes, path_length])

        df = pd.DataFrame(data, columns=['System', 'Nodes', 'Path Length'])
        
        # Clean and convert data
        df['Nodes'] = pd.to_numeric(df['Nodes'], errors='coerce')
        df['Path Length'] = pd.to_numeric(df['Path Length'], errors='coerce')
        
        all_data[category_name] = df

    return all_data

def cohen_d(x, y):
    """
    Calculates Cohen's d for independent samples.
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def interpret_p_value(p_value):
    if p_value < 0.001:
        return f"p={p_value:.4f} (Very strong evidence against the null hypothesis; highly significant)."
    elif p_value < 0.01:
        return f"p={p_value:.4f} (Strong evidence against the null hypothesis; very significant)."
    elif p_value < 0.05:
        return f"p={p_value:.4f} (Moderate evidence against the null hypothesis; significant)."
    elif p_value < 0.1:
        return f"p={p_value:.4f} (Weak evidence against the null hypothesis; marginally significant)."
    else:
        return f"p={p_value:.4f} (Little to no evidence against the null hypothesis; not significant)."

def interpret_cohen_d(d):
    if abs(d) < 0.2:
        return f"d={d:.4f} (Trivial effect size)."
    elif abs(d) < 0.5:
        return f"d={d:.4f} (Small effect size)."
    elif abs(d) < 0.8:
        return f"d={d:.4f} (Medium effect size)."
    else:
        return f"d={d:.4f} (Large effect size)."

def run_analysis(data_by_category, output_dir, plot_dir, log_file):
    """
    Performs statistical analysis and generates a report.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting statistical analysis.")

    report_path = os.path.join(output_dir, 'statistical_analysis_report.md')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("# Statistical Analysis Report\n\n")
        f.write("This report presents a statistical comparison of 'Nodes' and 'Path Length' between WT and Mutant systems across the four categories found in the data file. The analysis aims to determine if there are significant differences between the two systems for these metrics.\n\n")
        f.write("### Understanding the Statistics\n\n")
        f.write("**Null Hypothesis:** For each test, the null hypothesis (H0) states that there is no significant difference between the distributions of the WT and Mutant groups. A low p-value (typically < 0.05) suggests that we can reject this hypothesis, indicating a statistically significant difference.\n\n")
        f.write("**Not Significant:** A non-significant result (p-value >= 0.05) means that any observed differences between the groups are likely due to random chance, and there is not enough evidence to conclude that the groups are truly different.\n\n")

        for category, df in data_by_category.items():
            logging.info(f"Processing category: {category}")
            f.write(f"## Category: {category}\n\n")

            wt_data = df[df['System'] == 'WT'].dropna()
            mutant_data = df[df['System'] == 'Mutant'].dropna()
            
            f.write(f"Number of entries: WT = {len(wt_data)}, Mutant = {len(mutant_data)}\n\n")

            results = []
            interpretations = {}
            for col in ['Nodes', 'Path Length']:
                wt_series = wt_data[col]
                mutant_series = mutant_data[col]

                if len(wt_series) > 1 and len(mutant_series) > 1:
                    mwu_stat, mwu_p = mannwhitneyu(wt_series, mutant_series, alternative='two-sided')
                    ks_stat, ks_p = ks_2samp(wt_series, mutant_series)
                    effect_size = cohen_d(mutant_series, wt_series)

                    results.append([col, f"{mwu_stat:.4f}", f"{mwu_p:.4f}", f"{ks_stat:.4f}", f"{ks_p:.4f}", f"{effect_size:.4f}"])
                    interpretations[col] = {
                        'mwu': interpret_p_value(mwu_p),
                        'ks': interpret_p_value(ks_p),
                        'cohen_d': interpret_cohen_d(effect_size)
                    }
                    
                    plot_data = pd.concat([
                        pd.DataFrame({'Value': wt_series, 'System': 'WT'}),
                        pd.DataFrame({'Value': mutant_series, 'System': 'Mutant'})
                    ])

                    plt.figure(figsize=(10, 6))
                    ax = sns.boxplot(x='System', y='Value', data=plot_data, palette="Set2",
                                     medianprops={'color': 'black', 'linewidth': 2})
                    
                    ax.set_title(f'Distribution of {col} in {category}')
                    plot_filename = f"{category.replace(' ', '_')}_{col}_boxplot.png"
                    plot_filepath = os.path.join(plot_dir, plot_filename)
                    plt.savefig(plot_filepath)
                    plt.close()
                    logging.info(f"Generated plot: {plot_filepath}")

                else:
                    results.append([col, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
                    logging.warning(f"
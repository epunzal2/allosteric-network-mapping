import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime
import re
import argparse

def parse_optimal_paths(content):
    """
    Parses optimal_paths_details.md to extract relevant data.
    """
    data = []
    current_category = None
    for line in content.split('\n'):
        if line.startswith('## Category:'):
            current_category = line.replace('## Category:', '').strip()
        elif '|' in line and 'System' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 7:
                system, residue_pair, _, _, _, _, optimal_path = parts
                data.append([current_category, residue_pair, system, optimal_path])
    
    return pd.DataFrame(data, columns=['Category', 'Residue Pair', 'System', 'Optimal Path Residues'])

def parse_extended_report(content):
    """
    Parses extended_report.md to extract common optimal residues.
    """
    common_residues_map = {}
    pattern = re.compile(r"\| ([\d\-]+) \|.*?\| \*\*Comparison\*\* \|.*?\| Common Optimal Residues: ([\w\s,]+) \(\d+ shared\)")
    for line in content.split('\n'):
        match = pattern.search(line)
        if match:
            residue_pair = match.group(1)
            common_residues_str = match.group(2).strip()
            if common_residues_str.lower() != 'n/a':
                common_residues_map[residue_pair] = [res.strip() for res in common_residues_str.split(',')]
    return common_residues_map

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

def plot_jaccard_similarity(data, category, plot_dir):
    """
    Generates and saves a bar plot for Jaccard similarities.
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Jaccard Similarity', y='Residue Pair', data=data, palette='viridis')
    plt.title(f'Jaccard Similarity of Optimal Paths for {category}')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Residue Pair')
    plt.tight_layout()
    plot_filename = f"jaccard_similarity_{category.replace(' ', '_')}.png"
    plot_filepath = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()
    return plot_filepath

def run_analysis(optimal_paths_df, common_residues_map, output_dir, plot_dir, log_file):
    """
    Performs statistical analysis and generates a report.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting statistical analysis.")

    report_path = os.path.join(output_dir, 'statistical_analysis_report.md')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    jaccard_results = []
    shared_residue_analysis = []

    with open(report_path, 'w') as f:
        f.write("# Statistical Analysis Report\n\n")
        f.write("This report presents a statistical comparison of optimal path similarity using the Jaccard index and an analysis of shared intermediate residues between WT and Mutant systems.\n\n")

        categories = optimal_paths_df['Category'].unique()
        for category in categories:
            if "high contact" not in category.lower():
                continue

            f.write(f"## Category: {category}\n\n")
            logging.info(f"Processing category: {category}")

            category_df = optimal_paths_df[optimal_paths_df['Category'] == category]
            residue_pairs = category_df['Residue Pair'].unique()
            
            category_jaccard_data = []

            for pair in residue_pairs:
                wt_row = category_df[(category_df['Residue Pair'] == pair) & (category_df['System'] == 'WT')]
                mutant_row = category_df[(category_df['Residue Pair'] == pair) & (category_df['System'] == 'Mutant')]

                if not wt_row.empty and not mutant_row.empty:
                    wt_path_str = wt_row.iloc[0]['Optimal Path Residues']
                    mutant_path_str = mutant_row.iloc[0]['Optimal Path Residues']

                    if 'N/A' in wt_path_str or 'N/A' in mutant_path_str:
                        continue

                    wt_residues = set(wt_path_str.split(' -> '))
                    mutant_residues = set(mutant_path_str.split(' -> '))
                    
                    intersection = len(wt_residues.intersection(mutant_residues))
                    union = len(wt_residues.union(mutant_residues))
                    
                    jaccard_similarity = intersection / union if union > 0 else 0
                    
                    jaccard_results.append({
                        'Category': category,
                        'Residue Pair': pair,
                        'Jaccard Similarity': jaccard_similarity
                    })
                    category_jaccard_data.append({'Residue Pair': pair, 'Jaccard Similarity': jaccard_similarity})

            if not category_jaccard_data:
                f.write("No valid pairs for Jaccard similarity analysis.\n\n")
                continue

            # Jaccard Similarity Analysis
            jaccard_df_category = pd.DataFrame(category_jaccard_data).sort_values(by='Jaccard Similarity', ascending=False)
            
            highest_sim = jaccard_df_category.iloc[0]
            lowest_sim = jaccard_df_category.iloc[-1]

            f.write("### Jaccard Similarity Analysis\n\n")
            f.write(f"- **Highest Similarity Pair:** {highest_sim['Residue Pair']} ({highest_sim['Jaccard Similarity']:.4f})\n")
            f.write(f"- **Lowest Similarity Pair:** {lowest_sim['Residue Pair']} ({lowest_sim['Jaccard Similarity']:.4f})\n\n")

            plot_filepath = plot_jaccard_similarity(jaccard_df_category, category, plot_dir)
            relative_plot_path = os.path.relpath(plot_filepath, output_dir)
            f.write(f"![Jaccard Similarity for {category}]({relative_plot_path})\n\n")

            # Shared Intermediate Residue Analysis
            category_intermediate_residues = set()
            for pair in residue_pairs:
                if pair in common_residues_map:
                    start_node, end_node = pair.split('-')
                    common = set(common_residues_map[pair])
                    intermediate = common - {start_node, end_node}
                    category_intermediate_residues.update(intermediate)
            
            sorted_intermediates = sorted([int(r) for r in category_intermediate_residues if r.isdigit()])
            
            f.write("### Shared Intermediate Residue Analysis\n\n")
            f.write(f"- **Total Unique Shared Intermediate Residues:** {len(sorted_intermediates)}\n")
            if sorted_intermediates:
                f.write(f"- **Shared Intermediate Residues List:** `{', '.join(map(str, sorted_intermediates))}`\n\n")
            else:
                f.write("- **Shared Intermediate Residues List:** None\n\n")

            shared_residue_analysis.append({
                'Category': category,
                'Shared_Intermediate_Residues_Count': len(sorted_intermediates),
                'Shared_Intermediate_Residues_List': ', '.join(map(str, sorted_intermediates))
            })

    # Save CSVs
    jaccard_df = pd.DataFrame(jaccard_results)
    jaccard_csv_path = os.path.join(output_dir, 'residue_pair_jaccard_similarity.csv')
    jaccard_df.to_csv(jaccard_csv_path, index=False)
    logging.info(f"Saved Jaccard similarity data to {jaccard_csv_path}")

    shared_residue_df = pd.DataFrame(shared_residue_analysis)
    shared_csv_path = os.path.join(output_dir, 'shared_residue_analysis.csv')
    shared_residue_df.to_csv(shared_csv_path, index=False)
    logging.info(f"Saved shared residue analysis to {shared_csv_path}")

    logging.info(f"Report generated at {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Run statistical analysis on protein path data.")
    parser.add_argument('--optimal-paths', type=str, default='analysis_results/reports/optimal_paths_details.md', help='Path to the optimal paths details file.')
    parser.add_argument('--extended-report', type=str, default='analysis_results/reports/extended_report.md', help='Path to the extended report file.')
    args = parser.parse_args()

    report_dir = 'analysis_results/reports'
    plot_dir = 'analysis_results/plots/statistical_analysis'
    log_dir = 'logs'
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"statistical_analysis_{datetime.now().strftime('%Y-%m-%d')}.log")

    try:
        with open(args.optimal_paths, 'r') as f:
            optimal_paths_content = f.read()
        with open(args.extended_report, 'r') as f:
            extended_report_content = f.read()

        optimal_paths_df = parse_optimal_paths(optimal_paths_content)
        common_residues_map = parse_extended_report(extended_report_content)
        
        run_analysis(optimal_paths_df, common_residues_map, report_dir, plot_dir, log_file)
        
    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e.filename}")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
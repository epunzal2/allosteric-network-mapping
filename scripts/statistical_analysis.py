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
from collections import Counter
import csv

def load_residue_name_map(filepath):
    """Loads residue ID to name mapping from a CSV file."""
    name_map = {}
    try:
        with open(filepath, mode='r', newline='') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                name_map[row['resid1']] = row['res1']
                name_map[row['resid2']] = row['res2']
    except FileNotFoundError:
        logging.warning(f"Residue name map file not found: {filepath}")
    return name_map

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
                system, residue_pair, _, nodes, path_length, _, optimal_path = parts
                data.append([current_category, residue_pair, system, nodes, path_length, optimal_path])
    
    df = pd.DataFrame(data, columns=['Category', 'Residue Pair', 'System', 'Nodes', 'Path Length', 'Optimal Path Residues'])
    df['Nodes'] = pd.to_numeric(df['Nodes'], errors='coerce')
    df['Path Length'] = pd.to_numeric(df['Path Length'], errors='coerce')
    return df

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

def plot_jaccard_similarity(data, category, plot_dir, name_map):
    """
    Generates and saves a bar plot for Jaccard similarities.
    """
    def get_label(pair):
        res1, res2 = pair.split('-')
        name1 = name_map.get(res1, res1)
        name2 = name_map.get(res2, res2)
        return f"{name1}{res1}-{name2}{res2}"

    data['Residue Pair Label'] = data['Residue Pair'].apply(get_label)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Jaccard Similarity', y='Residue Pair Label', data=data, palette='viridis')
    plt.title(f'Jaccard Similarity of Optimal Paths for {category}')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Residue Pair')
    plt.tight_layout()
    plot_filename = f"jaccard_similarity_{category.replace(' ', '_')}.png"
    plot_filepath = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()
    return plot_filepath

def run_analysis(data_by_category, common_residues_map, output_dir, plot_dir, log_file, name_map):
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
        f.write("This report presents a statistical comparison of 'Nodes' and 'Path Length' between WT and Mutant systems, alongside Jaccard similarity and shared residue analyses.\n\n")
        f.write("### Understanding the Statistics\n\n")
        f.write("**Null Hypothesis:** For each test, the null hypothesis (H0) states that there is no significant difference between the distributions of the WT and Mutant groups. A low p-value (typically < 0.05) suggests that we can reject this hypothesis, indicating a statistically significant difference.\n\n")
        f.write("**Not Significant:** A non-significant result (p-value >= 0.05) means that any observed differences between the groups are likely due to random chance, and there is not enough evidence to conclude that the groups are truly different.\n\n")

        for category, df in data_by_category.items():
            f.write(f"## Category: {category}\n\n")
            logging.info(f"Processing category: {category}")

            wt_data = df[df['System'] == 'WT'].dropna()
            mutant_data = df[df['System'] == 'Mutant'].dropna()
            
            f.write(f"Number of entries: WT = {len(wt_data)}, Mutant = {len(mutant_data)}\n\n")

            # Original Statistical Analysis
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
                else:
                    results.append([col, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

            f.write("| Metric | Mann-Whitney U Statistic | Mann-Whitney U p-value | KS Statistic | KS p-value | Cohen's d |\n")
            f.write("|--------|--------------------------|------------------------|--------------|------------|-----------|\n")
            for row in results:
                f.write(f"| {'# Residues' if row[0] == 'Nodes' else row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |\n")
            f.write("\n")

            f.write("### Interpretation\n\n")
            for col, interp in interpretations.items():
                f.write(f"**For {'# Residues' if col == 'Nodes' else col}:**\n\n")
                f.write(f"- **Mann-Whitney U Test:** {interp['mwu']}\n")
                f.write(f"- **Kolmogorov-Smirnov Test:** {interp['ks']}\n")
                f.write(f"- **Effect Size (Cohen's d):** {interp['cohen_d']}\n\n")

            if "high contact" in category.lower():
                residue_pairs = df['Residue Pair'].unique()
                category_jaccard_data = []

                for pair in residue_pairs:
                    wt_pair_row = wt_data[wt_data['Residue Pair'] == pair]
                    mutant_pair_row = mutant_data[mutant_data['Residue Pair'] == pair]

                    if not wt_pair_row.empty and not mutant_pair_row.empty:
                        wt_path_str = wt_pair_row.iloc[0]['Optimal Path Residues']
                        mutant_path_str = mutant_pair_row.iloc[0]['Optimal Path Residues']

                        if 'N/A' in wt_path_str or 'N/A' in mutant_path_str:
                            continue

                        wt_residues = set(wt_path_str.split(' -> '))
                        mutant_residues = set(mutant_path_str.split(' -> '))
                        
                        intersection = len(wt_residues.intersection(mutant_residues))
                        union = len(wt_residues.union(mutant_residues))
                        
                        jaccard_similarity = intersection / union if union > 0 else 0
                        
                        jaccard_results.append({'Category': category, 'Residue Pair': pair, 'Jaccard Similarity': jaccard_similarity})
                        category_jaccard_data.append({'Residue Pair': pair, 'Jaccard Similarity': jaccard_similarity})

                if category_jaccard_data:
                    jaccard_df_category = pd.DataFrame(category_jaccard_data).sort_values(by='Jaccard Similarity', ascending=False)
                    highest_sim = jaccard_df_category.iloc[0]
                    lowest_sim = jaccard_df_category.iloc[-1]

                    f.write("### Jaccard Similarity Analysis\n\n")
                    f.write(f"- **Highest Similarity Pair:** {highest_sim['Residue Pair']} ({highest_sim['Jaccard Similarity']:.4f})\n")
                    f.write(f"- **Lowest Similarity Pair:** {lowest_sim['Residue Pair']} ({lowest_sim['Jaccard Similarity']:.4f})\n\n")

                    plot_filepath = plot_jaccard_similarity(jaccard_df_category, category, plot_dir, name_map)
                    relative_plot_path = os.path.relpath(plot_filepath, output_dir)
                    f.write(f"![Jaccard Similarity for {category}]({relative_plot_path})\n\n")

                # Shared Intermediate Residue Analysis
                category_intermediate_residue_counts = Counter()
                for pair in residue_pairs:
                    if pair in common_residues_map:
                        start_node, end_node = pair.split('-')
                        common = set(common_residues_map[pair])
                        intermediate = common - {start_node, end_node}
                        category_intermediate_residue_counts.update(intermediate)
                
                unique_intermediates = sorted([int(r) for r in category_intermediate_residue_counts.keys() if r.isdigit()])
                
                f.write("### Shared Intermediate Residue Analysis\n\n")
                f.write(f"- **Total Unique Shared Intermediate Residues:** {len(unique_intermediates)}\n")
                if unique_intermediates:
                    f.write(f"- **Shared Intermediate Residues List:** `{', '.join(map(str, unique_intermediates))}`\n\n")
                else:
                    f.write("- **Shared Intermediate Residues List:** None\n\n")

                f.write("#### Most Frequent Shared Intermediate Residues\n\n")
                if category_intermediate_residue_counts:
                    top_5 = category_intermediate_residue_counts.most_common(5)
                    for residue, count in top_5:
                        f.write(f"- **Residue {residue}:** Appeared in {count} common paths.\n")
                    f.write("\n")
                else:
                    f.write("No shared intermediate residues to analyze for frequency.\n\n")

                shared_residue_analysis.append({
                    'Category': category,
                    'Shared_Intermediate_Residues_Count': len(unique_intermediates),
                    'Shared_Intermediate_Residues_List': ', '.join(map(str, unique_intermediates))
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
    parser.add_argument('--residue-map-wt', type=str, default='Data/residues_to_test_WT.csv', help='Path to the WT residue map CSV.')
    parser.add_argument('--residue-map-mutant', type=str, default='Data/residues_to_test_Mutant.csv', help='Path to the Mutant residue map CSV.')
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

        name_map_wt = load_residue_name_map(args.residue_map_wt)
        name_map_mutant = load_residue_name_map(args.residue_map_mutant)
        name_map = {**name_map_wt, **name_map_mutant}

        optimal_paths_df = parse_optimal_paths(optimal_paths_content)
        
        # Create a combined category for high contact
        high_contact_df = optimal_paths_df[optimal_paths_df['Category'].str.contains("high contact", case=False)]
        combined_df = high_contact_df.copy()
        combined_df['Category'] = 'Combined High Contact'
        
        # Pass a dictionary of dataframes to run_analysis
        all_dfs = {cat: df for cat, df in optimal_paths_df.groupby('Category')}
        if not combined_df.empty:
            all_dfs['Combined High Contact'] = combined_df

        common_residues_map = parse_extended_report(extended_report_content)
        
        run_analysis(all_dfs, common_residues_map, report_dir, plot_dir, log_file, name_map)
        
    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e.filename}")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
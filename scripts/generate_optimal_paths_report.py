import os
import re
import pandas as pd
from collections import defaultdict
import logging
from datetime import datetime

# Configure logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = datetime.now().strftime('generate_optimal_paths_report_%Y-%m-%d.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(log_dir, log_filename),
    filemode='w'
)

def load_residue_categories(wt_csv_path, mutant_csv_path):
    """Loads residue pair categories from CSV files."""
    categories = {}
    try:
        wt_df = pd.read_csv(wt_csv_path)
        for _, row in wt_df.iterrows():
            residue_pair = f"{row['resid1']}-{row['resid2']}"
            categories[residue_pair] = row['notes']

        mutant_df = pd.read_csv(mutant_csv_path)
        for _, row in mutant_df.iterrows():
            residue_pair = f"{row['resid1']}-{row['resid2']}"
            if residue_pair not in categories:
                 categories[residue_pair] = row['notes']

    except FileNotFoundError as e:
        logging.error(f"Error loading CSV file: {e}")
    return categories

def parse_analysis_report(file_path):
    """Parses an analysis_report.txt file to extract the optimal path and other metrics."""
    data = {
        "optimal_path": "N/A",
        "nodes": "N/A",
        "path_length": "N/A",
        "avg_correlation": "N/A"
    }
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            path_match = re.search(r"Optimal path \(Residue IDs\):\s*(.*)", content)
            if path_match:
                data["optimal_path"] = path_match.group(1).strip()

            nodes_match = re.search(r"Optimal path found: (\d+) nodes", content)
            if nodes_match:
                data["nodes"] = nodes_match.group(1).strip()

            length_match = re.search(r"Path length \(sum of 1-\|corr\| weights\):\s*([\d\.]+)", content)
            if length_match:
                data["path_length"] = length_match.group(1).strip()

            correlation_match = re.search(r"Average \|correlation\| along path:\s*([\d\.]+)", content)
            if correlation_match:
                data["avg_correlation"] = correlation_match.group(1).strip()

    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
    return data

def generate_report(wt_dir, mutant_dir, output_file, categories):
    """Generates the markdown report."""
    results = defaultdict(list)

    for base_dir, system_name in [(wt_dir, 'WT'), (mutant_dir, 'Mutant')]:
        for root, _, files in os.walk(base_dir):
            if 'analysis_report.txt' in files:
                report_path = os.path.join(root, 'analysis_report.txt')
                logging.info(f"Processing {report_path}")
                dir_name = os.path.basename(root)
                
                parts = dir_name.split('_')
                if len(parts) < 2:
                    continue

                residue_pair_str = f"{parts[0]}-{parts[1]}"
                analysis_params = '_'.join(parts[2:])
                
                parsed_data = parse_analysis_report(report_path)
                
                category = categories.get(residue_pair_str, 'Uncategorized')

                results[category].append({
                    'System': system_name,
                    'Residue Pair': residue_pair_str,
                    'Analysis Params': analysis_params,
                    'Optimal Path Residues': parsed_data["optimal_path"],
                    'Nodes': parsed_data["nodes"],
                    'Path Length': parsed_data["path_length"],
                    'Avg Correlation': parsed_data["avg_correlation"]
                })

    with open(output_file, 'w') as f:
        f.write("# Full Optimal Path Details by Category\n\n")
        
        sorted_categories = sorted(results.keys())

        for category in sorted_categories:
            f.write(f"## Category: {category}\n\n")
            f.write("| System | Residue Pair | Analysis Params | Nodes | Path Length | Avg Correlation | Optimal Path Residues |\n")
            f.write("|--------|--------------|-----------------|-------|-------------|-----------------|-------------------------|\n")
            
            sorted_rows = sorted(results[category], key=lambda x: (x['Residue Pair'], x['System']))

            for row in sorted_rows:
                f.write(f"| {row['System']} | {row['Residue Pair']} | {row['Analysis Params']} | {row['Nodes']} | {row['Path Length']} | {row['Avg Correlation']} | {row['Optimal Path Residues']} |\n")
            f.write("\n")

if __name__ == "__main__":
    logging.info("Starting script to generate optimal paths report.")
    
    WT_DATA_DIR = 'analysis_results/Data/AF2_LM211_WT'
    MUTANT_DATA_DIR = 'analysis_results/Data/AF2_LM2_Y138H_11_Mutant'
    OUTPUT_MD = 'analysis_results/optimal_paths_details.md'
    WT_CSV = 'Data/residues_to_test_WT.csv'
    MUTANT_CSV = 'Data/residues_to_test_Mutant.csv'

    residue_categories = load_residue_categories(WT_CSV, MUTANT_CSV)
    generate_report(WT_DATA_DIR, MUTANT_DATA_DIR, OUTPUT_MD, residue_categories)
    
    logging.info(f"Report generated at {OUTPUT_MD}")
    print(f"Report generated at {OUTPUT_MD}")
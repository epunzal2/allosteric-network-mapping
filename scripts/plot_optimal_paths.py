import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd

def setup_logging():
    """Sets up logging to a date-stamped file in the logs directory."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = datetime.now().strftime('%Y-%m-%d') + '_plot_generation.log'
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_subunit(residue_id):
    """Maps residue ID to subunit and color."""
    residue_id = int(residue_id)
    if 1 <= residue_id <= 305:
        return "alpha", "red"
    elif 306 <= residue_id <= 612:
        return "beta", "blue"
    elif 613 <= residue_id <= 917:
        return "gamma", "green"
    return "unknown", "gray"

def read_residue_names(file_path, logger):
    """Reads residue names from a CSV file."""
    logger.info(f"Reading residue names from {file_path}")
    residue_map = {}
    try:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            residue_map[str(row['resid2'])] = row['res2']
    except Exception as e:
        logger.error(f"Could not read or parse the residue names file: {e}")
    return residue_map

def parse_markdown_for_paths(file_path, logger):
    """Parses the Markdown file to extract optimal path data."""
    logger.info(f"Parsing data for detailed path plots from {file_path}")
    data = {}
    current_category = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("## Category:"):
                current_category = line.replace("## Category:", "").strip()
                data[current_category] = []
            elif current_category and line.startswith("|") and "Optimal Path Residues" not in line and "------" not in line:
                parts = [part.strip() for part in line.split("|")[1:-1]]
                if len(parts) == 7:
                    system, residue_pair_str, _, _, _, _, optimal_path_str = parts
                    if optimal_path_str.lower() == "n/a":
                        path_residues = []
                    else:
                        path_residues = [res.strip() for res in optimal_path_str.split("->")]
                    
                    found_pair = False
                    for entry in data[current_category]:
                        if entry["residue_pair"] == residue_pair_str:
                            if system.lower() == "wt":
                                entry["wt_path"] = path_residues
                            elif system.lower() == "mutant":
                                entry["mutant_path"] = path_residues
                            found_pair = True
                            break
                    if not found_pair:
                        new_entry = {"residue_pair": residue_pair_str, "wt_path": [], "mutant_path": []}
                        if system.lower() == "wt":
                            new_entry["wt_path"] = path_residues
                        elif system.lower() == "mutant":
                            new_entry["mutant_path"] = path_residues
                        data[current_category].append(new_entry)
    logger.info("Finished parsing for detailed path plots.")
    return data

def parse_markdown_for_bar_graphs(file_path, logger):
    """Parses the Markdown file to extract optimal path data for bar graphs."""
    logger.info(f"Parsing data for bar graphs from {file_path}")
    data = {}
    current_category = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("## Category:"):
                current_category = line.replace("## Category:", "").strip()
                data[current_category] = []
                logger.info(f"Found category: {current_category}")
            elif current_category and line.startswith("|") and "Optimal Path Residues" not in line and "------" not in line:
                parts = [part.strip() for part in line.split("|")[1:-1]]
                if len(parts) == 7:
                    system, residue_pair, _, nodes_str, _, _, _ = parts
                    try:
                        nodes = int(nodes_str)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse nodes value '{nodes_str}' for {residue_pair} in {system}. Skipping.")
                        continue
                    
                    data[current_category].append({
                        "System": system,
                        "Residue Pair": residue_pair,
                        "Nodes": nodes
                    })
    logger.info("Finished parsing for bar graphs.")
    return data

def plot_paths_for_category(category_name, path_data_list, output_dir, logger):
    """Plots optimal paths for a given category."""
    if not path_data_list:
        logger.info(f"No data to plot for category: {category_name}")
        return

    num_pairs = len(path_data_list)
    
    max_len = 0
    for pair_data in path_data_list:
        max_len = max(max_len, len(pair_data.get("wt_path", [])), len(pair_data.get("mutant_path", [])))
    
    fig_width = max(20, max_len * 0.7)
    fig_height = max(10, num_pairs * 3.5)

    fig, axes = plt.subplots(num_pairs, 1, figsize=(fig_width, fig_height), squeeze=False)
    fig.suptitle(f"Optimal Paths: {category_name}", fontsize=16, y=0.99)

    for i, pair_data in enumerate(path_data_list):
        ax = axes[i, 0]
        residue_pair_label = pair_data["residue_pair"]
        wt_path = pair_data.get("wt_path", [])
        mutant_path = pair_data.get("mutant_path", [])

        y_offset_wt = 0.6
        y_offset_mutant = 0.2
        
        plotted_wt = False
        if wt_path:
            plotted_wt = True
            prev_subunit_wt = None
            for j, res_id_str in enumerate(wt_path):
                if not res_id_str: continue
                subunit, color = get_subunit(res_id_str)
                ax.plot(j, y_offset_wt, 'o', markersize=8, color=color)
                ax.text(j, y_offset_wt - 0.05, res_id_str, ha='center', va='top', fontsize=7)
                if prev_subunit_wt and subunit != prev_subunit_wt:
                    ax.text((j - 0.5), y_offset_wt + 0.1, f"{wt_path[j-1]}-{res_id_str}", color='purple', fontsize=12, ha='center', va='bottom', rotation=45)
                prev_subunit_wt = subunit
            if len(wt_path) > 1:
                 for k in range(len(wt_path) - 1):
                    ax.plot([k, k+1], [y_offset_wt, y_offset_wt], '-', color='lightgray', linewidth=0.8)
            ax.text(-0.5, y_offset_wt, "WT", ha='right', va='center', fontsize=9, color='black', fontweight='bold')

        plotted_mutant = False
        if mutant_path:
            plotted_mutant = True
            prev_subunit_mutant = None
            for j, res_id_str in enumerate(mutant_path):
                if not res_id_str: continue
                subunit, color = get_subunit(res_id_str)
                ax.plot(j, y_offset_mutant, 'o', markersize=8, color=color)
                ax.text(j, y_offset_mutant - 0.05, res_id_str, ha='center', va='top', fontsize=7)
                if prev_subunit_mutant and subunit != prev_subunit_mutant:
                     ax.text((j - 0.5), y_offset_mutant + 0.1, f"{mutant_path[j-1]}-{res_id_str}", color='purple', fontsize=12, ha='center', va='bottom', rotation=45)
                prev_subunit_mutant = subunit
            if len(mutant_path) > 1:
                for k in range(len(mutant_path) - 1):
                    ax.plot([k, k+1], [y_offset_mutant, y_offset_mutant], '-', color='lightgray', linewidth=0.8)
            ax.text(-0.5, y_offset_mutant, "Mutant", ha='right', va='center', fontsize=9, color='black', fontweight='bold')

        current_max_len = max(len(wt_path), len(mutant_path))
        ax.set_xlim(-1, max(1, max_len))
        
        if plotted_wt and plotted_mutant:
            ax.set_yticks([y_offset_mutant, y_offset_wt])
            ax.set_yticklabels(["Mutant", "WT"])
        elif plotted_wt:
            ax.set_yticks([y_offset_wt])
            ax.set_yticklabels(["WT"])
        elif plotted_mutant:
            ax.set_yticks([y_offset_mutant])
            ax.set_yticklabels(["Mutant"])
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_title(f"Residue Pair: {residue_pair_label}", fontsize=10, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Alpha (1-305)', markersize=10, markerfacecolor='red'),
        plt.Line2D([0], [0], marker='o', color='w', label='Beta (306-612)', markersize=10, markerfacecolor='blue'),
        plt.Line2D([0], [0], marker='o', color='w', label='Gamma (613-917)', markersize=10, markerfacecolor='green'),
        plt.Line2D([0], [0], marker='o', color='w', label='Unknown', markersize=10, markerfacecolor='gray')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=12)

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename_map = {
        "high contact WT": "wt_trimer",
        "high contact Mutant": "mutant_trimer",
        "high contact both": "contact_map",
    }
    filename_base = filename_map.get(category_name, category_name.replace(' ', '_').lower())
    plot_filename = os.path.join(output_dir, f"{filename_base}_optimal_paths.png")
    plt.savefig(plot_filename, dpi=600)
    plt.close(fig)
    logger.info(f"Saved detailed path plot: {plot_filename}")

def generate_bar_graphs(data, output_dir, residue_map, logger):
    """Generates and saves bar graphs for each category."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    for category, records in data.items():
        if not records:
            logger.info(f"No data for category '{category}', skipping bar graph generation.")
            continue

        logger.info(f"Generating bar graph for category: {category}")
        
        df = pd.DataFrame(records)
        
        pivot_df = df.pivot(index='Residue Pair', columns='System', values='Nodes').fillna(0)
        
        # Create new labels for the x-axis
        new_labels = []
        for label in pivot_df.index:
            res2_id = label.split('-')[1]
            res2_name = residue_map.get(res2_id, '')
            new_labels.append(f"{res2_name} {res2_id}")

        fig, ax = plt.subplots(figsize=(12, 8))
        
        bar_width = 0.35
        index = np.arange(len(pivot_df.index))

        ax.bar(index - bar_width/2, pivot_df.get('Mutant', [0]*len(index)), bar_width, label='Mutant (HIS)')
        ax.bar(index + bar_width/2, pivot_df.get('WT', [0]*len(index)), bar_width, label='WT (TYR)')

        ax.set_xlabel('Residue Pair')
        ax.set_ylabel('Path Length (# Residues)')
        ax.set_title(f'Path Lengths for {category}')
        ax.set_xticks(index)
        ax.set_xticklabels(new_labels, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        
        filename_cat = re.sub(r'[^a-zA-Z0-9_]', '', category.replace(' ', '_'))
        plot_filename = os.path.join(output_dir, f"{filename_cat}_path_lengths.png")
        
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)
        logger.info(f"Saved bar graph to {plot_filename}")

def main():
    """Main function to parse data and generate plots."""
    logger = setup_logging()
    markdown_file = "analysis_results/optimal_paths_details.md"
    residue_name_file = "Data/residues_to_test_WT.csv"
    
    residue_map = read_residue_names(residue_name_file, logger)

    # For detailed path plots
    path_plot_dir = "analysis_results/plots"
    path_data = parse_markdown_for_paths(markdown_file, logger)
    for category, data in path_data.items():
        if data:
            plot_paths_for_category(category, data, path_plot_dir, logger)

    # For bar graphs
    bar_graph_dir = "analysis_results/plots/bar_graph/"
    bar_graph_data = parse_markdown_for_bar_graphs(markdown_file, logger)
    generate_bar_graphs(bar_graph_data, bar_graph_dir, residue_map, logger)
    
    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main()
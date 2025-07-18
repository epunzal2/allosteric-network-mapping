#!/usr/bin/env python3

import argparse
import re
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# File handler will be added in main() after args are parsed, or use a default log dir here

def find_analysis_reports(base_dir: Path) -> list[Path]:
    """
    Recursively finds all 'analysis_report.txt' files within the base directory.
    """
    if not base_dir.is_dir():
        logger.warning(f"Base directory '{base_dir}' does not exist or is not a directory.")
        return []
    logger.info(f"Searching for 'analysis_report.txt' files in '{base_dir}'.")
    reports = sorted(list(base_dir.rglob("analysis_report.txt")))
    logger.info(f"Found {len(reports)} analysis report files.")
    return reports

def parse_analysis_report(report_path: Path) -> tuple[set[int], set[int]]:
    """
    Parses an analysis_report.txt file to extract optimal path residue IDs
    and top 10 critical residue IDs.
    """
    logger.info(f"Parsing analysis report: {report_path}")
    optimal_path_residues = set()
    critical_residues = set()

    try:
        with open(report_path, 'r') as f:
            content = f.read()
        logger.info(f"Successfully read content from {report_path}.")

        # Extract Optimal Path
        optimal_path_match = re.search(r"Optimal path \(Residue IDs\):\s*(.*)", content)
        if optimal_path_match:
            logger.info(f"Found 'Optimal path (Residue IDs)' section in {report_path}.")
            path_str = optimal_path_match.group(1)
            if path_str.strip() and path_str.strip().lower() != 'n/a':
                try:
                    optimal_path_residues = set(map(int, re.findall(r'\d+', path_str)))
                    logger.info(f"Extracted {len(optimal_path_residues)} optimal path residues from '{path_str}' in {report_path}.")
                except ValueError:
                    logger.warning(f"Could not parse optimal path residues from '{path_str}' in {report_path}")
            else:
                logger.info(f"'Optimal path (Residue IDs)' section is present but empty or 'N/A' in {report_path}.")
        else:
            logger.info(f"'Optimal path (Residue IDs)' section NOT found in {report_path}.")

        # Extract Top 10 Critical Residues
        # Regex looks for lines like: "  1. Residue GLU176 (Node 175): 0.1082"
        # or "1. Residue ILE101 (Node 100): 0.08"
        # or "Node ID: 175, Residue: GLU176, Betweenness Centrality: 0.1082"
        
        # Try to find the section header first
        critical_residues_section_match = re.search(
            r"Top \d+ critical residues.*?:\n((?:\s*(?:\d+\.\s+Residue\s+\w+\d+\s+\(Node\s+\d+\)|\s*Node ID:\s*\d+).*?\n)*)",
            content,
            re.IGNORECASE | re.MULTILINE
        )

        if critical_residues_section_match:
            logger.info(f"Found 'Top critical residues' section in {report_path}.")
            section_content = critical_residues_section_match.group(1)
            # Regex for "  1. Residue GLU176 (Node 175): 0.1082" or "1. Residue ILE101 (Node 100): 0.08"
            matches_format1 = re.findall(r"^\s*\d+\.\s+Residue\s+\w+\d+\s+\(Node\s+(\d+)\):", section_content, re.MULTILINE)
            # Regex for "Node ID: 175, Residue: GLU176, Betweenness Centrality: 0.1082"
            matches_format2 = re.findall(r"^\s*Node ID:\s*(\d+),", section_content, re.MULTILINE)
            
            parsed_ids_from_section = set()
            for node_id_str in matches_format1 + matches_format2:
                try:
                    parsed_ids_from_section.add(int(node_id_str))
                except ValueError:
                    logger.warning(f"Could not parse critical residue Node ID '{node_id_str}' from section in {report_path}")
            critical_residues.update(parsed_ids_from_section)
            if parsed_ids_from_section:
                logger.info(f"Extracted {len(parsed_ids_from_section)} critical residues from 'Top critical residues' section in {report_path}.")
            else:
                logger.info(f"'Top critical residues' section found but no residues extracted (or section empty) in {report_path}.")
        else:
            logger.info(f"'Top critical residues' section NOT found with primary regex in {report_path}.")
            # Fallback if the specific header isn't found, try to find Node IDs more broadly
            # This is less precise and might pick up other numbers if the format is very different
            all_node_lines = re.findall(r"\(Node\s+(\d+)\)", content)
            for node_id_str in all_node_lines:
                 try:
                    # This is a heuristic; ideally, the format is consistent.
                    # We add this as a fallback, but it's less reliable.
                    # For now, we'll stick to the more structured parsing above.
                    pass # critical_residues.add(int(node_id_str))
                 except ValueError:
                    pass


    except FileNotFoundError:
        logger.error(f"File not found {report_path}")
    except Exception as e:
        logger.error(f"Error parsing file {report_path}: {e}", exc_info=True)

    logger.info(f"Finished parsing {report_path}. Optimal: {len(optimal_path_residues)} residues, Critical: {len(critical_residues)} residues.")
    return optimal_path_residues, critical_residues

def extract_input_residues_from_path(relative_path_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts input residue1 and residue2 strings from a relative simulation path.
    Example path: Data/AF2_LM211_WT/calcium/101-225_dmd_oec_cb_cf0.3_multi/...
    or Data/AF2_LM2_Y138H_11_Mutant/calcium/101_43_displacement...
    Returns (res1_str, res2_str) or (None, None).
    """
    try:
        parts = Path(relative_path_str).parts
        # Expected structure: Data / ProteinType / IonType / ResiduePairInfo_... / ...
        if len(parts) > 3:
            residue_pair_dir_name = parts[3] # e.g., "101-225_dmd..." or "101_43_disp..."
            # Regex to find patterns like "101-225" or "101_43" at the beginning of the string
            match = re.match(r"^(\d+(?:-\d+)?)[_-](\d+(?:-\d+)?)", residue_pair_dir_name)
            if match:
                res1_str = match.group(1)
                res2_str = match.group(2)
                logger.debug(f"Extracted input residues '{res1_str}' and '{res2_str}' from path component '{residue_pair_dir_name}'.")
                return res1_str, res2_str
            else:
                logger.warning(f"Could not extract residue pair from path component: {residue_pair_dir_name} in path {relative_path_str}")
        else:
            logger.warning(f"Path {relative_path_str} is too short to extract residue pair information.")
    except Exception as e:
        logger.error(f"Error extracting input residues from path '{relative_path_str}': {e}", exc_info=True)
    return None, None

def generate_markdown_report(
    all_collected_data: list[dict],
    wt_summary_entries: list[dict],
    mutant_summary_entries: list[dict],
    output_path: Path,
    mutations_df: pd.DataFrame,
    searched_residues: set[int]
):
    """
    Generates a Markdown report from the collected data, including summary statistics.
    """
    logger.info(f"Generating Markdown report at: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        logger.debug(f"Opened {output_path} for writing.")
        f.write(f"# Untested Mutations Involvement Analysis Report\n\n")
        
        f.write("## Analyzed Mutations\n\n")
        f.write("| Residue ID | Full Original Label |\n")
        f.write("|------------|---------------------|\n")
        for _, row in mutations_df.iterrows():
            f.write(f"| {row['resid']} | {row['full_orig_label']} |\n")
        f.write("\n")

        f.write("## Detailed Results\n\n")
        f.write("| Simulation Path                                  | Found in Optimal Path (Residues) | Found in Critical Residues (Residues) |\n")
        f.write("|--------------------------------------------------|------------------------------------|---------------------------------------|\n")

        if not all_collected_data:
            logger.info("No analysis data to report for detailed results.")
            f.write("| No analysis reports found or processed.          | N/A                                | N/A                                   |\n")
        else:
            logger.info(f"Writing {len(all_collected_data)} entries to the detailed results table.")
            for entry in all_collected_data:
                sim_path = entry["simulation_path"]
                
                optimal_found_str = "No"
                if entry["optimal_path_found_residues"]:
                    optimal_found_str = f"Yes ({', '.join(map(str, sorted(list(entry['optimal_path_found_residues']))))})"
                
                critical_found_str = "No"
                if entry["critical_path_found_residues"]:
                    critical_found_str = f"Yes ({', '.join(map(str, sorted(list(entry['critical_path_found_residues']))))})"
                
                f.write(f"| {sim_path} | {optimal_found_str} | {critical_found_str} |\n")
        
        f.write("\n\n## Summary Statistics\n\n")

        for summary_type, summary_data in [("WT", wt_summary_entries), ("Mutant", mutant_summary_entries)]:
            f.write(f"### {summary_type} Simulations\n\n")
            if not summary_data:
                f.write(f"- No {summary_type} simulation data found for summary.\n")
                f.write(f"- **Percentage of pairs with target mutations (as intermediate) in optimal path:** 0.0% (0 out of 0 pairs)\n\n")
                continue

            pairs_with_target_as_intermediate_list = []
            count_has_target_as_intermediate_in_optimal = 0
            
            for item in summary_data:
                if item['has_target_in_optimal_as_intermediate']:
                    count_has_target_as_intermediate_in_optimal += 1
                    pair_display = item['input_residue_pair_str']
                    
                    if item['is_overlap_with_input']:
                        pair_display += " (N/A - target in input)"
                    
                    found_intermediate_set = item.get('intermediate_target_residues_found', set())
                    if found_intermediate_set:
                        found_residues_str = ', '.join(map(str, sorted(list(found_intermediate_set))))
                        pair_display += f" (Found as intermediate: {found_residues_str})"
                        
                    pairs_with_target_as_intermediate_list.append(pair_display)
            
            f.write(f"- **Residue pairs with target mutations in optimal path (as intermediate):**\n")
            if pairs_with_target_as_intermediate_list:
                for pair_str in sorted(list(set(pairs_with_target_as_intermediate_list))):
                    f.write(f"  - {pair_str}\n")
            else:
                f.write("  - None\n")
            
            total_pairs_in_category = len(summary_data)
            percentage_all_pairs = (count_has_target_as_intermediate_in_optimal / total_pairs_in_category) * 100 if total_pairs_in_category > 0 else 0
            f.write(f"- **Percentage of pairs with target mutations (as intermediate) in optimal path (all {summary_type} pairs):** {percentage_all_pairs:.1f}% ({count_has_target_as_intermediate_in_optimal} out of {total_pairs_in_category} pairs)\n")

            non_input_overlap_entries = [item for item in summary_data if not item['is_overlap_with_input']]
            count_intermediate_in_non_input_overlap = 0
            for item in non_input_overlap_entries:
                if item['has_target_in_optimal_as_intermediate']:
                    count_intermediate_in_non_input_overlap += 1
            
            total_non_input_overlap_pairs = len(non_input_overlap_entries)
            percentage_non_input_overlap = (count_intermediate_in_non_input_overlap / total_non_input_overlap_pairs) * 100 if total_non_input_overlap_pairs > 0 else 0
            f.write(f"- **Percentage of pairs with target mutations (as intermediate) in optimal path (excluding pairs where target is input):** {percentage_non_input_overlap:.1f}% ({count_intermediate_in_non_input_overlap} out of {total_non_input_overlap_pairs} pairs)\n\n")

        f.write("\n\n---\nReport generated by `analyze_untested_mutations_involvement.py`.\n")
    logger.info(f"Markdown report successfully generated at: {output_path.resolve()}")


def main():
    # Initial log to console before file handler is set up
    logger.info("Starting script execution: analyze_untested_mutations_involvement.py (pre-file-log setup)")

    PROJECT_ROOT_ALLOSTERIC = Path(__file__).resolve().parent.parent
    
    parser = argparse.ArgumentParser(
        description="Analyze residue involvement in optimal paths and critical residues from simulation reports."
    )
    parser.add_argument(
        "--mutations_csv",
        type=str,
        default="Data/untested_mutations.csv",
        help="Path to the CSV file containing mutation data. Defaults to 'Data/untested_mutations.csv'."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="analysis_results/Data",
        help="Base directory containing simulation results. Defaults to 'analysis_results/Data/' relative to the project root."
    )
    parser.add_argument(
        "--output_report_name",
        type=str,
        default="reports/untested_mutations_analysis_report.md",
        help="Name of the output Markdown report file."
    )
    parser.add_argument(
        "--output_report_dir",
        type=str,
        default="analysis_results",
        help="Directory to save the output Markdown report. Defaults to 'analysis_results/' relative to the project root."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files. Defaults to 'logs/' relative to the project root."
    )

    args = parser.parse_args()

    log_file_dir = PROJECT_ROOT_ALLOSTERIC / args.log_dir
    base_data_dir = PROJECT_ROOT_ALLOSTERIC / args.data_dir
    output_report_dir_path = PROJECT_ROOT_ALLOSTERIC / args.output_report_dir
    output_report_path = output_report_dir_path / args.output_report_name
    mutations_csv_path = PROJECT_ROOT_ALLOSTERIC / args.mutations_csv

    log_file_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"analyze_untested_mutations_involvement_{timestamp}.log"
    log_file_path = log_file_dir / log_file_name

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Script execution started. Logging to console and to: {log_file_path.resolve()}")
    logger.info(f"Arguments parsed: {args}")

    try:
        mutations_df = pd.read_csv(mutations_csv_path)
        target_residues = set(mutations_df['resid'].astype(int))
        logger.info(f"Successfully loaded {len(target_residues)} mutations from {mutations_csv_path}.")
        logger.info(f"Target residues for search: {target_residues}")
    except FileNotFoundError:
        logger.error(f"Mutations CSV file not found at: {mutations_csv_path}")
        return
    except KeyError:
        logger.error(f"'resid' column not found in {mutations_csv_path}. Please ensure the CSV has a 'resid' column.")
        return
    except Exception as e:
        logger.error(f"Error reading or processing mutations CSV {mutations_csv_path}: {e}")
        return

    analysis_report_files = find_analysis_reports(base_data_dir)

    if not analysis_report_files:
        logger.warning(f"No 'analysis_report.txt' files found in {base_data_dir.resolve()}")
        generate_markdown_report([], [], [], output_report_path, mutations_df, target_residues)
        logger.info("Generated an empty report as no analysis files were found.")
        return

    all_collected_data = []
    wt_summary_entries = []
    mutant_summary_entries = []
    logger.info(f"Processing {len(analysis_report_files)} report files.")

    for i, report_file_path in enumerate(analysis_report_files):
        logger.info(f"Processing report {i+1}/{len(analysis_report_files)}: {report_file_path}")
        optimal_path_nodes, critical_nodes = parse_analysis_report(report_file_path)
        
        found_in_optimal_target = target_residues.intersection(optimal_path_nodes)
        found_in_critical_target = target_residues.intersection(critical_nodes)
        logger.debug(f"Report: {report_file_path} - Target in optimal: {found_in_optimal_target}, Target in critical: {found_in_critical_target}")
        
        relative_sim_path_str = ""
        try:
            relative_sim_path = report_file_path.relative_to(base_data_dir.parent)
            relative_sim_path_str = str(relative_sim_path)
        except ValueError:
            relative_sim_path_str = str(report_file_path)
            logger.warning(f"Could not make path relative for {report_file_path}")

        all_collected_data.append({
            "simulation_path": relative_sim_path_str,
            "optimal_path_found_residues": found_in_optimal_target,
            "critical_path_found_residues": found_in_critical_target,
        })

        res1_str, res2_str = extract_input_residues_from_path(relative_sim_path_str)
        input_pair_display_str = "UnknownPair"
        set_res1, set_res2 = set(), set()

        # This part for parsing input residues from path remains tricky.
        # We will assume it works as intended for now.
        if res1_str:
            try:
                # A helper to parse simple ranges or numbers, not the full function
                if '-' in res1_str:
                    start, end = map(int, res1_str.split('-'))
                    set_res1 = set(range(start, end + 1))
                else:
                    set_res1 = {int(res1_str)}
            except ValueError as e_parse:
                logger.warning(f"Could not parse input residue res1_str '{res1_str}' from path {relative_sim_path_str} for summary: {e_parse}")
        if res2_str:
            try:
                if '-' in res2_str:
                    start, end = map(int, res2_str.split('-'))
                    set_res2 = set(range(start, end + 1))
                else:
                    set_res2 = {int(res2_str)}
            except ValueError as e_parse:
                logger.warning(f"Could not parse input residue res2_str '{res2_str}' from path {relative_sim_path_str} for summary: {e_parse}")
        
        if res1_str and res2_str:
             input_pair_display_str = f"{res1_str}_{res2_str}"
        elif res1_str:
            input_pair_display_str = res1_str
        
        input_residues_for_current_sim = set_res1.union(set_res2)
        
        is_overlap_with_input = bool(target_residues.intersection(input_residues_for_current_sim))
        if is_overlap_with_input:
            logger.debug(f"Target residues {target_residues} overlap with input pair {input_pair_display_str} (parsed as {input_residues_for_current_sim}) for {relative_sim_path_str}")

        optimal_path_intermediate_nodes = optimal_path_nodes - input_residues_for_current_sim
        found_target_as_intermediate_in_optimal = target_residues.intersection(optimal_path_intermediate_nodes)
        
        if bool(found_target_as_intermediate_in_optimal):
            logger.info(f"SUMMARY CHECK: Target {target_residues} FOUND as INTERMEDIATE in optimal path {found_target_as_intermediate_in_optimal} for {relative_sim_path_str} (Inputs: {input_residues_for_current_sim}, Original Optimal: {optimal_path_nodes})")
        elif bool(found_in_optimal_target):
             logger.info(f"SUMMARY CHECK: Target {target_residues} was in optimal path for {relative_sim_path_str} BUT ONLY AS INPUT. (Inputs: {input_residues_for_current_sim}, Original Optimal: {optimal_path_nodes})")
        else:
            logger.info(f"SUMMARY CHECK: Target {target_residues} NOT in optimal path for {relative_sim_path_str} (Original Optimal: {optimal_path_nodes})")
        
        summary_entry = {
            'input_residue_pair_str': input_pair_display_str,
            'is_overlap_with_input': is_overlap_with_input,
            'has_target_in_optimal_as_intermediate': bool(found_target_as_intermediate_in_optimal),
            'intermediate_target_residues_found': found_target_as_intermediate_in_optimal
        }
        
        path_parts = Path(relative_sim_path_str).parts
        if len(path_parts) > 1:
            protein_type_dir = path_parts[1].upper()
            if "_WT" in protein_type_dir:
                wt_summary_entries.append(summary_entry)
            elif "_MUTANT" in protein_type_dir:
                mutant_summary_entries.append(summary_entry)
            else:
                logger.warning(f"Could not categorize {relative_sim_path_str} as WT or Mutant for summary.")
        else:
            logger.warning(f"Path {relative_sim_path_str} too short to categorize for summary.")

    generate_markdown_report(all_collected_data, wt_summary_entries, mutant_summary_entries, output_report_path, mutations_df, target_residues)
    logger.info("Script execution finished successfully.")

if __name__ == "__main__":
    main()
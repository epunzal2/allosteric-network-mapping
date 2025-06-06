import os
import re
import csv

def parse_report(report_path):
    """Parses an analysis_report.txt file to extract path information."""
    path_length = None
    num_nodes = None
    avg_correlation = None
    optimal_path_residues = []
    critical_residues_list = []

    try:
        with open(report_path, 'r') as f:
            content = f.read()

            # Optimal path found: 9 nodes (0.00s).
            match_nodes = re.search(r"Optimal path found: (\d+) nodes", content)
            if match_nodes:
                num_nodes = int(match_nodes.group(1))

            # Path length (sum of 1-|corr| weights): 0.0641
            match_length = re.search(r"Path length \(sum of 1-\|corr\| weights\): ([\d.]+)", content)
            if match_length:
                path_length = float(match_length.group(1))

            # Average |correlation| along path: 0.9920
            match_avg_corr = re.search(r"Average \|correlation\| along path: ([\d.]+)", content)
            if match_avg_corr:
                avg_correlation = float(match_avg_corr.group(1))
            
            # Optimal path (Residue IDs): 101 -> 522 -> 523 -> 532 -> 533 -> 341 -> 356 -> 369 -> 367
            match_optimal_path = re.search(r"Optimal path \(Residue IDs\): ([\s\d\->]+)", content)
            if match_optimal_path:
                path_str = match_optimal_path.group(1).strip()
                optimal_path_residues = [res.strip() for res in path_str.split("->")]

            # Critical Residues
            # Top 10 critical residues (based on Betweenness Centrality):
            #   1. Residue LYS340 (Node 339): 0.0030
            #   2. Residue ALA355 (Node 354): 0.0029
            critical_residues_section = re.search(r"Top \d+ critical residues \(based on Betweenness Centrality\):\s*\n((?:\s+\d+\. Residue .*?\n)+)", content, re.MULTILINE)
            if critical_residues_section:
                lines = critical_residues_section.group(1).strip().split('\n')
                for line in lines:
                    match_crit_res = re.match(r"\s*\d+\.\s*Residue\s*([A-Z0-9]+)\s*\(Node \d+\):\s*([\d.]+)", line.strip())
                    if match_crit_res:
                        critical_residues_list.append({"id": match_crit_res.group(1), "score": float(match_crit_res.group(2))})
                
    except FileNotFoundError:
        # print(f"Warning: Report file not found: {report_path}")
        pass
    except Exception as e:
        # print(f"Warning: Error parsing file {report_path}: {e}")
        pass
        
    return num_nodes, path_length, avg_correlation, optimal_path_residues, critical_residues_list

def main():
    base_results_dir = "analysis_results/Data" # Corrected path
    protein_systems = {
        "WT": "AF2_LM211_WT",
        "Mutant": "AF2_LM2_Y138H_11_Mutant"
    }
    
    # Use one of the CSV files to get the list of pairs, assuming they are the same
    # and we want to process all pairs mentioned in those files.
    # However, it's more robust to scan the directories as some pairs might not have results
    # or results might exist for pairs not explicitly in a filtered list.
    
    all_data = {} # Key: (residue_pair_str, analysis_params_str), Value: {"WT": data, "Mutant": data}

    for system_label, system_folder in protein_systems.items():
        system_path = os.path.join(base_results_dir, system_folder, "calcium")
        if not os.path.isdir(system_path):
            continue

        for res_pair_dir_name in sorted(os.listdir(system_path)):
            res_pair_path = os.path.join(system_path, res_pair_dir_name)
            if os.path.isdir(res_pair_path):
                report_file_path = os.path.join(res_pair_path, "analysis_report.txt")
                
                parts = res_pair_dir_name.split('_')
                residue_pair_str = "N/A"
                analysis_params_str = "N/A"

                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    residue_pair_str = f"{parts[0]}-{parts[1]}"
                    analysis_params_str = "_".join(parts[2:])
                else:
                    analysis_params_str = res_pair_dir_name # Fallback

                num_nodes, path_length, avg_corr, opt_path, crit_res = parse_report(report_file_path)
                
                data_key = (residue_pair_str, analysis_params_str)
                if data_key not in all_data:
                    all_data[data_key] = {}
                
                all_data[data_key][system_label] = {
                    "num_nodes": num_nodes,
                    "path_length": path_length,
                    "avg_correlation": avg_corr,
                    "optimal_path_residues": opt_path,
                    "critical_residues": crit_res
                }

    # Generate Markdown Report
    print("# Extended XS23 Analysis Report: Allosteric Network Mapping (Detailed Comparison)")
    print("\n## Introduction")
    print("This report provides a detailed side-by-side comparison of allosteric network mapping analysis for Wild-Type (WT) and Mutant (Y138H) protein systems. It focuses on optimal communication pathways, their characteristics, common residues, path lengths, and critical residues for key residue pairs.")
    print("The primary analysis parameters for most runs include `displacement_mean_dot` for covariance calculation, `original_ec` for graph pruning, `cbeta` for contact atoms, and a contact frequency cutoff of `0.5` (unless otherwise specified in 'Analysis Params').")

    print("\n## Comparative Pathway Analysis")
    # Removed "Critical Residues (Top 5)" from the next line
    print("| Residue Pair | Analysis Params | System | Num Nodes | Path Length | Avg Correlation |")
    # Adjusted hyphens for the removed column
    print("|--------------|-----------------|--------|-----------|-------------|-----------------|")

    sorted_keys = sorted(all_data.keys())
    
    critical_residues_table_content = ["\n## Top 5 Critical Residues per Analysis\n"]
    critical_residues_table_content.append("| Residue Pair | Analysis Params | System | Critical Residues (Top 5) |")
    critical_residues_table_content.append("|--------------|-----------------|--------|---------------------------|")
    
    # Content for the separate optimal_paths_details.md file
    optimal_paths_details_content = ["# Full Optimal Path Details\n"]
    optimal_paths_details_content.append("| System | Residue Pair | Analysis Params | Optimal Path Residues |")
    optimal_paths_details_content.append("|--------|--------------|-----------------|-------------------------|")

    frequently_common_optimal_residues_counter = {} # For new analysis

    for data_key in sorted_keys:
        res_pair, ana_params = data_key
        
        wt_data = all_data[data_key].get("WT")
        mut_data = all_data[data_key].get("Mutant")

        if wt_data:
            num_n_wt = str(wt_data['num_nodes']) if wt_data['num_nodes'] is not None else "N/A"
            pl_wt = f"{wt_data['path_length']:.4f}" if wt_data['path_length'] is not None else "N/A"
            ac_wt = f"{wt_data['avg_correlation']:.4f}" if wt_data['avg_correlation'] is not None else "N/A"
            op_wt = " -> ".join(wt_data['optimal_path_residues']) if wt_data['optimal_path_residues'] else "N/A"
            # Removed cr_wt_top5 from the print statement below
            print(f"| {res_pair} | {ana_params} | WT | {num_n_wt} | {pl_wt} | {ac_wt} |")
            optimal_paths_details_content.append(f"| WT | {res_pair} | {ana_params} | {op_wt} |")
            cr_wt_top5_str = ", ".join([f"{cr['id']} ({cr['score']:.4f})" for cr in wt_data['critical_residues'][:5]]) if wt_data['critical_residues'] else "N/A"
            critical_residues_table_content.append(f"| {res_pair} | {ana_params} | WT | {cr_wt_top5_str} |")


        if mut_data:
            num_n_mut = str(mut_data['num_nodes']) if mut_data['num_nodes'] is not None else "N/A"
            pl_mut = f"{mut_data['path_length']:.4f}" if mut_data['path_length'] is not None else "N/A"
            ac_mut = f"{mut_data['avg_correlation']:.4f}" if mut_data['avg_correlation'] is not None else "N/A"
            op_mut = " -> ".join(mut_data['optimal_path_residues']) if mut_data['optimal_path_residues'] else "N/A"
            # Removed cr_mut_top5 from the print statement below
            print(f"| {res_pair} | {ana_params} | Mutant | {num_n_mut} | {pl_mut} | {ac_mut} |")
            optimal_paths_details_content.append(f"| Mutant | {res_pair} | {ana_params} | {op_mut} |")
            cr_mut_top5_str = ", ".join([f"{cr['id']} ({cr['score']:.4f})" for cr in mut_data['critical_residues'][:5]]) if mut_data['critical_residues'] else "N/A"
            critical_residues_table_content.append(f"| {res_pair} | {ana_params} | Mutant | {cr_mut_top5_str} |")

        if wt_data and mut_data:
            common_opt_res_str = "N/A"
            common_opt_res_count = 0
            if wt_data['optimal_path_residues'] and mut_data['optimal_path_residues']:
                common_set = set(wt_data['optimal_path_residues']) & set(mut_data['optimal_path_residues'])
                if common_set:
                    common_opt_res_list = sorted(list(common_set))
                    common_opt_res_str = ", ".join(common_opt_res_list)
                    common_opt_res_count = len(common_opt_res_list)
                    
                    # For frequently_common_optimal_residues_counter, exclude start/end nodes
                    start_node, end_node = res_pair.split('-')
                    for res_id in common_set:
                        if res_id != start_node and res_id != end_node:
                            frequently_common_optimal_residues_counter[res_id] = frequently_common_optimal_residues_counter.get(res_id, 0) + 1
                else:
                    common_opt_res_str = "None"
            
            shorter_path_system = "N/A"
            if wt_data['path_length'] is not None and mut_data['path_length'] is not None:
                if wt_data['path_length'] < mut_data['path_length']:
                    shorter_path_system = f"WT ({wt_data['path_length']:.4f} vs Mutant {mut_data['path_length']:.4f})"
                elif mut_data['path_length'] < wt_data['path_length']:
                    shorter_path_system = f"Mutant ({mut_data['path_length']:.4f} vs WT {wt_data['path_length']:.4f})"
                else:
                    shorter_path_system = f"Equal ({wt_data['path_length']:.4f})"
            elif wt_data['path_length'] is not None:
                 shorter_path_system = "WT (Mutant N/A)"
            elif mut_data['path_length'] is not None:
                 shorter_path_system = "Mutant (WT N/A)"


            # Adjusted print for removed column and added count
            print(f"| {res_pair} | {ana_params} | **Comparison** | | Shorter Path: {shorter_path_system} | Common Optimal Residues: {common_opt_res_str} ({common_opt_res_count} shared) |")
        # Adjusted hyphens for the removed column
        print(f"| {'-'*14} | {'-'*17} | {'-'*8} | {'-'*11} | {'-'*13} | {'-'*17} |")

    # Print the new critical residues table
    for line in critical_residues_table_content:
        print(line)

    # Analysis of Critical Residues Consistency
    print("\n## Critical Residue Consistency Analysis")
    
    overall_critical_wt = {} # res_id: [scores]
    overall_critical_mutant = {}

    for data_key in sorted_keys:
        res_pair, ana_params = data_key
        wt_data = all_data[data_key].get("WT")
        mut_data = all_data[data_key].get("Mutant")

        if wt_data and wt_data['critical_residues']:
            for cr in wt_data['critical_residues']:
                if cr['id'] not in overall_critical_wt:
                    overall_critical_wt[cr['id']] = []
                overall_critical_wt[cr['id']].append(cr['score'])
        
        if mut_data and mut_data['critical_residues']:
            for cr in mut_data['critical_residues']:
                if cr['id'] not in overall_critical_mutant:
                    overall_critical_mutant[cr['id']] = []
                overall_critical_mutant[cr['id']].append(cr['score'])

    print("\n### Wild-Type (WT) System - Frequently Identified Critical Residues (Top 10 by appearance count, then avg score)")
    # Sort by count, then by average score
    sorted_crit_wt = sorted(overall_critical_wt.items(), key=lambda item: (len(item[1]), sum(item[1])/len(item[1])), reverse=True)
    for i, (res_id, scores) in enumerate(sorted_crit_wt[:10]):
        avg_score = sum(scores) / len(scores)
        print(f"- {res_id}: Appeared in {len(scores)} pairs, Avg. Score: {avg_score:.4f}")

    print("\n### Mutant System - Frequently Identified Critical Residues (Top 10 by appearance count, then avg score)")
    sorted_crit_mut = sorted(overall_critical_mutant.items(), key=lambda item: (len(item[1]), sum(item[1])/len(item[1])), reverse=True)
    for i, (res_id, scores) in enumerate(sorted_crit_mut[:10]):
        avg_score = sum(scores) / len(scores)
        print(f"- {res_id}: Appeared in {len(scores)} pairs, Avg. Score: {avg_score:.4f}")

    print("\n### Comparison of Top Critical Residues (WT vs Mutant)")
    top_crit_wt_ids = {item[0] for item in sorted_crit_wt[:10]}
    top_crit_mut_ids = {item[0] for item in sorted_crit_mut[:10]}

    common_top_critical = top_crit_wt_ids & top_crit_mut_ids
    wt_only_top_critical = top_crit_wt_ids - top_crit_mut_ids
    mut_only_top_critical = top_crit_mut_ids - top_crit_wt_ids

    if common_top_critical:
        print("\n#### Critical Residues in Top 10 of BOTH WT and Mutant:")
        for res_id in sorted(list(common_top_critical)):
            wt_info = next(item for item in sorted_crit_wt if item[0] == res_id)
            mut_info = next(item for item in sorted_crit_mut if item[0] == res_id)
            wt_avg_score = sum(wt_info[1]) / len(wt_info[1])
            mut_avg_score = sum(mut_info[1]) / len(mut_info[1])
            print(f"- {res_id} (WT: {len(wt_info[1])} pairs, Avg Score: {wt_avg_score:.4f}; Mutant: {len(mut_info[1])} pairs, Avg Score: {mut_avg_score:.4f})")
    else:
        print("\n#### No critical residues found in the Top 10 of BOTH WT and Mutant.")

    if wt_only_top_critical:
        print("\n#### Critical Residues in Top 10 of WT ONLY:")
        for res_id in sorted(list(wt_only_top_critical)):
            wt_info = next(item for item in sorted_crit_wt if item[0] == res_id)
            wt_avg_score = sum(wt_info[1]) / len(wt_info[1])
            print(f"- {res_id} (WT: {len(wt_info[1])} pairs, Avg Score: {wt_avg_score:.4f})")
    else:
        print("\n#### No critical residues found in the Top 10 of WT ONLY (all WT top 10 are shared or not in Mutant's top 10).")

    if mut_only_top_critical:
        print("\n#### Critical Residues in Top 10 of Mutant ONLY:")
        for res_id in sorted(list(mut_only_top_critical)):
            mut_info = next(item for item in sorted_crit_mut if item[0] == res_id)
            mut_avg_score = sum(mut_info[1]) / len(mut_info[1])
            print(f"- {res_id} (Mutant: {len(mut_info[1])} pairs, Avg Score: {mut_avg_score:.4f})")
    else:
        print("\n#### No critical residues found in the Top 10 of Mutant ONLY (all Mutant top 10 are shared or not in WT's top 10).")
        
    print("\n**Note on Critical Residue Consistency:**")
    print("The lists above show residues that frequently appear as 'critical' (high betweenness centrality) across different residue pair analyses within each system. A high appearance count suggests a globally important role in network communication for that system. Variations in scores across different pairs indicate context-dependent importance.")
    print("Comparing the top critical residues between WT and Mutant, as done in the 'Comparison of Top Critical Residues' subsection, reveals how the mutation might shift the key communication hubs within the protein by highlighting shared and unique critical players.")

    print("\n## Frequently Common Optimal Path Residues Analysis")
    print("This section lists residues that most frequently appear as part of the 'Common Optimal Residues' set when comparing WT and Mutant pathways across different residue pairs.")
    if frequently_common_optimal_residues_counter:
        sorted_frequent_common = sorted(frequently_common_optimal_residues_counter.items(), key=lambda item: item[1], reverse=True)
        print("\n### Top Residues Appearing in Common WT-Mutant Optimal Paths (Top 10 by appearance count):")
        for i, (res_id, count) in enumerate(sorted_frequent_common[:10]):
            print(f"- {res_id}: Appeared in {count} common optimal path sets.")
    else:
        print("No common optimal path residues found to analyze for frequency.")

    print("\n## Summary and Verification Notes (Placeholder - to be filled based on VERIFICATION PROMPT)")
    print("This section would typically address the VERIFICATION PROMPT for Task ID: 2025-06-05_23:02__RQ55, including HPC run log summaries, cryptographic hash details, and how these findings support HPC-lifespan synergy goals, incorporating the detailed comparisons made above.")
    
    # Print the content for optimal_paths_details.md, delimited
    print("\n\n---OPTIMAL_PATHS_DETAILS_MD_CONTENT_START---")
    for line in optimal_paths_details_content:
        print(line)
    print("---OPTIMAL_PATHS_DETAILS_MD_CONTENT_END---")


if __name__ == "__main__":
    main()
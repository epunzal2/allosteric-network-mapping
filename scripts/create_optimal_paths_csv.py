import os
import re
import pandas as pd

def parse_markdown_and_create_csv(markdown_file, output_dir_base):
    """
    Parses a markdown file to extract optimal path details and creates CSV files.

    Args:
        markdown_file (str): The path to the markdown file.
        output_dir_base (str): The base directory to save the CSV files.
    """
    with open(markdown_file, 'r') as f:
        content = f.read()

    categories = re.split(r'## Category: ', content)[1:]

    for category_block in categories:
        lines = category_block.strip().split('\n')
        category_name = lines[0].strip()
        
        data_lines = [line for line in lines if '|' in line and '---' not in line and 'System' not in line]

        for line in data_lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) < 7:
                continue

            system = parts[0]
            residue_pair = parts[1]
            optimal_path_str = parts[6]

            if optimal_path_str == 'N/A':
                continue

            output_dir = os.path.join(output_dir_base, f"{category_name.replace(' ', '_')}_{system}")
            os.makedirs(output_dir, exist_ok=True)

            path_residues = [res.strip() for res in optimal_path_str.split('->')]
            
            df = pd.DataFrame(path_residues, columns=['optimal_path_residues'])
            
            csv_filename = os.path.join(output_dir, f"path_{residue_pair}.csv")
            df.to_csv(csv_filename, index=False)
            print(f"Created CSV: {csv_filename}")

def get_residue_mapping(file_path):
    """Reads a residue mapping file and returns a dictionary."""
    mapping = {}
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        # Use resid_orig for lookup as it's in the filenames
        mapping[str(row['resid_orig'])] = f"{row['subunit_greek']}{row['res']}{row['resid_orig']}"
    return mapping

def create_orig_label_csv(paths_all_file, mapping_file):
    """Creates a paths_all_orig_label.csv file with original labels."""
    print(f"Creating original label file for {paths_all_file}")
    try:
        paths_df = pd.read_csv(paths_all_file, dtype=str).fillna('')
        mapping_df = pd.read_csv(mapping_file)

        # resid -> full_orig_label
        resid_to_label = pd.Series(mapping_df.full_orig_label.values, index=mapping_df.resid.astype(str)).to_dict()
        
        # header -> full_orig_label
        header_to_label = {}
        for _, row in mapping_df.iterrows():
            header = f"{row['subunit_greek']}{row['res']}{row['resid_orig']}"
            header_to_label[header] = row['full_orig_label']

        labeled_df = paths_df.map(lambda x: resid_to_label.get(x, x))
        labeled_df.rename(columns=header_to_label, inplace=True)

        output_path = os.path.join(os.path.dirname(paths_all_file), 'paths_all_orig_label.csv')
        labeled_df.to_csv(output_path, index=False)
        print(f"Created {output_path}")
    except Exception as e:
        print(f"Error creating original label file for {paths_all_file}: {e}")

def create_comprehensive_csv(root_dir, mapping_wt_path, mapping_mutant_path):
    """
    Creates a comprehensive CSV file from individual path files in subdirectories.
    """
    mapping_wt = get_residue_mapping(mapping_wt_path)
    mapping_mutant = get_residue_mapping(mapping_mutant_path)

    for subdir, _, _ in os.walk(root_dir):
        # Process only immediate subdirectories of root_dir
        if os.path.dirname(subdir) != root_dir and root_dir != subdir:
            continue
        if not os.path.basename(subdir):
            continue

        path_files = [f for f in os.listdir(subdir) if f.startswith('path_') and f.endswith('.csv')]
        if not path_files:
            continue

        print(f"Processing subdirectory: {subdir}")

        all_paths_data = {}
        max_len = 0

        # Determine which mapping to use
        if "Mutant" in os.path.basename(subdir):
            current_mapping = mapping_mutant
        else:
            current_mapping = mapping_wt

        for filename in path_files:
            match = re.match(r'path_\d+-(\d+)\.csv', filename)
            if match:
                residue_num = match.group(1)
                
                # Get the new header from the mapping
                new_header = current_mapping.get(residue_num, residue_num)

                file_path = os.path.join(subdir, filename)
                try:
                    path_df = pd.read_csv(file_path)
                    if 'optimal_path_residues' in path_df.columns:
                        residues = path_df['optimal_path_residues'].tolist()
                        all_paths_data[new_header] = residues
                        if len(residues) > max_len:
                            max_len = len(residues)
                    else:
                        print(f"Warning: 'optimal_path_residues' column not found in {filename}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        # Pad shorter paths with empty strings to ensure equal length
        for header, residues in all_paths_data.items():
            if len(residues) < max_len:
                all_paths_data[header].extend([''] * (max_len - len(residues)))

        if all_paths_data:
            comprehensive_df = pd.DataFrame(all_paths_data)
            
            # Sort columns based on the original residue number
            sorted_headers = sorted(comprehensive_df.columns, key=lambda x: int(''.join(filter(str.isdigit, x))) if ''.join(filter(str.isdigit, x)) else 0)
            comprehensive_df = comprehensive_df[sorted_headers]

            output_path = os.path.join(subdir, 'paths_all.csv')
            comprehensive_df.to_csv(output_path, index=False)
            print(f"Created {output_path}")

            # Determine which mapping file to use for the new CSV
            if "Mutant" in os.path.basename(subdir):
                mapping_file = mapping_mutant_path
            else:
                mapping_file = mapping_wt_path
            
            create_orig_label_csv(output_path, mapping_file)

if __name__ == '__main__':
    markdown_file = 'analysis_results/reports/optimal_paths_details.md'
    output_dir_base = 'analysis_results/reports/optimal_paths_csv'
    parse_markdown_and_create_csv(markdown_file, output_dir_base)

    MAPPING_WT = 'Data/residue_mapping_WT.csv'
    MAPPING_MUTANT = 'Data/residue_mapping_Mutant.csv'
    
    if not os.path.exists(MAPPING_MUTANT):
        print(f"Warning: Mutant mapping file not found at {MAPPING_MUTANT}.")
        if "Mutant" in [d for d in os.listdir(output_dir_base) if os.path.isdir(os.path.join(output_dir_base, d))]:
             print("A 'Mutant' directory exists but no mapping file. The script might fail.")
             if os.path.exists(MAPPING_WT):
                 print("Using WT mapping as a fallback for Mutant.")
                 MAPPING_MUTANT = MAPPING_WT

    create_comprehensive_csv(output_dir_base, MAPPING_WT, MAPPING_MUTANT)
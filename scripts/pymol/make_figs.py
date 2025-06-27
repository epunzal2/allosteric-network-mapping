import os
import re
import logging
from pymol import cmd

def parse_optimal_paths(markdown_file):
    """
    Parses the optimal path data from a markdown file.

    Args:
        markdown_file (str): The path to the markdown file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row
              in the markdown table.
    """
    with open(markdown_file, 'r') as f:
        content = f.read()

    data = []
    # Split the content by category sections
    sections = re.split(r'## Category: ', content)
    
    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n')
        category = lines[0].strip()
        
        table_started = False
        for line in lines:
            # Find table header
            if '|' in line and 'System' in line and 'Residue Pair' in line:
                table_started = True
                continue
            
            if not table_started or not line.strip().startswith('|'):
                continue

            # Skip header separator
            if '---' in line:
                continue

            columns = [col.strip() for col in line.split('|')]
            
            # Expected columns: '', 'System', 'Residue Pair', ..., 'Optimal Path Residues', ''
            if len(columns) > 7 and columns[1] and columns[2] and columns[7] and "N/A" not in columns[7]:
                system = columns[1]
                residue_pair = columns[2]
                path_str = columns[7]
                
                path_residues = [int(r) for r in re.findall(r'\d+', path_str)]
                if path_residues:
                    data.append({
                        "System": system,
                        "Category": category.replace(' ', '_'),
                        "Residue Pair": residue_pair,
                        "Optimal Path Residues": path_residues
                    })
    return data

def get_resname(residue_id):
    """Gets the 3-letter residue name for a given residue ID."""
    resn_list = []
    cmd.iterate(f"resi {residue_id} and name CA", "resn_list.append(resn)", space=locals())
    return resn_list[0] if resn_list else ""

def visualize_paths(system_name, category, paths, output_dir):
    """
    Generates and saves a PyMOL visualization for the given paths.

    Args:
        system_name (str): 'WT' or 'Mutant'.
        category (str): The category of the paths (e.g., 'gamma_loop').
        paths (list): A list of dictionaries, where each dictionary has 'residues' and 'pair' keys.
        output_dir (str): The directory to save the output files.
    """
    # Load structure
    if system_name == 'WT':
        pdb_file = 'Data/AF2_LM211_WT/calcium/frame1.pdb'
        parm7_file = 'Data/AF2_LM211_WT/calcium/af2_lm211_wt_ca2+.parm7'
    else:
        pdb_file = 'Data/AF2_LM2_Y138H_11_Mutant/calcium/frame1.pdb'
        parm7_file = 'Data/AF2_LM2_Y138H_11_Mutant/calcium/af2_lm2_y138h_11_mutant_ca2+.parm7'

    cmd.load(pdb_file, 'protein')
    cmd.load(parm7_file)

    # Basic representation
    cmd.hide('all')
    cmd.show('cartoon')
    cmd.color('gray80', 'protein')
    cmd.label('name CA', '""') # Clear all C-alpha labels at the start

    # Expanded color palette
    colors = [
        "brightorange", "red", "green", "blue", "yellow", "cyan", "orange",
        "purple", "pink", "lime", "teal", "olive", "brown", "violet", "gold",
        "salmon", "skyblue", "hotpink", "lightteal", "warmpink", "limon", "deeppurple"
    ]

    for i, path_info in enumerate(paths):
        # Preserve order of residues from file, remove duplicates
        path_with_dupes = path_info['residues']
        path = []
        [path.append(x) for x in path_with_dupes if x not in path]

        pair = path_info['pair']
        start_resid, end_resid = map(int, pair.split('-'))

        path_name = f"{category}_{system_name}_path_{i+1}"
        selection = " or ".join([f"resi {res}" for res in path])
        cmd.select(path_name, selection)
        
        path_color = colors[i % len(colors)]
        cmd.color(path_color, path_name)

        # Make cartoon representation of the path semi-transparent
        cmd.show('cartoon', selection)
        cmd.set('cartoon_transparency', 0.5, selection)

        # Show all nodes in the path as spheres
        all_nodes_selection = " or ".join([f"resi {r} and name CA" for r in path])
        cmd.show("spheres", all_nodes_selection)
        cmd.color(path_color, all_nodes_selection)
        
        # Set sphere sizes
        cmd.set("sphere_scale", 0.4, all_nodes_selection) # Intermediate node default
        cmd.set("sphere_scale", 0.6, f"resi {start_resid} and name CA") # Start node
        cmd.set("sphere_scale", 0.6, f"resi {end_resid} and name CA") # End node

        # Create thicker connections (cylinders) between C-alpha atoms
        for j in range(len(path) - 1):
            res1 = path[j]
            res2 = path[j+1]
            dist_name = f"path_{i}_conn_{j}"
            cmd.distance(dist_name, f"resi {res1} and name CA", f"resi {res2} and name CA")
            cmd.set("dash_color", path_color, dist_name)
            cmd.set("dash_gap", 0, dist_name)
            cmd.set("dash_radius", 0.2, dist_name)
            cmd.hide('labels', dist_name)

        # Add labels for start and end residues ONLY
        start_resname = get_resname(start_resid)
        end_resname = get_resname(end_resid)
        
        # Label start and end residues
        if start_resid != 101:
            cmd.label(f"resi {start_resid} and name CA", f'"{start_resname}{start_resid}"')
            cmd.set("label_color", path_color, f"resi {start_resid} and name CA")
        
        cmd.label(f"resi {end_resid} and name CA", f'"{end_resname}{end_resid}"')
        cmd.set("label_color", path_color, f"resi {end_resid} and name CA")


    # Highlight Residue 101 globally
    res101_selection = "resi 101 and name CA"
    cmd.show("spheres", res101_selection)
    cmd.color("magenta", res101_selection)
    cmd.set("sphere_scale", 0.8, res101_selection) # Make it largest
    
    res101_resname = get_resname(101)
    cmd.label(res101_selection, f'"{res101_resname}101"')
    cmd.set("label_color", "magenta", res101_selection)

    # Ray trace and save
    output_base = os.path.join(output_dir, f"{category}_{system_name}")
    cmd.ray(800, 600)
    cmd.png(f"{output_base}.png")
    cmd.save(f"{output_base}.pse")

    cmd.delete('all')

def main():
    """
    Main function to generate all visualizations.
    """
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=os.path.join(log_dir, 'pymol_visualization.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')

    markdown_file = 'analysis_results/optimal_paths_details.md'
    output_dir = 'analysis_results/pymol_visualizations/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_paths = parse_optimal_paths(markdown_file)

    # Group paths by system and category
    grouped_paths = {}
    for path in all_paths:
        key = (path['System'], path['Category'])
        if key not in grouped_paths:
            grouped_paths[key] = []
        grouped_paths[key].append({'residues': path['Optimal Path Residues'], 'pair': path['Residue Pair']})

    # Generate individual visualizations
    for (system, category), paths in grouped_paths.items():
        logging.info(f"Visualizing {system} - {category.replace('_', ' ')}...")
        visualize_paths(system, category, paths, output_dir)
        cmd.reinitialize()

    # Generate combined visualizations for high contact categories
    for system in ['WT', 'Mutant']:
        logging.info(f"Visualizing {system} - Combined High Contact...")
        combined_paths = []
        for category in ['high_contact_both', 'high_contact_WT', 'high_contact_Mutant']:
            key = (system, category)
            if key in grouped_paths:
                combined_paths.extend(grouped_paths[key])
        
        if combined_paths:
            visualize_paths(system, 'combined_high_contact', combined_paths, output_dir)
            cmd.reinitialize()

    logging.info("All visualizations generated.")

if __name__ == '__main__':
    main()
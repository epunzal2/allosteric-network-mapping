import os
import re
import logging
from chimerax.core.commands import run

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

def get_resname(session, resid, chain_id=None):
    model = session.models.list()[0]        # first atomic model (#1)
    res   = model.residues                  # a Residues array

    mask = (res.numbers == resid)
    if chain_id is not None:
        mask &= (res.chain_ids == chain_id)

    return res.names[mask][0] if mask.any() else ""

def visualize_paths(session, system_name, category, paths, output_dir):
    """
    Generates and saves a ChimeraX visualization for the given paths.

    Args:
        session: The ChimeraX session object.
        system_name (str): 'WT' or 'Mutant'.
        category (str): The category of the paths (e.g., 'gamma_loop').
        paths (list): A list of dictionaries, where each dictionary has 'residues' and 'pair' keys.
        output_dir (str): The directory to save the output files.
    """
    # Load structure
    if system_name == 'WT':
        pdb_file = 'Data/AF2_LM211_WT/calcium/frame1_CA.pdb'
    else:
        pdb_file = 'Data/AF2_LM2_Y138H_11_Mutant/calcium/frame1_CA.pdb'

    run(session, f'open {pdb_file}')

    # Basic representation
    run(session, 'hide all')
    run(session, 'show cartoon')
    run(session, 'color light gray')
    run(session, 'label delete') # Clear all labels

    # Check for and visualize Calcium ions
    calcium_selection = "#1:918-928"
    
    # Programmatically select atoms to check for existence
    # Use run command to select and check for existence
    # Use run command to select and check for existence
    selected_atoms = run(session, f'select {calcium_selection}')
    
    # The result of the select command tells us if atoms were selected
    if selected_atoms and len(selected_atoms.atoms) > 0:
        logging.info(f"Calcium ions found ({len(selected_atoms.atoms)} atoms), visualizing them.")
        # Use run for visualization commands
        run(session, f'show {calcium_selection} atoms')
        run(session, f'style {calcium_selection} sphere')
        run(session, f'color {calcium_selection} purple')
        run(session, f'size {calcium_selection} atomRadius 1.0')
    else:
        logging.info("No Calcium ions found in the structure.")

    # Expanded color palette
    colors = [
        "orange", "red", "green", "blue", "yellow",
        "cyan", "lime", "teal", "olive", "brown",
        "gold", "salmon", "skyblue", "hotpink", "coral",
        "crimson", "darkgreen", "dodgerblue", "chocolate", "tomato",
        "khaki", "seagreen", "turquoise", "indianred", "sienna"
    ]

    seen = set()
    for i, path_info in enumerate(paths):
        path_with_dupes = path_info['residues']
        path = list(dict.fromkeys(path_with_dupes)) # Remove duplicates, preserve order

        pair = path_info['pair']
        start_resid, end_resid = map(int, pair.split('-'))

        selection = f"#1:{','.join(map(str, path))}"
        path_color = colors[i % len(colors)]
        
        run(session, f'color {selection} {path_color} target c')
        run(session, f'transparency {selection} 50 target c')

        # Show all nodes in the path as spheres
        all_nodes_selection = f"{selection}"
        run(session, f'show {all_nodes_selection} atoms')
        run(session, f'style {all_nodes_selection} sphere')
        run(session, f'color {all_nodes_selection} {path_color}')

        # NEW – make the spheres 70 % transparent
        # run(session, f'transparency 70 {all_nodes_selection} target a') 
        
        # Set sphere sizes
        run(session, f'size {all_nodes_selection} atomRadius 0.4') # Intermediate node default
        run(session, f'size #1:{start_resid}@CA atomRadius 0.6') # Start node
        run(session, f'size #1:{end_resid}@CA atomRadius 0.6') # End node

        # Create thicker connections between C-alpha atoms
        for j in range(len(path) - 1):
            res1 = path[j]
            res2 = path[j+1]
            
            pair = tuple(sorted((res1, res2)))
            if pair not in seen:
                # Use distance command to create a pseudobond
                dist_cmd = (
                    f'distance #1:{res1}@CA #1:{res2}@CA '
                    f'color {path_color} radius 0.2'
                )
                run(session, dist_cmd)
                seen.add(pair)

        # Add labels for start and end residues ONLY
        start_resname = get_resname(session, start_resid)
        end_resname = get_resname(session, end_resid)
        
        if start_resid != 101:
            run(session, f'label #1:{start_resid}@CA text "{start_resname}{start_resid}" color {path_color}')
        
        run(session, f'label #1:{end_resid}@CA text "{end_resname}{end_resid}" color {path_color}')

    # Set the style for all distances created in the loop to solid.
    # run(session, "distance style solid")
    # NEW – make every distance pseudobond 70 % transparent
    # run(session, 'transparency 70 #distances target p') 

    # Highlight Residue 101 globally
    res101_selection = "#1:101@CA"
    run(session, f'show {res101_selection} atoms')
    run(session, f'style {res101_selection} sphere')
    run(session, f'color {res101_selection} magenta')
    run(session, f'size {res101_selection} atomRadius 0.8') # Make it largest
    
    res101_resname = get_resname(session, 101)
    run(session, f'label {res101_selection} text "{res101_resname}101" color magenta')

    # Save image and session only if GUI is available
    # Save image and session
    output_base = os.path.join(output_dir, f"{category}_{system_name}")
    run(session, f'save "{output_base}.png" width 800 height 600')
    run(session, f'save "{output_base}.cxs"')

    run(session, 'close all')

def main(session):
    """
    Main function to generate all visualizations in ChimeraX.
    """
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=os.path.join(log_dir, 'chimerax_visualization.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')

    markdown_file = 'analysis_results/reports/optimal_paths_details.md'
    output_dir = 'analysis_results/chimera_visualizations/'

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
        visualize_paths(session, system, category, paths, output_dir)

    # Generate combined visualizations for high contact categories
    for system in ['WT', 'Mutant']:
        logging.info(f"Visualizing {system} - Combined High Contact...")
        combined_paths = []
        for category in ['high_contact_both', 'high_contact_WT', 'high_contact_Mutant']:
            key = (system, category)
            if key in grouped_paths:
                combined_paths.extend(grouped_paths[key])
        
        if combined_paths:
            visualize_paths(session, system, 'combined_high_contact', combined_paths, output_dir)

main(session)

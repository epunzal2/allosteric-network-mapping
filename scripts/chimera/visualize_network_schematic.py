import os
import re
import logging
import csv
import sys
from datetime import datetime
from chimerax.core.commands import run

# Determine the project's root directory, assuming the script is in scripts/chimera/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the script's directory to the Python path to allow for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from visualize_residues_schematic import visualize_residues

# Global flag to track which method works for this ChimeraX version
_tube_method = None

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

def load_residue_mapping(mapping_file):
    """
    Loads the residue mapping from a CSV file.

    Args:
        mapping_file (str): The path to the CSV file.

    Returns:
        dict: A dictionary mapping residue ID (str) to full original label (str).
    """
    mapping = {}
    with open(mapping_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['resid']] = row['full_orig_label']
    return mapping

def get_resname(atomic_model, resid, chain_id=None):
    """
    Gets the residue name from a specific atomic model.
    """
    if not hasattr(atomic_model, 'residues'):
        return ""  # Return empty if it's not an atomic model

    res = atomic_model.residues
    mask = (res.numbers == resid)
    if chain_id is not None:
        mask &= (res.chain_ids == chain_id)

    return res.names[mask][0] if mask.any() else ""

def setup_wireframe_backbone(session, model_id):
    """
    Set up uniform wireframe backbone using cartoon representation
    """
    global _tube_method
    
    logging.info("Setting up uniform wireframe backbone...")
    try:
        # Hide ALL atoms initially
        run(session, f"hide {model_id} atoms")
        
        # Use cartoon representation with UNIFORM thickness
        logging.info("Using cartoon backbone with uniform thickness")
        
        try:
            # Show cartoon representation (ribbon/backbone)
            run(session, f"show {model_id} cartoons")
            # Force uniform thickness regardless of secondary structure
            run(session, f"cartoon style {model_id} width 0.15 thickness 0.15")
            run(session, f"color {model_id} lightgray")
            run(session, f"transparency {model_id} 80")
            logging.info("Uniform cartoon backbone displayed - consistent thickness")
            _tube_method = "cartoon"
            
        except Exception as e:
            logging.error(f"Failed to create cartoon backbone: {e}")
            # Fallback: try uniform spheres
            try:
                run(session, f"show {model_id}@CA")
                run(session, f"style {model_id}@CA sphere")
                run(session, f"size {model_id}@CA atomRadius 0.1")
                run(session, f"color {model_id}@CA lightgray")
                run(session, f"transparency {model_id}@CA 80")
                logging.info("Fallback: uniform CA spheres displayed")
                _tube_method = "spheres"
            except Exception as e2:
                logging.error(f"All backbone methods failed: {e2}")
                _tube_method = "none"
        
        # No silhouettes for clean look
        run(session, "graphics silhouettes false")
        logging.info("Uniform wireframe backbone setup complete")
        return True
        
    except Exception as e:
        logging.error(f"Failed to setup backbone: {e}")
        return False

def setup_path_tube(session, model_id, residues, color, tube_name, radius=0.45, transparency=40):
    """
    Create a tube for specific residues with optimized method detection
    """
    global _tube_method
    
    try:
        if not residues:
            return False
            
        sel = f"{model_id}:{','.join(map(str, residues))}"
        
        # Use the method we know works for this ChimeraX version
        if _tube_method == "no_name":
            cmd = f"shape tube {sel}@CA radius {radius} color {color}"
            run(session, cmd)
            if transparency > 0:
                run(session, f"transparency {transparency} surfaces")
        elif _tube_method == "with_name":
            cmd = f"shape tube {sel}@CA radius {radius} color {color} name {tube_name}"
            run(session, cmd)
            run(session, f"transparency {transparency} surfaces name {tube_name}")
        elif _tube_method == "cartoon":
            # For cartoon method, style the specific residues
            run(session, f"cartoon style {sel} width {radius*0.4} thickness {radius*0.2}")
            run(session, f"color {sel} {color}")
            if transparency > 0:
                run(session, f"transparency {sel} {transparency}")
        else:
            # Fallback - try the methods in order
            try:
                cmd = f"shape tube {sel}@CA radius {radius} color {color}"
                run(session, cmd)
                if transparency > 0:
                    run(session, f"transparency {transparency} surfaces")
            except Exception:
                # Final fallback to sphere style
                run(session, f"show {sel}@CA")
                run(session, f"style {sel}@CA sphere")
                run(session, f"size {sel}@CA atomRadius {radius}")
                run(session, f"color {sel}@CA {color}")
                if transparency > 0:
                    run(session, f"transparency {sel}@CA {transparency}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to create tube {tube_name}: {e}")
        return False

def visualize_single_path(session,
                         system_name: str,
                         category: str,
                         path_data: dict,
                         output_dir: str,
                         residue_mapping: dict):
    """
    Visualize a single network path with clean wireframe background.
    """
    
    print(f"=== VISUALIZE_SINGLE_PATH: {system_name} - {category} - {path_data['pair']} ===")

    # Open structure
    pdb_file = (
        os.path.join(project_root, "Data/AF2_LM211_WT/calcium/frame1_CA.pdb")
        if system_name == "WT" else
        os.path.join(project_root, "Data/AF2_LM2_Y138H_11_Mutant/calcium/frame1_CA.pdb")
    )
    logging.info(f"Opening PDB file: {pdb_file}")
    try:
        run(session, f'open "{pdb_file}"')
        logging.info("PDB file opened successfully")
        
        # Get the current model number (the one just opened)
        models = session.models.list()
        current_model = models[-1]  # Last opened model
        model_id = f"#{current_model.id_string}"
        logging.info(f"Working with model: {model_id}")
        
    except Exception as e:
        logging.error(f"Failed to open PDB file: {e}")
        return

    # Basic scene setup - transparent background for raytracing
    run(session, "set bgColor transparent")
    run(session, "hide atoms")
    run(session, "hide cartoons")
    
    # Show protein structure properly FIRST
    logging.info("Setting up protein visualization...")
    try:
        # Show all atoms and bonds
        run(session, f"show {model_id}")
        run(session, f"show {model_id} atoms")
        run(session, f"show {model_id} bonds")
        logging.info("Protein structure setup complete")
    except Exception as e:
        logging.error(f"Failed to setup protein structure: {e}")
        return
    
    # Set desired orientation - CONSISTENT for all models (AFTER showing objects)
    logging.info("Setting consistent orientation for all visualizations")
    run(session, "view all")  # Reset to default view first (now objects are displayed)
    run(session, "turn x 90")
    run(session, "turn z 90")
    run(session, "turn x 90")
    run(session, "turn z -10")
    
    # Set up wireframe backbone
    if not setup_wireframe_backbone(session, model_id):
        return
    
    # Highlight single path with THICKER colored tube (translucent)
    try:
        seg = list(dict.fromkeys(path_data["residues"]))
        logging.info(f"Styling single path: residues {seg} with color royalblue")
        setup_path_tube(session, model_id, seg, "royalblue", "path_tube", 0.6, 40)  # Increased from 0.45 to 0.6
    except Exception as e:
        logging.error(f"Failed to style path: {e}")
        return

    # Residue 101 as SUPER prominent magenta tube (MUCH larger, with white shadow)
    try:
        # Create an even larger white shadow for maximum contrast
        setup_path_tube(session, model_id, [101], "white", "res101_shadow", 1.0, 0)
        # Create a very prominent magenta tube on top
        setup_path_tube(session, model_id, [101], "magenta", "res101_tube", 0.9, 0)
        
        # Add very prominent label for residue 101
        label_txt = residue_mapping.get("101", f"{get_resname(current_model, 101)}101")
        run(session, f'label {model_id}:101@CA text "{label_txt}" color magenta size 18 height 2.0')
        logging.info("Residue 101 styled with MAXIMUM visibility (large white shadow + large magenta overlay)")
    except Exception as e:
        logging.error(f"Failed to style residue 101: {e}")

    # Color the second residue in the pair - SAME PROMINENCE as residue 101
    try:
        # Extract the second residue number from the pair
        pair_parts = path_data['pair'].replace('→', '-').replace(' ', '').split('-')
        if len(pair_parts) >= 2:
            second_residue = int(pair_parts[1])
            
            # Make second residue EQUALLY prominent as residue 101
            # Create white shadow for maximum contrast
            setup_path_tube(session, model_id, [second_residue], "white", "second_res_shadow", 1.0, 0)
            # Create prominent yellow tube on top (same size as magenta res 101)
            setup_path_tube(session, model_id, [second_residue], "yellow", "second_res_tube", 0.9, 0)
            
            # Add very prominent label (same size as res 101)
            label_txt = residue_mapping.get(str(second_residue), f"{get_resname(current_model, second_residue)}{second_residue}")
            run(session, f'label {model_id}:{second_residue}@CA text "{label_txt}" color yellow size 18 height 2.0')
            logging.info(f"Second residue {second_residue} styled with MAXIMUM visibility (same as res 101)")
    except Exception as e:
        logging.error(f"Failed to style second residue: {e}")

    # Show ONLY Ca 918, hide all other calcium and protein atoms
    try:
        # Hide ALL atoms first
        run(session, f"hide {model_id} atoms")
        
        # Show only Ca 918 as a prominent sphere
        ca918 = f"{model_id}:918"
        run(session, f"show {ca918} atoms")
        run(session, f"style {ca918} sphere")
        run(session, f"size {ca918} atomRadius 1.8")
        run(session, f"color {ca918} darkgreen")
        
        # Add label for Ca 918
        run(session, f'label {ca918} text "Ca²⁺ 918" color darkgreen size 12')
        
        # Make sure no other atoms or bonds are shown
        run(session, f"hide {model_id} bonds")
        logging.info("Only Ca 918 visible, all other atoms hidden")
        
    except Exception as e:
        logging.error(f"Failed to handle calcium ions: {e}")

    # Final cleanup and lighting
    try:
        # Ensure ONLY Ca 918 atoms are visible, everything else hidden
        run(session, f"hide {model_id} atoms")
        run(session, f"show {ca918} atoms")
        
        # Clean lighting for clear visualization
        run(session, "lighting soft")
        logging.info("Final visualization setup complete")
    except Exception as e:
        logging.error(f"Failed final setup: {e}")

    # Save outputs
    run(session, "view all")
    run(session, "zoom 1.25")
    
    # Create path-specific filename
    pair_name = path_data['pair'].replace(' ', '-').replace('→', '-').replace('→', '-')
    base = os.path.join(output_dir, f"path")
    try:
        logging.info(f"Saving single path to: {base}")
        run(session, f'save "{base}.png" supersample 3 width 1200 height 900')
        logging.info("High quality screenshot saved")
        run(session, f'save "{base}.cxs"')
        logging.info(f"Successfully saved single path: {base}")
    except Exception as e:
        logging.error(f"Save failed for single path {base}: {e}")

    # Clean up
    try:
        run(session, f"close {model_id}")
        logging.info(f"Closed model {model_id}")
    except Exception as e:
        logging.error(f"Failed to close model {model_id}: {e}")

def visualize_paths(session,
                    system_name: str,
                    category: str,
                    paths: list[dict],
                    output_dir: str,
                    residue_mapping: dict):
    """
    Clean wireframe visualization with translucent colored network paths.
    White background, gray translucent backbone, thick colored paths.
    """
    
    print(f"=== VISUALIZE_PATHS CALLED: {system_name} - {category} ===")

    # Open structure
    pdb_file = (
        os.path.join(project_root, "Data/AF2_LM211_WT/calcium/frame1_CA.pdb")
        if system_name == "WT" else
        os.path.join(project_root, "Data/AF2_LM2_Y138H_11_Mutant/calcium/frame1_CA.pdb")
    )
    logging.info(f"Opening PDB file: {pdb_file}")
    try:
        run(session, f'open "{pdb_file}"')
        logging.info("PDB file opened successfully")
        
        # Get the current model number
        models = session.models.list()
        current_model = models[-1]
        model_id = f"#{current_model.id_string}"
        logging.info(f"Working with model: {model_id}")
    except Exception as e:
        logging.error(f"Failed to open PDB file: {e}")
        return

    # Basic scene setup - transparent background for raytracing
    run(session, "set bgColor transparent")
    run(session, "hide atoms")
    run(session, "hide cartoons")
    
    # Show protein structure properly FIRST
    logging.info("Setting up protein visualization...")
    try:
        run(session, f"show {model_id}")
        run(session, f"show {model_id} atoms")
        run(session, f"show {model_id} bonds")
        logging.info("Protein structure setup complete")
    except Exception as e:
        logging.error(f"Failed to setup protein structure: {e}")
        return
    
    # Set desired orientation - CONSISTENT for all models (AFTER showing objects)
    logging.info("Setting consistent orientation for all visualizations")
    run(session, "view all")  # Reset to default view first (now objects are displayed)
    run(session, "turn x 90")
    run(session, "turn z 90")
    run(session, "turn x 90")
    run(session, "turn z -10")
    
    # Set up wireframe backbone
    if not setup_wireframe_backbone(session, model_id):
        return
    
    # Thicker translucent colored tubes for network paths
    # Colors chosen for maximum distinction from each other and from Ca 918 (darkgreen) and residue 101 (magenta)
    palette = [
        "royalblue","crimson","orange","teal","maroon",
        "gold","indigo","brown","coral","steelblue",
        "orangered","darkslategray","darkcyan","darkkhaki","firebrick",
        "chocolate","wheat","lightcoral","salmon","lightsteelblue",
        "khaki","tan","navy"
    ]
    
    for i, p in enumerate(paths):
        try:
            seg = list(dict.fromkeys(p["residues"]))
            clr = palette[i % len(palette)]
            
            logging.info(f"Styling path {i+1}: residues {seg} with color {clr}")
            setup_path_tube(session, model_id, seg, clr, f"path_tube_{i}", 0.6, 50)  # Increased from 0.45 to 0.6
        except Exception as e:
            logging.error(f"Failed to style path {i+1}: {e}")
            continue

    # Residue 101 as SUPER prominent magenta tube (overlays on top of everything)
    try:
        # Create an even larger white shadow for maximum contrast  
        setup_path_tube(session, model_id, [101], "white", "res101_shadow", 1.0, 0)
        # Create a very prominent magenta tube on top
        setup_path_tube(session, model_id, [101], "magenta", "res101_tube", 0.9, 0)
        
        # Add very prominent label for residue 101
        label_txt = residue_mapping.get("101", f"{get_resname(current_model, 101)}101")
        run(session, f'label {model_id}:101@CA text "{label_txt}" color magenta size 18 height 2.0')
        logging.info("Residue 101 styled with MAXIMUM visibility (large white shadow + large magenta overlay)")
    except Exception as e:
        logging.error(f"Failed to style residue 101: {e}")

    # Show ONLY Ca 918, hide all other atoms
    try:
        # Hide ALL atoms first
        run(session, f"hide {model_id} atoms")
        
        # Show only Ca 918 as a prominent sphere
        ca918 = f"{model_id}:918"
        run(session, f"show {ca918} atoms")
        run(session, f"style {ca918} sphere")
        run(session, f"size {ca918} atomRadius 1.8")
        run(session, f"color {ca918} darkgreen")
        
        # Add label for Ca 918
        run(session, f'label {ca918} text "Ca²⁺ 918" color darkgreen size 12')
        
        # Make sure no other atoms or bonds are shown
        run(session, f"hide {model_id} bonds")
        logging.info("Only Ca 918 visible, all other atoms hidden")
        
    except Exception as e:
        logging.error(f"Failed to handle calcium ions: {e}")

    # Final cleanup and lighting
    try:
        # Ensure ONLY Ca 918 atoms are visible, everything else hidden
        run(session, f"hide {model_id} atoms")
        run(session, f"show {ca918} atoms")
        
        # Clean lighting for clear visualization
        run(session, "lighting soft")
        logging.info("Final visualization setup complete")
    except Exception as e:
        logging.error(f"Failed final setup: {e}")

    # Save outputs
    run(session, "view all")
    run(session, "zoom 1.25")
    
    base = os.path.join(output_dir, f"{category}_{system_name}")
    try:
        logging.info(f"Saving to: {base}")
        run(session, f'save "{base}.png" supersample 3 width 1200 height 900')
        logging.info("High quality screenshot saved")
        run(session, f'save "{base}.cxs"')
        logging.info(f"Successfully saved: {base}")
    except Exception as e:
        logging.error(f"Save failed for {base}: {e}")

    # Clean up
    try:
        run(session, f"close {model_id}")
        logging.info(f"Closed model {model_id}")
    except Exception as e:
        logging.error(f"Failed to close model {model_id}: {e}")


def main(session):
    """
    Main function to generate all visualizations in ChimeraX.
    
    Args:
        session: ChimeraX session
    """
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_name = f"chimerax_visualization_schematic_{datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(filename=os.path.join(log_dir, log_file_name),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')

    markdown_file = os.path.join(project_root, 'analysis_results/reports/optimal_paths_details.md')
    
    # Define and create the output directory
    base_output_dir = os.path.join(project_root, 'analysis_results/chimera_visualizations_schematics/schematics')
    os.makedirs(base_output_dir, exist_ok=True)

    all_paths = parse_optimal_paths(markdown_file)

    wt_mapping_file = os.path.join(project_root, 'Data/residue_mapping_WT.csv')
    mutant_mapping_file = os.path.join(project_root, 'Data/residue_mapping_Mutant.csv')
    wt_mapping = load_residue_mapping(wt_mapping_file)
    mutant_mapping = load_residue_mapping(mutant_mapping_file)

    # Group paths by system and category
    grouped_paths = {}
    for path in all_paths:
        key = (path['System'], path['Category'])
        if key not in grouped_paths:
            grouped_paths[key] = []
        grouped_paths[key].append({'residues': path['Optimal Path Residues'], 'pair': path['Residue Pair']})

    # Generate individual path visualizations
    logging.info("Starting individual path visualizations...")
    for (system, category), paths in grouped_paths.items():
        # Create category-specific directory
        category_dir = os.path.join(base_output_dir, f"{category}_{system}")
        os.makedirs(category_dir, exist_ok=True)
        
        residue_mapping = wt_mapping if system == 'WT' else mutant_mapping
        
        for path_data in paths:
            # Create path-specific subdirectory
            pair_name = path_data['pair'].replace(' ', '-').replace('→', '-').replace('→', '-')
            path_dir = os.path.join(category_dir, pair_name)
            os.makedirs(path_dir, exist_ok=True)
            
            logging.info(f"Visualizing individual path: {system} - {category} - {path_data['pair']}")
            visualize_single_path(session, system, category, path_data, path_dir, residue_mapping)

    # Generate combined visualizations
    logging.info("Starting combined path visualizations...")
    for (system, category), paths in grouped_paths.items():
        logging.info(f"Visualizing {system} - {category.replace('_', ' ')}...")
        residue_mapping = wt_mapping if system == 'WT' else mutant_mapping
        visualize_paths(session, system, category, paths, base_output_dir, residue_mapping)

    # Generate combined visualizations for high contact categories
    for system in ['WT', 'Mutant']:
        logging.info(f"Visualizing {system} - Combined High Contact...")
        combined_paths = []
        for category in ['high_contact_both', 'high_contact_WT', 'high_contact_Mutant']:
            key = (system, category)
            if key in grouped_paths:
                combined_paths.extend(grouped_paths[key])
        
        if combined_paths:
            residue_mapping = wt_mapping if system == 'WT' else mutant_mapping
            visualize_paths(session, system, 'combined_high_contact', combined_paths, base_output_dir, residue_mapping)

# Call main function with session
main(session)
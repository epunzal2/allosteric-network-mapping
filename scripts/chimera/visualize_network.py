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

from visualize_residues import visualize_residues

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

def get_resname(session, resid, chain_id=None):
    model = session.models.list()[0]        # first atomic model (#1)
    res   = model.residues                  # a Residues array

    mask = (res.numbers == resid)
    if chain_id is not None:
        mask &= (res.chain_ids == chain_id)

    return res.names[mask][0] if mask.any() else ""

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
    
    print(f"=== VISUALIZE_PATHS CALLED: {system_name} - {category} ===")  # Debug

    # ── 0. open structure ───────────────────────────────────────────────
    pdb_file = (
        os.path.join(project_root, "Data/AF2_LM211_WT/calcium/frame1_CA.pdb")
        if system_name == "WT" else
        os.path.join(project_root,
                     "Data/AF2_LM2_Y138H_11_Mutant/calcium/frame1_CA.pdb")
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
        
        # Debug: Check what was loaded
        run(session, "info models")
        
    except Exception as e:
        logging.error(f"Failed to open PDB file: {e}")
        return

    # ── 1. basic scene setup ────────────────────────────────────────────
    run(session, "set bgColor white")
    run(session, "hide atoms")
    run(session, "hide cartoons")
    
    # ── 2. Show protein structure properly ─────────────────────────────
    # This is a full atomic structure, not just CA atoms
    logging.info("Setting up protein visualization...")
    
    try:
        # Show all atoms and bonds
        run(session, f"show {model_id}")
        run(session, f"show {model_id} atoms")
        run(session, f"show {model_id} bonds")
        run(session, f"style {model_id} stick")
        
        logging.info("Protein structure setup complete")
        
    except Exception as e:
        logging.error(f"Failed to setup protein structure: {e}")
        return
    
    # ── 3. Gray translucent wireframe backbone ─────────────────────────
    logging.info("Setting up wireframe backbone...")
    try:
        # Hide all atoms, show only bonds as thin gray sticks
        run(session, f"hide {model_id} atoms")
        run(session, f"hide {model_id} cartoons")  # Make sure no ribbons
        run(session, f"show {model_id} bonds")
        run(session, f"style {model_id} stick")
        
        # Make all sticks thin and gray
        run(session, f"size {model_id} stickRadius 0.06")  # Very thin
        run(session, f"color {model_id} gray")
        run(session, f"transparency {model_id} 85")  # Very transparent
        
        # No silhouettes for clean look
        run(session, "graphics silhouettes false")
        logging.info("Wireframe backbone setup complete")
        
    except Exception as e:
        logging.error(f"Failed to setup backbone: {e}")
        return
    
    # ── 4. Thicker translucent colored sticks for network paths ────────
    palette = [
        "royalblue","crimson","forestgreen","darkorange","purple","teal",
        "maroon","darkslateblue","darkgoldenrod","indigo","darkred","brown",
        "mediumvioletred","darkgreen","chocolate","steelblue","orangered",
        "darkslategray","darkmagenta","darkblue","darkcyan","darkkhaki"
    ]
    
    for i, p in enumerate(paths):
        try:
            seg = list(dict.fromkeys(p["residues"]))
            sel = f"{model_id}:{','.join(map(str, seg))}"
            clr = palette[i % len(palette)]
            
            logging.info(f"Styling path {i+1}: residues {seg} with color {clr}")
            
            # Make path sticks thicker, colored, and translucent
            run(session, f"size {sel} stickRadius 0.35")  # Much thicker
            run(session, f"color {sel} {clr}")
            run(session, f"transparency {sel} 30")  # Translucent so colors don't block
            
        except Exception as e:
            logging.error(f"Failed to style path {i+1}: {e}")
            continue

    # ── 5. Residue 101 as thick magenta stick ──────────────────────────
    try:
        res101 = f"{model_id}:101"
        run(session, f"size {res101} stickRadius 0.35")  # Same thickness as paths
        run(session, f"color {res101} magenta")
        run(session, f"transparency {res101} 30")  # Translucent
        
        # Add label for residue 101
        label_txt = residue_mapping.get("101", f"{get_resname(session,101)}101")
        run(session, f'label {model_id}:101@CA text "{label_txt}" color magenta')
        logging.info("Residue 101 styled successfully")
        
    except Exception as e:
        logging.error(f"Failed to style residue 101: {e}")

    # ── 6. Show only Ca 918, hide all other calcium ions ──────────────
    try:
        # Hide all calcium ions first
        run(session, f"hide {model_id} & :CA atoms")  # Hide all calcium atoms
        
        # Show only Ca 918 as a sphere
        ca918 = f"{model_id}:918"
        run(session, f"show {ca918} atoms")
        run(session, f"style {ca918} sphere")
        run(session, f"size {ca918} atomRadius 1.5")
        run(session, f"color {ca918} darkgreen")
        
        # Make sure no bonds are shown for calcium ions
        run(session, f"hide {model_id} & :CA bonds")
        logging.info("Calcium ions handled successfully")
        
    except Exception as e:
        logging.error(f"Failed to handle calcium ions: {e}")

    # ── 7. Final cleanup and lighting ──────────────────────────────────
    try:
        # Ensure only protein bonds are visible as sticks
        run(session, f"hide {model_id} atoms")  # Hide all atoms
        run(session, f"show {ca918} atoms")  # Show only Ca 918
        run(session, f"show {model_id} bonds")  # Show protein bonds
        run(session, f"hide {model_id} & :CA bonds")  # Hide calcium ion bonds
        
        # Clean lighting for clear visualization
        run(session, "lighting soft")
        logging.info("Final visualization setup complete")
        
    except Exception as e:
        logging.error(f"Failed final setup: {e}")

    # ── 8. save outputs ─────────────────────────────────────────────────
    base = os.path.join(output_dir, f"{category}_{system_name}")
    try:
        logging.info(f"Saving to: {base}")
        run(session, f'save "{base}.png" width 800 height 600')
        run(session, f'save "{base}.cxs"')
        logging.info(f"Successfully saved: {base}")
    except Exception as e:
        logging.error(f"Save failed for {base}: {e}")

    # ── 9. clean up ─────────────────────────────────────────────────────
    try:
        run(session, f"close {model_id}")  # Close only this model, not all
        logging.info(f"Closed model {model_id}")
    except Exception as e:
        logging.error(f"Failed to close model {model_id}: {e}")


def main(session):
    """
    Main function to generate all visualizations in ChimeraX.
    """
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_name = f"chimerax_visualization_{datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(filename=os.path.join(log_dir, log_file_name),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')

    markdown_file = os.path.join(project_root, 'analysis_results/reports/optimal_paths_details.md')
    
    # Define and create the output directory
    output_dir = os.path.join(project_root, 'analysis_results/chimera_visualizations_schematics/')
    os.makedirs(output_dir, exist_ok=True)

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

    # Generate individual visualizations
    for (system, category), paths in grouped_paths.items():
        logging.info(f"Visualizing {system} - {category.replace('_', ' ')}...")
        residue_mapping = wt_mapping if system == 'WT' else mutant_mapping
        visualize_paths(session, system, category, paths, output_dir, residue_mapping)

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
            visualize_paths(session, system, 'combined_high_contact', combined_paths, output_dir, residue_mapping)

main(session)
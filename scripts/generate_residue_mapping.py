import os
import logging
import csv
import argparse
import re

# --- Mappings ---
RES_MAP_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
RES_MAP_1_TO_3 = {v: k for k, v in RES_MAP_3_TO_1.items()}

SUBUNIT_CONVERSION = {
    'alpha': {'offset': -37, 'prefix': 'a', 'range': (1, 305)},
    'beta':  {'offset': 277, 'prefix': 'b', 'range': (306, 612)},
    'gamma': {'offset': 576, 'prefix': 'g', 'range': (613, 917)}
}
PREFIX_TO_SUBUNIT = {v['prefix']: k for k, v in SUBUNIT_CONVERSION.items()}

# --- Logging Setup ---
def setup_logging():
    """Sets up logging to a file."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(log_dir, 'residue_mapping.log'),
        filemode='w'
    )
    return logging.getLogger(__name__)

# --- Conversion Functions ---
def get_new_id_from_full_orig_label(full_orig_label):
    """Converts a full original label (e.g., 'aD262') to its new residue ID."""
    match = re.match(r"([abg])([A-Za-z]{1,3})(\d+)", full_orig_label)
    if not match:
        return None, "Invalid label format."

    prefix = match.group(1).lower()
    original_id = int(match.group(3))

    subunit_name = PREFIX_TO_SUBUNIT.get(prefix)
    if not subunit_name:
        return None, "Invalid subunit prefix."

    offset = SUBUNIT_CONVERSION[subunit_name]['offset']
    new_id = original_id + offset
    return new_id, None

def get_full_orig_label_from_new_id(new_id, res_name_3_letter):
    """Converts a new residue ID and 3-letter name to a full original label."""
    for subunit_name, info in SUBUNIT_CONVERSION.items():
        start, end = info['range']
        if start <= new_id <= end:
            offset = info['offset']
            prefix = info['prefix']
            original_id = new_id - offset
            return f"{prefix}{res_name_3_letter}{original_id}", None
    return None, "New ID is out of defined subunit ranges."

# --- Core Functions ---
def generate_residue_mapping_csv(pdb_file_path, output_csv_path, logger):
    """Generates the residue mapping CSV file."""
    logger.info(f"Generating residue mapping CSV from {pdb_file_path} to {output_csv_path}")
    
    residues = []
    processed_residues = set()

    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                try:
                    resid = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse ATOM line: {line.strip()}")
                    continue
                
                if (resid, chain_id) not in processed_residues:
                    processed_residues.add((resid, chain_id))
                    
                    full_label, err = get_full_orig_label_from_new_id(resid, res_name)
                    if err:
                        logger.warning(f"Skipping resid {resid}: {err}")
                        continue

                    match = re.match(r"([abg])([A-Za-z]{3})(\d+)", full_label)
                    prefix, _, resid_orig_str = match.groups()
                    resid_orig = int(resid_orig_str)
                    subunit_name = PREFIX_TO_SUBUNIT[prefix]
                    
                    residues.append({
                        'resid': resid,
                        'res': res_name,
                        'res_1let': RES_MAP_3_TO_1.get(res_name, 'X'),
                        'resid_orig': resid_orig,
                        'subunit': subunit_name,
                        'subunit_greek': {'alpha': 'α', 'beta': 'β', 'gamma': 'γ'}[subunit_name],
                        'orig_label': f"{res_name}{resid_orig}",
                        'full_orig_label': full_label
                    })

    header = ['resid', 'res', 'res_1let', 'resid_orig', 'subunit', 'subunit_greek', 'orig_label', 'full_orig_label']
    with open(output_csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(residues)
    
    print(f"Successfully created {output_csv_path}")
    logger.info(f"Successfully created CSV file at: {output_csv_path}")

def reindex_pdb_to_orig(pdb_in_path, pdb_out_path, logger):
    """Creates a new PDB file with original residue IDs and subunit chain IDs."""
    logger.info(f"Re-indexing PDB to original IDs: {pdb_in_path} -> {pdb_out_path}")
    
    with open(pdb_in_path, 'r') as fin, open(pdb_out_path, 'w') as fout:
        for line in fin:
            if line.startswith('ATOM'):
                try:
                    new_id = int(line[22:26].strip())
                    
                    original_id = -1
                    subunit_prefix = ' '
                    for info in SUBUNIT_CONVERSION.values():
                        if info['range'][0] <= new_id <= info['range'][1]:
                            original_id = new_id - info['offset']
                            subunit_prefix = info['prefix']
                            break
                    
                    if original_id == -1:
                         fout.write(line)
                         continue

                    new_line = f"{line[:21]}{subunit_prefix}{original_id:4d}{line[26:]}"
                    fout.write(new_line)
                except (ValueError, IndexError):
                    fout.write(line)
            else:
                fout.write(line)
    
    print(f"Successfully created re-indexed PDB: {pdb_out_path}")

def reindex_pdb_to_new(pdb_in_path, pdb_out_path, logger):
    """Creates a new PDB file with new residue IDs from a PDB with original IDs and subunit chain IDs."""
    logger.info(f"Re-indexing PDB to new IDs: {pdb_in_path} -> {pdb_out_path}")

    with open(pdb_in_path, 'r') as fin, open(pdb_out_path, 'w') as fout:
        for line in fin:
            if line.startswith('ATOM'):
                try:
                    original_id = int(line[22:26].strip())
                    chain_id = line[21].strip()
                    
                    subunit_name = PREFIX_TO_SUBUNIT.get(chain_id)
                    if not subunit_name:
                        fout.write(line)
                        continue

                    offset = SUBUNIT_CONVERSION[subunit_name]['offset']
                    new_id = original_id + offset
                    
                    # Here, we might want to restore the original chain ID if it was different,
                    # but for now, we'll just leave it as the subunit prefix.
                    new_line = f"{line[:22]}{new_id:4d}{line[26:]}"
                    fout.write(new_line)
                except (ValueError, IndexError):
                    fout.write(line)
            else:
                fout.write(line)

    print(f"Successfully created re-indexed PDB: {pdb_out_path}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="A script for residue mapping, ID conversion, and PDB re-indexing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- 'map' command ---
    parser_map = subparsers.add_parser('map', help='Generate a residue mapping CSV from a PDB file.')
    parser_map.add_argument('pdb_file', help='Path to the input PDB file.')
    parser_map.add_argument('csv_file', help='Path to the output CSV file in Data/.')

    # --- 'convert' command ---
    parser_convert = subparsers.add_parser('convert', help='Convert between original and new residue IDs.')
    group = parser_convert.add_mutually_exclusive_group(required=True)
    group.add_argument('--to-new-id', metavar='LABEL', help="Convert full original label (e.g., 'aD262') to new ID.")
    group.add_argument('--to-orig-label', nargs=2, metavar=('ID', 'RES'), help="Convert new ID and 3-letter res name (e.g., 101 TYR) to original label.")

    # --- 'reindex' command ---
    parser_reindex = subparsers.add_parser('reindex', help='Create a new PDB file with re-indexed residue IDs.')
    parser_reindex.add_argument('input_pdb', help='Path to the input PDB file.')
    parser_reindex.add_argument('--direction', choices=['to_orig', 'to_new'], default='to_orig', help="Direction of re-indexing.")
    parser_reindex.add_argument('--output', help='Optional: Path for the new re-indexed PDB file.')

    args = parser.parse_args()
    logger = setup_logging()
    logger.info(f"Command: {args.command}")

    if args.command == 'map':
        output_csv_path = os.path.join('Data', args.csv_file)
        generate_residue_mapping_csv(args.pdb_file, output_csv_path, logger)
    
    elif args.command == 'convert':
        # ... (conversion logic remains the same)
        if args.to_new_id:
            new_id, err = get_new_id_from_full_orig_label(args.to_new_id)
            if err:
                print(f"Error: {err}")
            else:
                print(f"Original Label: '{args.to_new_id}' -> New ID: {new_id}")
        
        elif args.to_orig_label:
            try:
                new_id = int(args.to_orig_label[0])
                res_name = args.to_orig_label[1].upper()
                if len(res_name) != 3 or res_name not in RES_MAP_3_TO_1:
                    print("Error: Please provide a valid 3-letter amino acid code.")
                    return

                label, err = get_full_orig_label_from_new_id(new_id, res_name)
                if err:
                    print(f"Error: {err}")
                else:
                    print(f"New ID: {new_id} ({res_name}) -> Original Label: '{label}'")
            except ValueError:
                print("Error: The new ID must be an integer.")

    elif args.command == 'reindex':
        output_pdb_path = args.output
        if not output_pdb_path:
            base, ext = os.path.splitext(args.input_pdb)
            suffix = "_reindexed_orig" if args.direction == 'to_orig' else "_reindexed_new"
            output_pdb_path = f"{base}{suffix}{ext}"
        
        if args.direction == 'to_orig':
            reindex_pdb_to_orig(args.input_pdb, output_pdb_path, logger)
        else:
            reindex_pdb_to_new(args.input_pdb, output_pdb_path, logger)

if __name__ == "__main__":
    main()
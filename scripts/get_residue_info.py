import MDAnalysis as mda
import argparse

# Standard amino acid 3-to-1 letter codes
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'CYX': 'C', 'CYM': 'C', 'CY2': 'C', # Cysteine variants (disulfide-bonded, metal-bound)
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', # Histidine variants (protonation states)
    'LYN': 'K', # Lysine variant
}

def resname_to_one_letter(resname):
    """Converts a 3-letter residue name to 1-letter code using AA_MAP."""
    return AA_MAP.get(resname.upper(), 'X') # 'X' for unknown or unmapped

def get_residue_details_range(pdb_file, start_resid, end_resid):
    """
    Loads a PDB file and prints details for residues within a given PDB ID range.
    """
    try:
        u = mda.Universe(pdb_file)
    except Exception as e:
        print(f"Error loading PDB file {pdb_file}: {e}")
        return []

    results = []
    protein_residues = u.select_atoms("protein").residues
    # Create a dictionary for quick lookup of residues by their PDB ID
    residue_map = {res.resid: res for res in protein_residues}

    for resid_to_check in range(start_resid, end_resid + 1):
        found_res = residue_map.get(resid_to_check)
        
        if found_res:
            res_name = found_res.resname
            one_letter = resname_to_one_letter(res_name)
            results.append(f"PDB Residue ID: {found_res.resid}, Name: {res_name}, 1-Letter: {one_letter}")
        else:
            results.append(f"PDB Residue ID: {resid_to_check} not found in protein selection.")
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get details for a range of PDB residue IDs.")
    parser.add_argument("pdb_file", help="Path to the PDB file.")
    parser.add_argument("start_resid", type=int, help="Starting PDB residue ID.")
    parser.add_argument("end_resid", type=int, help="Ending PDB residue ID.")
    
    args = parser.parse_args()

    if args.start_resid > args.end_resid:
        print("Error: Start PDB_ID must be less than or equal to End PDB_ID.")
    else:
        details = get_residue_details_range(args.pdb_file, args.start_resid, args.end_resid)
        if details:
            for line in details:
                print(line)

import MDAnalysis as mda
import argparse

# Standard amino acid 3-to-1 letter codes
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'CYX': 'C', 'CYM': 'C', 'CY2': 'C', 
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', # Histidine variants (often protonation states)
    'LYN': 'K', # Lysine variant
}

def resname_to_one_letter(resname):
    """Converts a 3-letter residue name to 1-letter code using AA_MAP."""
    return AA_MAP.get(resname.upper(), 'X') # 'X' for unknown or unmapped

def find_pattern_in_pdb(pdb_file, pattern):
    """
    Loads a PDB file, extracts its protein sequence (handling CYX and other variants), 
    and searches for a given pattern.
    """
    # Flag to indicate if an error occurred that might affect "not found" message
    find_pattern_in_pdb.sequence_construction_error = False # Initialize flag

    try:
        # Load PDB with in_memory=True for potentially faster access if file is read multiple times
        # though for a single pass, it might not matter much.
        u = mda.Universe(pdb_file, in_memory=True) 
    except Exception as e:
        print(f"Error loading PDB file {pdb_file}: {e}")
        find_pattern_in_pdb.sequence_construction_error = True
        return []

    protein_selection = u.select_atoms("protein")
    if not protein_selection.n_atoms > 0:
        print(f"No protein atoms found in {pdb_file}.")
        find_pattern_in_pdb.sequence_construction_error = True
        return []

    # Use .n_residues to check if there are any residues
    if not protein_selection.residues.n_residues > 0: 
        print(f"No residues found in protein selection for {pdb_file}.")
        find_pattern_in_pdb.sequence_construction_error = True
        return []

    full_sequence_one_letter = "" # Initialize
    try:
        residue_objects = protein_selection.residues # Get residue objects once
        full_sequence_one_letter = "".join([resname_to_one_letter(r.resname) for r in residue_objects])
        
        # Check for unknown residues only if 'X' is in the sequence
        if 'X' in full_sequence_one_letter:
            # Efficiently find unique unknown residue names
            unknown_res_names = set()
            for r_idx, one_letter_code in enumerate(full_sequence_one_letter):
                if one_letter_code == 'X':
                    unknown_res_names.add(residue_objects[r_idx].resname)
            if unknown_res_names: # Only print if there are actual unknown names
                print(f"Warning: Unknown residue names encountered and mapped to 'X': {sorted(list(unknown_res_names))}")
    
    except Exception as e:
        print(f"Error during manual sequence construction: {e}")
        find_pattern_in_pdb.sequence_construction_error = True
        return [] 

    found_pdb_resids = []
    start_search_index = 0
    
    while True:
        match_index_in_sequence = full_sequence_one_letter.find(pattern, start_search_index)
        if match_index_in_sequence == -1:
            break
        
        # Map index in sequence string back to the PDB residue ID
        start_residue_of_match = protein_selection.residues[match_index_in_sequence]
        found_pdb_resids.append(start_residue_of_match.resid)
        start_search_index = match_index_in_sequence + 1 
        
    return found_pdb_resids

# Initialize the flag on the function object, accessible in the __main__ block
find_pattern_in_pdb.sequence_construction_error = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find an amino acid sequence pattern in a PDB file.")
    parser.add_argument("pdb_file", help="Path to the PDB file.")
    parser.add_argument("pattern", help="Amino acid sequence pattern (one-letter code).")
    args = parser.parse_args()

    matches = find_pattern_in_pdb(args.pdb_file, args.pattern.upper())

    if matches:
        print(f"Pattern '{args.pattern.upper()}' found starting at PDB residue ID(s): {matches}")
    else:
        # Only print "not found" if no earlier error (like loading error or no protein)
        # prevented proper sequence checking. Errors during construction are handled by the flag.
        if not find_pattern_in_pdb.sequence_construction_error:
            print(f"Pattern '{args.pattern.upper()}' not found in {args.pdb_file}.")
        # If sequence_construction_error is True, an informative error message was already printed.

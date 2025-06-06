import re
import argparse

# Mapping for single-letter to three-letter amino acid codes
AMINO_ACID_MAP = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

# Residue ID conversion mapping
# Based on:
# Alpha: original_id - 37
# Beta:  original_id + 277
# Gamma: original_id + 576
SUBUNIT_CONVERSION_MAP = {
    'a': {'name': 'alpha', 'offset': -37},
    'b': {'name': 'beta',  'offset': 277},
    'g': {'name': 'gamma', 'offset': 576}
}

def convert_residue_input(input_string):
    """
    Converts a residue input string (e.g., "aD262" or "bALA104") to its
    new residue ID, original ID, 3-letter amino acid code, and subunit.

    Args:
        input_string (str): The residue input string.

    Returns:
        tuple: (original_id, new_id, amino_acid_3_letter, subunit_name)
               Returns None if the input format is invalid.
    """
    # Regex to capture subunit, amino acid (1 or 3 letters), and residue number
    # Group 1: subunit (a, b, or g)
    # Group 2: amino acid (e.g., D or ALA)
    # Group 3: residue number (e.g., 262)
    match = re.match(r"([abg])([A-Za-z]{1,3})(\d+)", input_string)

    if not match:
        print(f"Error: Invalid input format: '{input_string}'")
        print("Expected format: e.g., 'aD262' or 'bALA104'")
        return None

    subunit_char = match.group(1).lower()
    amino_acid_input = match.group(2).upper()
    original_id_str = match.group(3)

    if subunit_char not in SUBUNIT_CONVERSION_MAP:
        print(f"Error: Invalid subunit character '{subunit_char}' in '{input_string}'. Must be 'a', 'b', or 'g'.")
        return None

    try:
        original_id = int(original_id_str)
    except ValueError:
        print(f"Error: Invalid residue number '{original_id_str}' in '{input_string}'. Must be an integer.")
        return None

    # Convert amino acid to 3-letter code if it's single letter
    if len(amino_acid_input) == 1:
        if amino_acid_input in AMINO_ACID_MAP:
            amino_acid_3_letter = AMINO_ACID_MAP[amino_acid_input]
        else:
            print(f"Error: Invalid single-letter amino acid '{amino_acid_input}' in '{input_string}'.")
            return None
    elif len(amino_acid_input) == 3:
        # Check if the 3-letter code is valid (is a value in our map)
        if amino_acid_input in AMINO_ACID_MAP.values():
            amino_acid_3_letter = amino_acid_input
        else:
            # Or if it's a key (e.g. user typed 'MET' instead of 'M')
            # This case is less likely given the problem description but good for robustness
            found = False
            for k, v in AMINO_ACID_MAP.items():
                if v == amino_acid_input:
                    amino_acid_3_letter = v
                    found = True
                    break
            if not found:
                print(f"Error: Invalid three-letter amino acid '{amino_acid_input}' in '{input_string}'.")
                return None
    else:
        print(f"Error: Amino acid code '{amino_acid_input}' in '{input_string}' must be 1 or 3 letters.")
        return None

    conversion_info = SUBUNIT_CONVERSION_MAP[subunit_char]
    subunit_name = conversion_info['name']
    offset = conversion_info['offset']

    # Apply the conversion
    new_id = original_id + offset

    return original_id, new_id, amino_acid_3_letter, subunit_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert residue input string to new ID and details.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "residue_input",
        type=str,
        help="Residue input string.\n"
             "Format: <subunit_char><amino_acid><original_id>\n"
             "  <subunit_char>: 'a' (alpha), 'b' (beta), or 'g' (gamma)\n"
             "  <amino_acid>: 1-letter (e.g., D) or 3-letter (e.g., ALA) code\n"
             "  <original_id>: The original residue number\n"
             "Examples:\n"
             "  aD262\n"
             "  bALA104\n"
             "  gASN233"
    )

    args = parser.parse_args()
    input_str = args.residue_input

    result = convert_residue_input(input_str)

    if result:
        original_id, new_id, amino_acid, subunit = result
        print(f"\nInput: {input_str}")
        print("------------------------------------")
        print(f"Original Residue ID: {original_id}")
        print(f"Converted Residue ID: {new_id}")
        print(f"Amino Acid: {amino_acid}")
        print(f"Subunit: {subunit}")
        print("------------------------------------")
        print("Conversion logic:")
        print(f"  - Alpha (a): original_id - 37 = {original_id} - 37 = {original_id - 37 if subunit == 'alpha' else 'N/A'}")
        print(f"  - Beta  (b): original_id + 277 = {original_id} + 277 = {original_id + 277 if subunit == 'beta' else 'N/A'}")
        print(f"  - Gamma (g): original_id + 576 = {original_id} + 576 = {original_id + 576 if subunit == 'gamma' else 'N/A'}")

# Example Usage from command line:
# python convert_residue_id.py aD262
# python convert_residue_id.py bALA104
# python convert_residue_id.py gN233
# python convert_residue_id.py aASN80
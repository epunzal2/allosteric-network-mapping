#!/usr/bin/env python

import MDAnalysis as mda
import os
import argparse
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def validate_trajectory(topology_file, dcd_file):
    """Loads a DCD file with its topology using MDAnalysis to check its validity."""
    logger.info(f"Attempting to validate: {dcd_file}")
    logger.info(f"Using topology: {topology_file}")

    if not os.path.exists(topology_file):
        logger.error(f"Topology file not found: {topology_file}")
        return False
    if not os.path.exists(dcd_file):
        logger.error(f"DCD file not found: {dcd_file}")
        return False

    try:
        universe = mda.Universe(topology_file, dcd_file)
        n_frames = len(universe.trajectory)
        n_atoms = len(universe.atoms)
        logger.info(f"SUCCESS: Loaded '{os.path.basename(dcd_file)}' with '{os.path.basename(topology_file)}'.")
        logger.info(f"         Frames: {n_frames}, Atoms: {n_atoms}")
        if n_frames == 0:
            logger.warning(f"The trajectory contains 0 frames.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error loading '{os.path.basename(dcd_file)}' with '{os.path.basename(topology_file)}':")
        logger.error(f"      {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate DCD trajectory files using MDAnalysis."
    )
    parser.add_argument(
        "-t", "--topology", 
        help="Path to the topology file (e.g., PDB, PARM7)."
    )
    parser.add_argument(
        "-d", "--dcd", 
        help="Path to the DCD trajectory file."
    )
    parser.add_argument(
        "--dir", 
        help="Directory containing processed DCD files (e.g., 'system/condition/processed/'). "
             "Script will look for '*_processed.dcd' and corresponding '*_processed_frame1.pdb' files."
    )

    args = parser.parse_args()

    validated_count = 0
    failed_count = 0

    if args.topology and args.dcd:
        if validate_trajectory(args.topology, args.dcd):
            validated_count += 1
        else:
            failed_count += 1
    elif args.dir:
        search_dir = os.path.abspath(args.dir)
        logger.info(f"Searching for DCD files in: {search_dir}")
        dcd_files = glob.glob(os.path.join(search_dir, "*_processed.dcd"))
        
        if not dcd_files:
            logger.warning(f"No '*_processed.dcd' files found in {search_dir}.")
        else:
            logger.info(f"Found {len(dcd_files)} DCD file(s) to validate.")

        for dcd_file_path in dcd_files:
            base_name = os.path.basename(dcd_file_path).replace("_processed.dcd", "")
            # Infer PDB name based on the convention from process_all_dcds.sh
            # It could be <base>_processed_frame1.pdb or <base_from_original_dcd>_processed_frame1.pdb
            # Let's try to be robust by checking a common pattern
            potential_pdb_name = base_name + "_processed_frame1.pdb"
            pdb_file_path = os.path.join(os.path.dirname(dcd_file_path), potential_pdb_name)
            
            # Fallback for cases like 'af2_lm211_wt_ca2_processed_frame1.pdb' where 'ca2' was part of original dcd name
            if not os.path.exists(pdb_file_path):
                # This part is a bit tricky as the original DCD name isn't directly known here
                # We assume the PDB file shares the same prefix up to the point before _processed.dcd
                # For simplicity, we'll stick to the primary convention for now.
                # A more robust solution might involve listing all PDBs and matching.
                logger.warning(f"Could not automatically determine topology for {dcd_file_path} using simple convention.")
                logger.warning(f"Tried: {pdb_file_path}")
                # Attempt to find any PDB that matches the base prefix if the strict one fails
                # This handles cases where the DCD might have been named slightly differently originally
                # e.g. af2_lm211_wt_ca2.dcd -> af2_lm211_wt_ca2_processed.dcd -> af2_lm211_wt_ca2_processed_frame1.pdb
                
                # More robust PDB finding:
                # List all PDBs in the directory and find one that matches the DCD's base name
                pdb_glob_pattern = os.path.join(os.path.dirname(dcd_file_path), base_name + "*_frame1.pdb")
                potential_pdbs = glob.glob(pdb_glob_pattern)
                if potential_pdbs:
                    pdb_file_path = potential_pdbs[0] # Take the first match
                    logger.info(f"Found potential PDB via glob: {pdb_file_path}")
                else:
                    logger.error(f"Still could not find a suitable PDB for {dcd_file_path} using glob pattern {pdb_glob_pattern}")
                    failed_count +=1
                    continue
            
            if validate_trajectory(pdb_file_path, dcd_file_path):
                validated_count += 1
            else:
                failed_count += 1
    else:
        logger.error("Please specify either a topology/DCD pair (-t and -d) or a directory (--dir).")
        parser.print_help()

    logger.info(f"\n--- Validation Summary ---")
    logger.info(f"Successfully validated: {validated_count}")
    logger.info(f"Failed to validate:   {failed_count}")
    logger.info(f"------------------------")


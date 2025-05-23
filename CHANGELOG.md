# Changelog

## 2025-05-07

### Fallback Logic for Pruning and Run Configuration Reporting
- Implemented robust fallback logic for `fragmentation_pruning` in `protein_network_analysis_updated.py`:
  - If no optimal path is found after standard pruning, the script lowers the Ec cutoff by 10% and retries.
  - If still no path, it switches to `original_ec` mode and attempts pathfinding again.
  - All fallback attempts are clearly logged in the output report.
- Added automatic reporting of the full run configuration at the end of every `analysis_report.txt`.
  - This includes all command-line arguments, their values, and input files used for each run.
- Refactored/cleaned main execution logic for clarity and maintainability.
- No changes to output file/directory conventions except for new configuration reporting.

---

# Analysis Log - MDM2 Pathfinding Comparison

**Date:** 2025-04-07 11:48 AM (America/New_York)

**Goal:** Analyze pathfinding between residues 25 and 109. Identify why the updated script failed initially.

**Run 1:** N/A
*   Command: N/A
*   Key Settings: N/A
*   Result: N/A

**Run 2: `protein_network_analysis_updated.py` (Initial Fail)**
*   Command: `python protein_network_analysis_updated.py trajectory_analysis_files/mdm2.pdb trajectory_analysis_files/mdm2.dcd 25 109 --cov_type=displacement_dot_mean`
*   Key Settings: C-beta/Gly-CA contacts, `displacement_dot_mean` covariance, `paper_critical` weight pruning (Ec=0.999575, kept |corr| <= 0.000425) applied after graph creation.
*   Result: **Failure**. No path found (disconnected graph). Reason: Different covariance method and extremely aggressive pruning.

**Run 3: `protein_network_analysis_updated.py` (Mean-Dot Cov, Paper Pruning)**
*   Command: `python protein_network_analysis_updated.py trajectory_analysis_files/mdm2.pdb trajectory_analysis_files/mdm2.dcd 25 109 --cov_type=displacement_mean_dot`
*   Key Settings: C-beta/Gly-CA contacts, `displacement_mean_dot` covariance, `paper_critical` weight pruning (Ec=0.893, kept |corr| <= 0.107) applied after graph creation.
*   Result: **Failure**. No path found (disconnected graph). Reason: Aggressive `paper_critical` pruning removed necessary edges.

**Run 4: `protein_network_analysis_updated.py` (Mean-Dot Cov, No Pruning)**
*   Command: `python protein_network_analysis_updated.py trajectory_analysis_files/mdm2.pdb trajectory_analysis_files/mdm2.dcd 25 109 --cov_type=displacement_mean_dot --pruning_method=none`
*   Result: **Success**. Path [25, 26, 109] found. Confirmed `paper_critical` pruning was the primary issue. Path differs from Run 1 due to different filtering/weighting/contact atoms.

**Plan 1: Modify `protein_network_analysis_updated.py` (Filtering)**
*   Changes:
    *   Add `--filtering_mode` argument (`contact_only`, `original_ec`, `fragmentation_pruning`). Default to `fragmentation_pruning` (current behavior).
    *   Implement `find_original_critical_ec` function (calculates covariance magnitude threshold targeting ~50% connectivity)
    *   Modify `build_graph` to optionally filter edges based on `abs(raw_covariance) >= original_Ec` when `filtering_mode == 'original_ec'`.
    *   Adjust main script logic to call new function and pass `Ec` to `build_graph`.
    *   Ensure `fragmentation_pruning` (current `paper_critical`) is skipped unless explicitly selected via `--filtering_mode`.
*   Status: **Completed**. Tested `--filtering_mode=original_ec`, found path [25, 26, 109]. Difference from Run 1 attributed to C-beta contacts affecting `Ec` calculation.

**Plan 2: Modify `protein_network_analysis_updated.py` (Contact Atoms)**
*   Goal: Add option to select atoms used for contact calculation (C-alpha or C-beta/Gly-CA).
*   Changes:
    *   Add `--contact_atoms` argument (`calpha`, `cbeta`). Default `cbeta`.
    *   Modify `select_atoms_for_contact` to return indices based on the choice.
    *   Ensure subsequent steps (contact calculation, CA selection for covariance) use the correct residue set derived from the chosen contact atoms.
*   Status: **Pending**.

---

## 2025-05-22

### Refactoring DCD Preprocessing Workflow

- **Automated DCD Processing:**
  - Introduced `scripts/process_all_dcds.sh`, a Bash script to automate the preprocessing of multiple DCD trajectory files across various protein systems and conditions.
  - The script iterates through specified data directories, automatically locates PARM7/PRMTOP topology files and DCD trajectory files.
- **Templated `cpptraj` Commands:**
  - Created `scripts/cpptraj_processing_template.in` to serve as a flexible template for `cpptraj` commands, with placeholders for input/output files.
- **Standardized Preprocessing Steps:**
  - The `cpptraj` process includes: stripping solvent (water), fixing periodic boundary conditions (`autoimage`), and RMS fitting to the C-alpha atoms of the first frame.
- **Organized Output Structure:**
  - Processed outputs (stripped DCD, first-frame PDB, and RMSD `.dat` file) are now consistently saved into a `processed/` subdirectory within each respective system/condition data directory (e.g., `Data/SystemName/ConditionName/processed/`).
- **Script Organization:**
  - All DCD processing helper scripts (`.sh` and `.in` files) have been consolidated into a new `scripts/` directory for better project organization.

### DCD Validation and Enhanced Logging

- **MDAnalysis-based DCD Validation:**
  - Developed `scripts/validate_dcd_mda.py`, a Python script utilizing `MDAnalysis` to check the integrity and loadability of DCD trajectory files against their corresponding PDB topology files.
  - The script uses Python's built-in `logging` module for its output.
- **Integrated Validation Workflow:**
  - The `validate_dcd_mda.py` script is now automatically invoked within `scripts/process_all_dcds.sh` immediately after each DCD file is processed by `cpptraj`, providing instant feedback on output validity.
- **Comprehensive Logging for `process_all_dcds.sh`:**
  - Implemented robust logging for the main `scripts/process_all_dcds.sh` script.
  - All standard output and standard error from the script are now redirected to a timestamped log file (e.g., `logs/process_all_dcds_YYYYMMDD_HHMMSS.log`) located in a `logs/` directory at the project root.
  - The `tee` command is used to ensure output is simultaneously displayed on the console and saved to the log file.

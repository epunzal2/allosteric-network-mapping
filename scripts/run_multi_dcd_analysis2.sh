#!/bin/bash

# Script to run protein network analysis for multiple DCD files (runs 1, 2, 3)
# using the same PDB file and specified parameters.

# --- Configuration (based on user example) ---
PDB_FILE="Data/AF2_LM211_WT/calcium/processed/af2_lm211_wt_calcium_1_processed_frame1.pdb"
DCD_FILE_BASENAME="Data/AF2_LM211_WT/calcium/processed/af2_lm211_wt_calcium"

START_RESIDUE="101"
END_RESIDUE="564"

# Base directory where run1, run2, run3 subdirectories will be created
# This is derived from the user's example output_dir, with '_multi' appended for clarity
BASE_OUTPUT_DIR="analysis_LM2/AF2_LM211_WT/101-564_dmd_oec_cb_cf0.3_multi"

COV_TYPE="displacement_mean_dot"
FILTERING_MODE="original_ec"
CONTACT_ATOMS="cbeta"
CONTACT_FREQ="0.3"

# Path to the Python analysis script, assuming this shell script is run from the project root
ANALYSIS_SCRIPT="protein_network_analysis_updated.py"

# --- Script Logic ---

echo "Starting multi-DCD analysis batch..."
echo "======================================================================"
echo "Static PDB File: ${PDB_FILE}"
echo "DCD File Basename: ${DCD_FILE_BASENAME}_<run_number>_processed.dcd"
echo "Residues: ${START_RESIDUE}-${END_RESIDUE}"
echo "Base Output Directory for this batch: ${BASE_OUTPUT_DIR}"
echo "Contact Frequency: ${CONTACT_FREQ}"
echo "Analysis Script: ${ANALYSIS_SCRIPT}"
echo "======================================================================"

# Create the main base output directory if it doesn't exist
mkdir -p "${BASE_OUTPUT_DIR}"

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "ERROR: Analysis script '${ANALYSIS_SCRIPT}' not found. Make sure this script is run from the project root." 
    exit 1
fi

if [ ! -f "${PDB_FILE}" ]; then
    echo "ERROR: PDB file '${PDB_FILE}' not found. Please check the path." 
    exit 1
fi

for i in 1 2 3
do
    RUN_DCD_FILE="${DCD_FILE_BASENAME}_${i}_processed.dcd"
    RUN_OUTPUT_SUBDIR="${BASE_OUTPUT_DIR}/run${i}"

    echo ""
    echo "--- Processing Run ${i} --- "
    echo "Input DCD File: ${RUN_DCD_FILE}"
    echo "Outputting to: ${RUN_OUTPUT_SUBDIR}"

    # Create the specific run output subdirectory
    mkdir -p "${RUN_OUTPUT_SUBDIR}"

    # Check if DCD file exists
    if [ ! -f "${RUN_DCD_FILE}" ]; then
        echo "WARNING: DCD file ${RUN_DCD_FILE} not found. Skipping Run ${i}."
        echo "----------------------------------------------------------------------"
        continue
    fi

    # Construct the command arguments in an array for safety
    COMMAND_ARGS=(
        "python" "${ANALYSIS_SCRIPT}"
        "${PDB_FILE}"
        "${RUN_DCD_FILE}"
        "${START_RESIDUE}"
        "${END_RESIDUE}"
        --output_dir "${RUN_OUTPUT_SUBDIR}"
        --cov_type "${COV_TYPE}"
        --filtering_mode "${FILTERING_MODE}"
        --contact_atoms "${CONTACT_ATOMS}"
        --contact_freq "${CONTACT_FREQ}"
    )

    echo "Executing command for Run ${i}:"
    echo "${COMMAND_ARGS[@]}" # Print the command array elements
    
    # Execute the command
    "${COMMAND_ARGS[@]}"

    if [ $? -eq 0 ]; then
        echo "Run ${i} completed successfully."
    else
        echo "ERROR: Run ${i} failed. Check logs in ${RUN_OUTPUT_SUBDIR}/analysis_report.txt"
    fi
    echo "----------------------------------------------------------------------"
done

echo ""
echo "Multi-DCD analysis batch finished."

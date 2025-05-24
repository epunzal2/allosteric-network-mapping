#!/bin/bash

# Script to run protein network analysis for multiple DCD files (runs 1, 2, 3)
# using the same PDB file and specified parameters.

# --- Configuration (based on user example) ---
# PDB files will be dynamically determined based on DCD_FILE_BASENAME and run number.
# Example: Data/AF2_LM211_WT/sodium/processed/af2_lm211_wt_sodium_1_processed_frame1.pdb
DCD_FILE_BASENAME="Data/AF2_LM211_WT/sodium/processed/af2_lm211_wt_sodium"

START_RESIDUE="101"
END_RESIDUE="225"

COV_TYPE="displacement_mean_dot"
FILTERING_MODE="original_ec"
CONTACT_ATOMS="cbeta"
CONTACT_FREQ="0.3"

# --- Determine Output Directory Dynamically ---
# Extract condition (e.g., sodium, calcium) from DCD_FILE_BASENAME
# Example: Data/AF2_LM211_WT/sodium/processed/af2_lm211_wt_sodium -> sodium
CONDITION_NAME=$(basename $(dirname $(dirname "${DCD_FILE_BASENAME}")))

# Determine abbreviations for path components based on analysis parameters
case "${COV_TYPE}" in
    "displacement_mean_dot") COV_ABBREV="dmd" ;;
    "pearson") COV_ABBREV="pea" ;;
    # Add more specific cases as needed
    *) COV_ABBREV="${COV_TYPE:0:3}" ;; # Default to first 3 chars
esac

case "${FILTERING_MODE}" in
    "original_ec") FIL_ABBREV="oec" ;;
    # Add more specific cases as needed
    *) FIL_ABBREV="${FILTERING_MODE:0:3}" ;; # Default to first 3 chars
esac

case "${CONTACT_ATOMS}" in
    "cbeta") ATOM_ABBREV="cb" ;;
    "calpha") ATOM_ABBREV="ca" ;;
    # Add more specific cases as needed
    *) ATOM_ABBREV="${CONTACT_ATOMS:0:2}" ;; # Default to first 2 chars
esac

# Construct the parameter-specific part of the output directory name
PARAMETERS_SUBDIR="${START_RESIDUE}-${END_RESIDUE}_${COV_ABBREV}_${FIL_ABBREV}_${ATOM_ABBREV}_cf${CONTACT_FREQ}_multi"

# Define the base output directory, incorporating the condition and dynamic parameters
BASE_OUTPUT_DIR="analysis_LM2/AF2_LM211_WT/${CONDITION_NAME}/${PARAMETERS_SUBDIR}"

# Path to the Python analysis script, assuming this shell script is run from the project root
ANALYSIS_SCRIPT="protein_network_analysis_updated.py"

# --- Script Logic ---

echo "Starting multi-DCD analysis batch..."
echo "======================================================================"
echo "PDB File Pattern: ${DCD_FILE_BASENAME}_<run_number>_processed_frame1.pdb"
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

for i in 1 2 3
do
    RUN_DCD_FILE="${DCD_FILE_BASENAME}_${i}_processed.dcd"
    RUN_PDB_FILE="${DCD_FILE_BASENAME}_${i}_processed_frame1.pdb"
    RUN_OUTPUT_SUBDIR="${BASE_OUTPUT_DIR}/run${i}"

    echo ""
    echo "--- Processing Run ${i} --- "
    echo "Input PDB File: ${RUN_PDB_FILE}"
    echo "Input DCD File: ${RUN_DCD_FILE}"
    echo "Outputting to: ${RUN_OUTPUT_SUBDIR}"

    # Create the specific run output subdirectory
    mkdir -p "${RUN_OUTPUT_SUBDIR}"

    # Check if PDB file exists
    if [ ! -f "${RUN_PDB_FILE}" ]; then
        echo "WARNING: PDB file ${RUN_PDB_FILE} not found. Skipping Run ${i}."
        echo "----------------------------------------------------------------------"
        continue
    fi

    # Check if DCD file exists
    if [ ! -f "${RUN_DCD_FILE}" ]; then
        echo "WARNING: DCD file ${RUN_DCD_FILE} not found. Skipping Run ${i}."
        echo "----------------------------------------------------------------------"
        continue
    fi

    # Construct the command arguments in an array for safety
    COMMAND_ARGS=(
        "python" "${ANALYSIS_SCRIPT}"
        "${RUN_PDB_FILE}"
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
# echo "Multi-DCD analysis batch finished."

#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Script to run protein network analysis for a SINGLE CONCATENATED DCD file.

# --- Configuration (Edit these paths if your concatenated files are named differently) ---
CONCAT_DCD_FILE="Data/AF2_LM211_WT/calcium/processed/af2_lm211_wt_calcium_concat_processed.dcd"
CONCAT_PDB_FILE="Data/AF2_LM211_WT/calcium/processed/af2_lm211_wt_calcium_concat_processed_frame1.pdb"
# Base directory for DCD/PDB to extract condition name (e.g., "calcium")
# This should point to the directory structure above the "processed" folder.
# Example: Data/AF2_LM211_WT/calcium
CONDITION_BASE_PATH=$(dirname $(dirname "${CONCAT_DCD_FILE}")) # Extracts "Data/AF2_LM211_WT/calcium"

START_RESIDUE="101"
END_RESIDUE="809"

COV_TYPE="displacement_mean_dot"
FILTERING_MODE="original_ec"
CONTACT_ATOMS="cbeta"
CONTACT_FREQ="0.3"

# --- Determine Output Directory Dynamically ---
# Extract condition (e.g., sodium, calcium)
CONDITION_NAME=$(basename "${CONDITION_BASE_PATH}") # Extracts "calcium"

# Determine abbreviations for path components based on analysis parameters
case "${COV_TYPE}" in
    "displacement_mean_dot") COV_ABBREV="dmd" ;;
    "pearson") COV_ABBREV="pea" ;;
    *) COV_ABBREV="${COV_TYPE:0:3}" ;;
esac

case "${FILTERING_MODE}" in
    "original_ec") FIL_ABBREV="oec" ;;
    *) FIL_ABBREV="${FILTERING_MODE:0:3}" ;;
esac

case "${CONTACT_ATOMS}" in
    "cbeta") ATOM_ABBREV="cb" ;;
    "calpha") ATOM_ABBREV="ca" ;;
    *) ATOM_ABBREV="${CONTACT_ATOMS:0:2}" ;;
esac

# Construct the parameter-specific part of the output directory name
PARAMETERS_SUBDIR="${START_RESIDUE}-${END_RESIDUE}_${COV_ABBREV}_${FIL_ABBREV}_${ATOM_ABBREV}_cf${CONTACT_FREQ}_concat"

# Define the base output directory, incorporating the condition and dynamic parameters
OUTPUT_DIR="analysis_LM2/AF2_LM211_WT/${CONDITION_NAME}/${PARAMETERS_SUBDIR}"

# Path to the Python analysis script, assuming this shell script is run from the project root
ANALYSIS_SCRIPT="protein_network_analysis_updated.py" # Relative to project root

# --- Script Logic ---

echo "Starting concatenated DCD analysis..."
echo "======================================================================"
echo "Input PDB File: ${CONCAT_PDB_FILE}"
echo "Input DCD File: ${CONCAT_DCD_FILE}"
echo "Residues: ${START_RESIDUE}-${END_RESIDUE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Contact Frequency: ${CONTACT_FREQ}"
echo "Analysis Script: ${ANALYSIS_SCRIPT}"
echo "======================================================================"

# Create the output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "ERROR: Analysis script '${ANALYSIS_SCRIPT}' not found in $(pwd)."
    echo "Make sure this script is run from the project root where '${ANALYSIS_SCRIPT}' is located."
    exit 1
fi

echo ""
echo "--- Processing Concatenated Trajectory --- "
echo "Input PDB File: ${CONCAT_PDB_FILE}"
echo "Input DCD File: ${CONCAT_DCD_FILE}"
echo "Outputting to: ${OUTPUT_DIR}"

# Check if PDB file exists
if [ ! -f "${CONCAT_PDB_FILE}" ]; then
    echo "ERROR: PDB file ${CONCAT_PDB_FILE} not found. Aborting."
    echo "Please ensure the path is correct and the file exists."
    echo "----------------------------------------------------------------------"
    exit 1
fi

# Check if DCD file exists
if [ ! -f "${CONCAT_DCD_FILE}" ]; then
    echo "ERROR: DCD file ${CONCAT_DCD_FILE} not found. Aborting."
    echo "Please ensure the path is correct and the file exists."
    echo "----------------------------------------------------------------------"
    exit 1
fi

# Construct the command arguments in an array for safety
COMMAND_ARGS=(
    "python" "${ANALYSIS_SCRIPT}"
    "${CONCAT_PDB_FILE}"
    "${CONCAT_DCD_FILE}"
    "${START_RESIDUE}"
    "${END_RESIDUE}"
    --output_dir "${OUTPUT_DIR}"
    --cov_type "${COV_TYPE}"
    --filtering_mode "${FILTERING_MODE}"
    --contact_atoms "${CONTACT_ATOMS}"
    --contact_freq "${CONTACT_FREQ}"
)

echo "Executing command for concatenated trajectory:"
echo "${COMMAND_ARGS[@]}" # Print the command array elements

# Execute the command
"${COMMAND_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo "Concatenated trajectory analysis completed successfully."
else
    echo "ERROR: Concatenated trajectory analysis failed. Check logs in ${OUTPUT_DIR}/analysis_report.txt"
    # The 'set -e' at the top will cause script to exit on failure anyway
fi
echo "----------------------------------------------------------------------"

echo ""
echo "Concatenated DCD analysis finished."
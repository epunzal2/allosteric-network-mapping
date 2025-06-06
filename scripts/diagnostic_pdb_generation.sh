#!/bin/bash
#
# Diagnostic PDB Generation Script
#
# Purpose:
#   Tests PDB file generation for individual processed DCDs and a concatenated DCD.
#   This helps verify that cpptraj commands for PDB extraction are working correctly,
#   especially after modifications to stripping or PDB output methods.
#
# How to Run:
#   This script is intended to be run from the root of the 'allosteric-network-mapping'
#   project directory.
#
#   Example:
#   cd /path/to/your/allosteric-network-mapping
#   ./scripts/diagnostic_pdb_generation.sh
#
#   Logs will be generated in the './logs' directory within 'allosteric-network-mapping'.
#
set -e

# --- Configuration ---
BASE_DIR="." # Adjusted for running from within allosteric-network-mapping
SCRIPTS_DIR="${BASE_DIR}/scripts"
DATA_DIR="${BASE_DIR}/Data/AF2_LM211_WT/calcium" # This is where original DCDs and parm are
DIAGNOSTIC_OUT_DIR="${DATA_DIR}/diagnostic_test_pdbs" # New output dir for these test PDBs
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p "${DIAGNOSTIC_OUT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/diagnostic_pdb_gen_${TIMESTAMP}.log"

# Assumed Parameter File (based on previous script contents)
PARM_FILE="${DATA_DIR}/af2_lm211_wt_ca2+.parm7"

# Original DCD files to test
ORIGINAL_DCD_FILES=(
    "${DATA_DIR}/af2_lm211_wt_calcium_1.dcd"
    "${DATA_DIR}/af2_lm211_wt_calcium_2.dcd"
    "${DATA_DIR}/af2_lm211_wt_calcium_3.dcd"
)

# Output name for the PDB from the concatenated trajectory
CONCAT_PDB_OUT="${DIAGNOSTIC_OUT_DIR}/af2_lm211_wt_calcium_1-3_concat_frame1_test.pdb"

# Temporary cpptraj input script
TEMP_CPPTRAJ_SCRIPT="${SCRIPTS_DIR}/temp_diagnostic_cpptraj.in"

# Redirect stdout and stderr to log file and console
exec &> >(tee -a "${LOG_FILE}")

echo "INFO: Diagnostic PDB Generation Script Started: $(date)"
echo "INFO: Full log will be available at: ${LOG_FILE}"
echo "INFO: Using Parameter File: ${PARM_FILE}"
echo "INFO: Outputting diagnostic PDBs to: ${DIAGNOSTIC_OUT_DIR}"

if [ ! -f "${PARM_FILE}" ]; then
    echo "ERROR: Parameter file ${PARM_FILE} not found. Cannot proceed." >&2
    exit 1
fi

echo ""
echo "--- Generating PDB for individual original DCDs ---"

for DCD_FILE_PATH in "${ORIGINAL_DCD_FILES[@]}"; do
    if [ ! -f "${DCD_FILE_PATH}" ]; then
        echo "WARNING: DCD file ${DCD_FILE_PATH} not found. Skipping."
        continue
    fi

    DCD_FILENAME=$(basename "${DCD_FILE_PATH}")
    PDB_OUT_BASENAME="${DCD_FILENAME%.dcd}" # Remove .dcd
    PDB_OUT_FILE="${DIAGNOSTIC_OUT_DIR}/${PDB_OUT_BASENAME}_frame1_test.pdb"
    TEMP_PROCESSED_DCD_INDIVIDUAL="${DIAGNOSTIC_OUT_DIR}/${PDB_OUT_BASENAME}_temp_processed_individual.dcd"

    echo "Processing: ${DCD_FILENAME}"
    echo "  Input Original DCD: ${DCD_FILE_PATH}"
    echo "  Temporary Processed DCD: ${TEMP_PROCESSED_DCD_INDIVIDUAL}"
    echo "  Output Test PDB: ${PDB_OUT_FILE}"

    # Step 1: Process original DCD and save temporary processed DCD
    echo "  Step 1: Processing original DCD to ${TEMP_PROCESSED_DCD_INDIVIDUAL}"
cat > "${TEMP_CPPTRAJ_SCRIPT}" << EOF
parm ${PARM_FILE}
trajin ${DCD_FILE_PATH}
strip :WAT
strip :CA
strip :Cl-
autoimage
trajout ${TEMP_PROCESSED_DCD_INDIVIDUAL} dcd
run
quit
EOF
    echo "    Running cpptraj for Step 1..."
    # cat "${TEMP_CPPTRAJ_SCRIPT}" # Optional: for verbose logging
    cpptraj -i "${TEMP_CPPTRAJ_SCRIPT}"
    STEP1_SUCCESS=$?

    if [ ${STEP1_SUCCESS} -eq 0 ] && [ -f "${TEMP_PROCESSED_DCD_INDIVIDUAL}" ]; then
        echo "    Step 1 SUCCESS: Created ${TEMP_PROCESSED_DCD_INDIVIDUAL}"
        # Step 2: Extract first frame PDB from the temporary processed DCD
        echo "  Step 2: Extracting PDB from ${TEMP_PROCESSED_DCD_INDIVIDUAL} to ${PDB_OUT_FILE}"
cat > "${TEMP_CPPTRAJ_SCRIPT}" << EOF
parm ${PARM_FILE} # Load original parm
parmstrip :WAT    # Strip parm to match the TEMP_PROCESSED_DCD_INDIVIDUAL
parmstrip :CA
parmstrip :Cl-
trajin ${TEMP_PROCESSED_DCD_INDIVIDUAL} 1 1 1
trajout ${PDB_OUT_FILE} pdb
run
quit
EOF
        echo "    Running cpptraj for Step 2..."
        # cat "${TEMP_CPPTRAJ_SCRIPT}" # Optional: for verbose logging
        cpptraj -i "${TEMP_CPPTRAJ_SCRIPT}"
        STEP2_SUCCESS=$?
    else
        echo "    Step 1 FAILED or ${TEMP_PROCESSED_DCD_INDIVIDUAL} not created." >&2
        STEP2_SUCCESS=1 # Mark step 2 as failed if step 1 failed
    fi
    if [ $? -eq 0 ]; then
        echo "  SUCCESS: Generated ${PDB_OUT_FILE}"
        if [ -f "${PDB_OUT_FILE}" ]; then
            echo "    File check: ${PDB_OUT_FILE} exists."
        else
            echo "    ERROR: File check: ${PDB_OUT_FILE} was NOT created." >&2
        fi
    else
        echo "  ERROR: cpptraj failed for ${DCD_FILENAME}" >&2
    fi
    rm -f "${TEMP_PROCESSED_DCD_INDIVIDUAL}" # Clean up temporary DCD
    echo "-------------------------------------"
done

echo ""
echo "--- Generating PDB from Concatenated Original DCDs ---"
echo "  Input Original DCDs for concatenation:"
for dcd_in in "${ORIGINAL_DCD_FILES[@]}"; do echo "    - ${dcd_in}"; done
echo "  Output Test PDB from Concatenated: ${CONCAT_PDB_OUT}"
TEMP_CONCAT_PROCESSED_DCD="${DIAGNOSTIC_OUT_DIR}/af2_lm211_wt_calcium_1-3_temp_concat_processed.dcd"
echo "  Temporary Concatenated Processed DCD: ${TEMP_CONCAT_PROCESSED_DCD}"


# Step 1: Concatenate original DCDs, process, and save temporary concatenated DCD
echo "  Step 1: Processing and concatenating original DCDs to ${TEMP_CONCAT_PROCESSED_DCD}"
cat > "${TEMP_CPPTRAJ_SCRIPT}" << EOF
parm ${PARM_FILE}
EOF

for DCD_FILE_PATH in "${ORIGINAL_DCD_FILES[@]}"; do
    if [ -f "${DCD_FILE_PATH}" ]; then
        echo "trajin ${DCD_FILE_PATH}" >> "${TEMP_CPPTRAJ_SCRIPT}"
    else
        echo "# WARNING: Input Original DCD for concatenation ${DCD_FILE_PATH} not found. Skipped." >> "${TEMP_CPPTRAJ_SCRIPT}"
    fi
done

cat >> "${TEMP_CPPTRAJ_SCRIPT}" << EOF
strip :WAT
strip :CA
strip :Cl-
autoimage
trajout ${TEMP_CONCAT_PROCESSED_DCD} dcd
run
quit
EOF
    echo "    Running cpptraj for Step 1 (concatenation)..."
    # cat "${TEMP_CPPTRAJ_SCRIPT}" # Optional: for verbose logging
    cpptraj -i "${TEMP_CPPTRAJ_SCRIPT}"
    CONCAT_STEP1_SUCCESS=$?

    if [ ${CONCAT_STEP1_SUCCESS} -eq 0 ] && [ -f "${TEMP_CONCAT_PROCESSED_DCD}" ]; then
        echo "    Step 1 SUCCESS: Created ${TEMP_CONCAT_PROCESSED_DCD}"
        # Step 2: Extract first frame PDB from the temporary concatenated processed DCD
        echo "  Step 2: Extracting PDB from ${TEMP_CONCAT_PROCESSED_DCD} to ${CONCAT_PDB_OUT}"
cat > "${TEMP_CPPTRAJ_SCRIPT}" << EOF
parm ${PARM_FILE} # Load original parm
parmstrip :WAT    # Strip parm to match the TEMP_CONCAT_PROCESSED_DCD
parmstrip :CA
parmstrip :Cl-
trajin ${TEMP_CONCAT_PROCESSED_DCD} 1 1 1
trajout ${CONCAT_PDB_OUT} pdb
run
quit
EOF
        echo "    Running cpptraj for Step 2 (PDB extraction)..."
        # cat "${TEMP_CPPTRAJ_SCRIPT}" # Optional: for verbose logging
        cpptraj -i "${TEMP_CPPTRAJ_SCRIPT}"
        CONCAT_STEP2_SUCCESS=$?
    else
        echo "    Step 1 FAILED or ${TEMP_CONCAT_PROCESSED_DCD} not created." >&2
        CONCAT_STEP2_SUCCESS=1 # Mark step 2 as failed
    fi

if [ ${CONCAT_STEP2_SUCCESS} -eq 0 ]; then
    echo "  SUCCESS: PDB extraction from concatenated trajectory process completed."
    if [ -f "${CONCAT_PDB_OUT}" ]; then
        echo "    File check: ${CONCAT_PDB_OUT} exists."
    else
        echo "    ERROR: File check: ${CONCAT_PDB_OUT} was NOT created (after successful cpptraj Step 2)." >&2
    fi
else
    echo "  ERROR: cpptraj processing failed for concatenated trajectory (either Step 1 or Step 2)." >&2
fi
rm -f "${TEMP_CONCAT_PROCESSED_DCD}" # Clean up temporary concatenated DCD
echo "-------------------------------------"

rm -f "${TEMP_CPPTRAJ_SCRIPT}"
echo ""
echo "INFO: Diagnostic PDB Generation Script Finished: $(date)"
echo "INFO: Check log file for details: ${LOG_FILE}"
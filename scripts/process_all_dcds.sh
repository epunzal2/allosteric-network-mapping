#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Activate conda environment
CONDA_BASE_PATH=$(conda info --base)
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate maupy-env

# Define project directory early for logging
BASE_PROJECT_DIR_ABS="/Users/exequielpunzalan/Library/CloudStorage/OneDrive-RutgersUniversity/OARC/amarel_ep523/scratch/kulczyk/code/allosteric-network-mapping"

# --- Logging Setup ---
LOG_DIR_REL="logs" # Relative to project directory
LOG_DIR_ABS="${BASE_PROJECT_DIR_ABS}/${LOG_DIR_REL}"
mkdir -p "${LOG_DIR_ABS}" # Create log directory if it doesn't exist
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR_ABS}/process_all_dcds_${TIMESTAMP}.log"

# Redirect stdout and stderr to log file and console
# IMPORTANT: This exec command affects the rest of the script.
exec &> >(tee -a "${LOG_FILE}")

echo "INFO: Logging stdout and stderr to ${LOG_FILE}"
echo "INFO: Script execution started at $(date)"
# --- End Logging Setup ---

# --- Configuration ---
BASE_DATA_DIR_ABS="${BASE_PROJECT_DIR_ABS}/Data" # Main directory where system-specific subdirectories reside

TEMPLATE_SCRIPT_ABS="${BASE_PROJECT_DIR_ABS}/scripts/cpptraj_processing_template.in"
TEMP_RUN_SCRIPT_ABS="${BASE_PROJECT_DIR_ABS}/current_run.in"

# Define system and condition subdirectories relative to BASE_DATA_DIR_ABS
# These are the paths like "SystemName/ConditionName"
SYSTEM_CONDITION_DIRS=(
    "AF2_LM211_WT/calcium"
    "AF2_LM211_WT/sodium"
    "AF2_LM2_Y138H_11_Mutant/calcium"
    "AF2_LM2_Y138H_11_Mutant/sodium"
)
# --- End Configuration ---

for SYS_COND_REL_PATH in "${SYSTEM_CONDITION_DIRS[@]}"; do
    CURRENT_SYS_DATA_DIR_ABS="${BASE_DATA_DIR_ABS}/${SYS_COND_REL_PATH}"
    CURRENT_PROCESSED_DIR_ABS="${CURRENT_SYS_DATA_DIR_ABS}/processed" # Processed files go into a 'processed' subdir

    echo "========================================================================"
    echo "Checking system/condition: ${SYS_COND_REL_PATH}"
    echo "Data directory: ${CURRENT_SYS_DATA_DIR_ABS}"
    echo "========================================================================"

    if [ ! -d "${CURRENT_SYS_DATA_DIR_ABS}" ]; then
        echo "INFO: Directory ${CURRENT_SYS_DATA_DIR_ABS} does not exist. Skipping."
        continue
    fi

    # Find PARM file (parm7 or prmtop)
    # Using find -maxdepth 1 to avoid looking into 'processed' or other subdirs
    shopt -s nullglob # Important: makes pattern expand to nothing if no match
    PARM_FILES_FOUND=($(find "${CURRENT_SYS_DATA_DIR_ABS}" -maxdepth 1 -type f \( -name "*.parm7" -o -name "*.prmtop" \)))
    shopt -u nullglob # Turn off nullglob

    if [ ${#PARM_FILES_FOUND[@]} -eq 0 ]; then
        echo "INFO: No .parm7 or .prmtop file found in ${CURRENT_SYS_DATA_DIR_ABS}. Skipping this directory."
        continue
    elif [ ${#PARM_FILES_FOUND[@]} -gt 1 ]; then
        echo "WARNING: Multiple .parm7/.prmtop files found in ${CURRENT_SYS_DATA_DIR_ABS}. Using the first one: ${PARM_FILES_FOUND[0]}"
        # To be stricter, you could make this an error:
        # echo "ERROR: Multiple .parm7/.prmtop files found. Please ensure only one exists. Skipping." >&2
        # continue
    fi
    PARM_FILE_ABS="${PARM_FILES_FOUND[0]}"
    echo "INFO: Using PARM file: ${PARM_FILE_ABS}"

    # Find DCD files (only in the current system data directory, not subdirectories)
    shopt -s nullglob
    DCD_FILES_FOUND=($(find "${CURRENT_SYS_DATA_DIR_ABS}" -maxdepth 1 -type f -name "*.dcd"))
    shopt -u nullglob

    if [ ${#DCD_FILES_FOUND[@]} -eq 0 ]; then
        echo "INFO: No .dcd files found in ${CURRENT_SYS_DATA_DIR_ABS}. Skipping this directory."
        continue
    fi

    # Create processed directory if it doesn't exist
    mkdir -p "${CURRENT_PROCESSED_DIR_ABS}"

    for INPUT_DCD_ABS in "${DCD_FILES_FOUND[@]}"; do
        DCD_FILENAME=$(basename "${INPUT_DCD_ABS}")
        DCD_BASENAME_NO_EXT="${DCD_FILENAME%.dcd}" # Remove .dcd extension

        # Define output file names
        OUTPUT_DCD_ABS="${CURRENT_PROCESSED_DIR_ABS}/${DCD_BASENAME_NO_EXT}_processed.dcd"
        OUTPUT_PDB_ABS="${CURRENT_PROCESSED_DIR_ABS}/${DCD_BASENAME_NO_EXT}_processed_frame1.pdb"
        # Place RMSD file in the 'processed' directory for the current system/condition
        OUTPUT_RMSD_ABS="${CURRENT_PROCESSED_DIR_ABS}/${DCD_BASENAME_NO_EXT}_rmsd.dat"

        echo "-----------------------------------------------------"
        echo "Processing DCD: ${INPUT_DCD_ABS}"
        echo "  Output DCD: ${OUTPUT_DCD_ABS}"
        echo "  Output PDB: ${OUTPUT_PDB_ABS}"
        echo "  Output RMSD: ${OUTPUT_RMSD_ABS}"
        echo "-----------------------------------------------------"

        # Create the specific cpptraj input script for this run
        # Using | as sed delimiter to avoid issues with paths containing /
        sed -e "s|__PARM_FILE__|${PARM_FILE_ABS}|g" \
            -e "s|__INPUT_DCD__|${INPUT_DCD_ABS}|g" \
            -e "s|__OUTPUT_DCD__|${OUTPUT_DCD_ABS}|g" \
            -e "s|__OUTPUT_PDB__|${OUTPUT_PDB_ABS}|g" \
            -e "s|__OUTPUT_RMSD__|${OUTPUT_RMSD_ABS}|g" \
            "${TEMPLATE_SCRIPT_ABS}" > "${TEMP_RUN_SCRIPT_ABS}"

        cpptraj -i "${TEMP_RUN_SCRIPT_ABS}"

        if [ $? -eq 0 ]; then
            echo "INFO: cpptraj processing completed for ${DCD_FILENAME}."
            echo "INFO: Validating processed DCD: ${OUTPUT_DCD_ABS} with PDB: ${OUTPUT_PDB_ABS}"
            
            # Call the validation script
            python "${BASE_PROJECT_DIR_ABS}/scripts/validate_dcd_mda.py" \
                -t "${OUTPUT_PDB_ABS}" \
                -d "${OUTPUT_DCD_ABS}"
            
            VALIDATION_EXIT_CODE=$?
            if [ ${VALIDATION_EXIT_CODE} -ne 0 ]; then
                echo "ERROR: MDAnalysis validation failed for ${OUTPUT_DCD_ABS} (Exit Code: ${VALIDATION_EXIT_CODE}). Please check the output above." >&2
                echo "       Consider checking the temporary cpptraj script: ${TEMP_RUN_SCRIPT_ABS}"
                # Optional: Exit if validation fails
                # echo "Exiting due to validation failure." >&2
                # conda deactivate
                # exit 1 
            else
                echo "INFO: MDAnalysis validation successful for ${OUTPUT_DCD_ABS}."
            fi
            echo "DCD ${DCD_FILENAME} processed and validated for ${SYS_COND_REL_PATH}."
        else
            echo "ERROR: cpptraj failed for ${DCD_FILENAME} in ${SYS_COND_REL_PATH} (Exit Code: $?). Check ${TEMP_RUN_SCRIPT_ABS} and cpptraj output." >&2
            conda deactivate
            exit 1 # Exit script on cpptraj error
        fi
    done
done

# Clean up temporary run script if it exists
rm -f "${TEMP_RUN_SCRIPT_ABS}"

conda deactivate

echo "========================================================================"
echo "All specified systems/conditions checked and DCDs processed where found."
echo "========================================================================"

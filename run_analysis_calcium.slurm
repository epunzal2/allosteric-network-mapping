#!/bin/bash
#SBATCH --job-name=Ca_NetAnalysis
#SBATCH --output=logs/Ca_NetAnalysis_%A_%a.out  # Array job output, %A for job ID, %a for task ID
#SBATCH --error=logs/Ca_NetAnalysis_%A_%a.err   # Array job error
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-1 # Two systems: WT and Mutant

# Load modules and activate conda environment
module purge
module use /projects/community/modulefiles
source /projects/community/anaconda/2024.06/ts840/etc/profile.d/conda.sh

ENVNAME=maupy-env
conda activate $ENVNAME

# print some env info (for debugging)
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "PYTHONPATH: $(echo ${PYTHONPATH})"
echo "python -m site: $(python -m site)"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Define system configurations
# Each element: "SYSTEM_NAME INPUT_BASE_DIR SYSTEM_TAG"
SYSTEMS=(
    "AF2_LM211_WT Data/AF2_LM211_WT/calcium WT"
    "AF2_LM2_Y138H_11_Mutant Data/AF2_LM2_Y138H_11_Mutant/calcium Mutant_Y138H"
)

# Get current system based on SLURM_ARRAY_TASK_ID
CURRENT_SYSTEM_CONFIG=(${SYSTEMS[$SLURM_ARRAY_TASK_ID]})
SYSTEM_NAME=${CURRENT_SYSTEM_CONFIG[0]}
INPUT_BASE_DIR_RELATIVE=${CURRENT_SYSTEM_CONFIG[1]}
SYSTEM_TAG=${CURRENT_SYSTEM_CONFIG[2]}

INPUT_DIR="$SLURM_SUBMIT_DIR/$INPUT_BASE_DIR_RELATIVE"
PDB_FILE="$INPUT_DIR/frame1.pdb"
DCD_FILE="$INPUT_DIR/concatenated.dcd"

echo "Processing System: $SYSTEM_NAME"
echo "Input Directory: $INPUT_DIR"
echo "PDB File: $PDB_FILE"
echo "DCD File: $DCD_FILE"
echo "System Tag: $SYSTEM_TAG"

# Define residue pairs (resid1,resid2)
# These are common for both systems as per the CSV files
# Modified for testing with a single pair
RESIDUE_PAIRS=(
    "101,225"
)

# Python script location (assuming it's in the same directory as the SLURM script)
PYTHON_SCRIPT="$SLURM_SUBMIT_DIR/protein_network_analysis_updated.py"

# Common analysis parameters
COV_TYPE="displacement_mean_dot"
CONTACT_CUTOFF="0.75"
CONTACT_FREQ="0.3"
CONTACT_ATOMS="cbeta"
FILTERING_MODE="original_ec"
TOP_N_CRITICAL="10"

# Create logs directory if it doesn't exist
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Loop through each residue pair
for PAIR in "${RESIDUE_PAIRS[@]}"; do
    IFS=',' read -r START_RESIDUE END_RESIDUE <<< "$PAIR"

    echo "----------------------------------------------------"
    echo "Running analysis for: $SYSTEM_NAME, Pair: $START_RESIDUE-$END_RESIDUE"
    echo "----------------------------------------------------"

    # Define output directory for this specific run
    # Example: output_results/AF2_LM211_WT/calcium/101_225_analysis
    OUTPUT_SUBDIR="${INPUT_BASE_DIR_RELATIVE}/${START_RESIDUE}_${END_RESIDUE}_${COV_TYPE}_${FILTERING_MODE}_${CONTACT_ATOMS}_cf${CONTACT_FREQ}"
    OUTPUT_DIR_FULL="$SLURM_SUBMIT_DIR/analysis_results/${OUTPUT_SUBDIR}"
    mkdir -p "$OUTPUT_DIR_FULL"

    # Define output image name
    # Example: protein_network_WT_S101_E225.png
    OUT_IMAGE_BASENAME="protein_network_${SYSTEM_TAG}_S${START_RESIDUE}_E${END_RESIDUE}.png"

    echo "Output Directory: $OUTPUT_DIR_FULL"
    echo "Output Image: $OUT_IMAGE_BASENAME"

    # Check if input files exist
    if [ ! -f "$PDB_FILE" ]; then
        echo "Error: PDB file not found: $PDB_FILE"
        continue # Skip to next pair
    fi
    if [ ! -f "$DCD_FILE" ]; then
        echo "Error: DCD file not found: $DCD_FILE"
        continue # Skip to next pair
    fi

    # Run the protein network analysis script
    python "$PYTHON_SCRIPT" \
        "$PDB_FILE" \
        "$DCD_FILE" \
        "$START_RESIDUE" \
        "$END_RESIDUE" \
        --output_dir "$OUTPUT_DIR_FULL" \
        --cov_type "$COV_TYPE" \
        --contact_cutoff "$CONTACT_CUTOFF" \
        --contact_freq "$CONTACT_FREQ" \
        --contact_atoms "$CONTACT_ATOMS" \
        --filtering_mode "$FILTERING_MODE" \
        --top_n_critical "$TOP_N_CRITICAL" \
        --out_image "$OUT_IMAGE_BASENAME"

    if [ $? -eq 0 ]; then
        echo "Analysis completed successfully for $SYSTEM_NAME, Pair $START_RESIDUE-$END_RESIDUE."
    else
        echo "Error during analysis for $SYSTEM_NAME, Pair $START_RESIDUE-$END_RESIDUE. Check error log."
    fi
    echo "----------------------------------------------------"
done

echo "All analyses for $SYSTEM_NAME completed."
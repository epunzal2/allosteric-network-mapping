#!/bin/bash
#SBATCH --job-name=protein_network
#SBATCH --output=protein_network_%j.out
#SBATCH --error=protein_network_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=main

# Load necessary modules (adjust these based on your cluster's available modules)
module purge
module load python/3.8
module load anaconda3

# Activate your conda environment if needed (uncomment and adjust if necessary)
# source activate your_env_name

# Set directories
INPUT_DIR="$SLURM_SUBMIT_DIR/trajectory_analysis_files"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/output_results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Define the residues for the path (adjust as needed for your protein)
START_RESIDUE=25  # Example value - adjust based on your protein
END_RESIDUE=109   # Example value - adjust based on your protein

# Run the protein network analysis script
python protein_network_analysis_updated.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --cov_type displacement_mean_dot \
    --contact_cutoff 0.75 \
    --contact_freq 0.5 \
    --contact_atoms calpha \
    --filtering_mode fragmentation_pruning \
    --top_n_critical 10 \
    --out_image "protein_network_${START_RESIDUE}_to_${END_RESIDUE}.png" \
    mdm2.pdb \
    mdm2.dcd \
    $START_RESIDUE \
    $END_RESIDUE

echo "Job completed"

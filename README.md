# allosteric-network-mapping: Protein Network Analysis via Molecular Dynamics and Graph Theory

---

## Overview

**allosteric-network-mapping** is a Python toolkit for analyzing allosteric communication pathways and critical residues in proteins using molecular dynamics (MD) simulations. It implements a graph-theoretical approach inspired by recent scientific literature, enabling:

- Identification of **optimal communication paths** between residues.
- Detection of **critical bottleneck residues** via network centrality.
- Flexible covariance calculation methods capturing residue dynamics.
- Multiple graph pruning strategies, including **percolation-based criticality**.
- Visualization of the protein network with highlighted paths and key residues.

---

## Scientific Background

This tool is based on methodologies described in:

- **Proctor et al. (2011)** and related works on **optimal path mapping** in protein allosteric networks.
- Covariance-based coupling matrices derived from MD simulations [(see Equation 6 in referenced paper)].
- Graph pruning via contact frequency and correlation strength to emphasize **dominant communication pathways**.
- Identification of **critical residues** whose removal disrupts network connectivity, relevant for drug design and protein engineering.

The approach involves:

1. **MD Simulation**: Generate trajectories capturing protein dynamics.
2. **Covariance Analysis**: Quantify correlated motions between residues.
3. **Contact Filtering**: Retain physically relevant residue pairs.
4. **Graph Construction**: Nodes = residues; edges = contacts weighted by correlation.
5. **Pruning**: Remove weak/non-physical edges to reveal key pathways.
6. **Pathfinding**: Use Dijkstra's algorithm to find optimal paths.
7. **Criticality Analysis**: Identify bottleneck residues via betweenness centrality.

---

## Features

- **Multiple Covariance Modes**:
  - Raw coordinate covariance.
  - Displacement covariance (mean of dot products).
  - Displacement covariance (dot product of mean deviations, per paper).
- **Flexible Contact Atom Selection**:
  - C-beta atoms (default), with Glycine fallback to C-alpha.
  - C-alpha atoms only.
- **Graph Filtering Options**:
  - Contact frequency only.
  - Covariance magnitude cutoff (original method).
  - Percolation-based fragmentation pruning (paper method).
- **Automated Critical Cutoff Calculation**:
  - Dynamically determines pruning thresholds to achieve ~50% graph fragmentation.
- **Visualization**:
  - Network graph with highlighted optimal path and critical residues.
- **Command-line Interface** with rich options.
- **Well-documented, modular code** for customization.

---

## Installation

1. **Clone or download** this repository.

2. **Install dependencies** (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

Dependencies include:

- `mdtraj`
- `numpy`
- `networkx`
- `matplotlib`
- `tqdm`
- `scikit-learn`
- `pandas`
- `scipy`

---

## Input Preparation

- **MD Trajectory**: Requires a **PDB** file (topology) and a **DCD** file (trajectory).
- Place files in the working directory or provide full paths.
- Example files:
  - `trajectory_analysis_files/mdm2.pdb`
  - `trajectory_analysis_files/mdm2.dcd`

Example test files were obtained from the [ProDy trajectory analysis tutorial](http://www.bahargroup.org/prody/tutorials/trajectory_analysis/trajectory.html).

---

## Usage

Run the main script:

```bash
python protein_network_analysis_updated.py [PDB_FILE] [DCD_FILE] [START_RESID] [END_RESID] [OPTIONS]
```

**Positional arguments:**

- `PDB_FILE`: Path to topology file (e.g., `mdm2.pdb`)
- `DCD_FILE`: Path to trajectory file (e.g., `mdm2.dcd`)
- `START_RESID`: Starting residue sequence number (PDB numbering)
- `END_RESID`: Ending residue sequence number (PDB numbering)

**Key options:**

| Option | Description | Default |
|---------|-------------|---------|
| `--cov_type` | Covariance method: `coordinate`, `displacement_mean_dot`, `displacement_dot_mean` | `displacement_mean_dot` |
| `--contact_atoms` | Atoms for contact calc: `cbeta` or `calpha` | `calpha` |
| `--contact_cutoff` | Contact distance cutoff in nm | `0.75` (7.5 Å) |
| `--contact_freq` | Min contact frequency (0-1) | `0.5` |
| `--filtering_mode` | Graph filtering: `contact_only`, `original_ec`, `fragmentation_pruning` | `original_ec` |
| `-n` | Number of top critical residues to report | `10` |
| `--out_image` | Output filename for network visualization | `protein_network.png` |

---

## Covariance Calculation Methods

- **coordinate**: Standard covariance of C-alpha Cartesian coordinates.
- **displacement_mean_dot**: Mean of dot products of unit displacement deviations (captures dynamic directional correlations).
- **displacement_dot_mean**: Dot product of mean deviations of unit displacement vectors (matches Eq.6 in the paper).

Use `--cov_type` to select.

---

## Graph Construction & Pruning Modes

- **contact_only**: Only contact frequency filter applied; no pruning by correlation.
- **original_ec**:
  - Calculates a critical covariance magnitude cutoff (`E_c`) so that ~50% of possible edges remain.
  - Edges with |covariance| < `E_c` are excluded during graph construction.
- **original_ec** (default):
  - Calculates a critical covariance magnitude cutoff (`E_c`) so that ~50% of possible edges remain.
  - Edges with |covariance| < `E_c` are excluded during graph construction.

- **fragmentation_pruning**:
  - Builds graph with contact + correlation weights.
  - Calculates a critical weight cutoff (`E_c`) such that removing edges with weight < `E_c` fragments ~50% of edges into disconnected subgraphs.
  - Closely follows the percolation-based pruning described in the paper.

Select via `--filtering_mode`.

---

## Output Interpretation

- **Console Output**:
  - Summary of parameters, pruning thresholds, and timings.
  - The **optimal path** as a list of residue sequence numbers.
  - The **top N critical residues** ranked by betweenness centrality.

- **Visualization** (`--out_image`):
  - Nodes = residues.
  - **Red**: residues on the optimal path.
  - **Orange**: critical residues.
  - **Purple**: residues that are both.
  - **Grey edges**: network connections.
  - **Red edges**: optimal path.

---

## Example Commands

Run with default settings (original_ec filtering, displacement_mean_dot covariance):

```bash
python protein_network_analysis_updated.py trajectory_analysis_files/mdm2.pdb trajectory_analysis_files/mdm2.dcd 25 109
```

Use displacement covariance (mean of dot products), original Ec pruning, C-beta contacts:

```bash
python protein_network_analysis_updated.py trajectory_analysis_files/mdm2.pdb trajectory_analysis_files/mdm2.dcd 25 109 --cov_type=displacement_mean_dot --filtering_mode=original_ec --contact_atoms=cbeta
```

Disable pruning (contact filter only):

```bash
python protein_network_analysis_updated.py trajectory_analysis_files/mdm2.pdb trajectory_analysis_files/mdm2.dcd 25 109 --filtering_mode=contact_only
```

---

## References

- Proctor EA, Ding F, Dokholyan NV. *Discrete molecular dynamics*. Wiley Interdiscip Rev Comput Mol Sci. 2011.
- Sethi A, Eargle J, Black AA, Luthey-Schulten Z. *Dynamical networks in tRNA:protein complexes*. Proc Natl Acad Sci USA. 2009.
- Chem. Rev. 2016, 116, 6463−6487.

---

## Development Notes & Changelog

- The script is modular and can be extended for other network metrics or visualization styles.

---

## License

This project is provided **as-is** for academic and research purposes. No warranty is implied.

# protein_network_analysis_updated.py

import argparse
import mdtraj as md
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm # Optional: for progress bars
import warnings
import time # To time functions
import itertools # Added for combinations

# Suppress specific warnings from mdtraj if they occur often
warnings.filterwarnings("ignore", message="topologies are inconsistent")
warnings.filterwarnings("ignore", message="WARNING: two consecutive structures were identical")

# --- File Loading & Atom Selection ---

def load_trajectory(pdb_file, dcd_file):
    """Loads trajectory data from PDB and DCD files."""
    print(f"Loading trajectory from PDB: {pdb_file}, DCD: {dcd_file}")
    try:
        traj = md.load(dcd_file, top=pdb_file)
        if traj.n_frames == 0:
             raise ValueError("Trajectory contains 0 frames.")
        print(f"Trajectory loaded: {traj.n_frames} frames, {traj.n_atoms} atoms, {traj.n_residues} residues.")
        # Ensure C-alpha atoms exist
        ca_indices = traj.topology.select('protein and name CA')
        if len(ca_indices) == 0:
             raise ValueError("No C-alpha atoms found in the protein selection. Check PDB file.")
        # Basic check for coordinates
        if np.isnan(traj.xyz).any():
            print("Warning: NaN values found in trajectory coordinates. Check DCD file.")
        return traj
    except Exception as e:
        print(f"Error loading trajectory files: {e}")
        exit(1)

def select_atoms_for_contact(traj, atom_type='calpha', use_ca_for_gly=True):
    """
    Selects atoms for distance calculation based on the specified type.
    - 'cbeta': Selects C-beta atoms, using C-alpha for Glycine (and as fallback).
    - 'calpha': Selects C-alpha atoms for all residues.

    Args:
        traj (md.Trajectory): The trajectory object.
        atom_type (str): The type of atom to select ('cbeta' or 'calpha'). Default 'cbeta'.
        use_ca_for_gly (bool): If atom_type is 'cbeta', whether to use CA for Glycine. Default True.

    Returns:
        tuple:
            - np.ndarray: Array of selected atom indices.
            - dict: Map from original residue index to the selected atom's index.
    """
    print(f"Selecting atoms for contact calculation (type: {atom_type}).")
    topology = traj.topology
    selected_atom_indices_list = []
    ca_indices_map = {res.index: -1 for res in topology.residues}  # Map residue index to C-alpha index
    selected_atom_map = {res.index: -1 for res in topology.residues}  # Map residue index to the selected atom index

    # First pass: find all CAs
    for atom in topology.atoms:
        if atom.name == 'CA':
            ca_indices_map[atom.residue.index] = atom.index

    # Second pass: find CB or use CA for GLY
    residues_with_valid_atom = []
    for res in topology.residues:
        # Skip residues not part of a standard protein chain (if needed, though usually handled by topology)
        # if not res.is_protein: continue

        ca_index = ca_indices_map.get(res.index, -1)
        if ca_index == -1:
            # print(f"Warning: Residue {res.resSeq}-{res.name} has no C-alpha atom. Skipping.")
            continue # Skip residue if no CA

        selected_atom_idx = -1
        if atom_type == 'calpha':
            selected_atom_idx = ca_index
        elif atom_type == 'cbeta':
            cb_atom = next((atom for atom in res.atoms if atom.name == 'CB'), None)
            if cb_atom:
                selected_atom_idx = cb_atom.index
            elif res.name == 'GLY' and use_ca_for_gly:
                selected_atom_idx = ca_index
            else:
                # Default to CA if CB missing and not GLY or if use_ca_for_gly=False
                selected_atom_idx = ca_index
                # print(f"Warning: Residue {res.resSeq}-{res.name} missing CB. Using CA for cbeta selection.")
        else:
            raise ValueError(f"Unknown contact atom type: {atom_type}")

        if selected_atom_idx != -1:
            selected_atom_indices_list.append(selected_atom_idx)
            selected_atom_map[res.index] = selected_atom_idx
            residues_with_valid_atom.append(res.index)

    valid_atom_indices = np.array(selected_atom_indices_list)
    if not valid_atom_indices.size:
        raise ValueError(f"Could not find any valid {atom_type} atoms for contact analysis.")

    print(f"Selected {len(valid_atom_indices)} {atom_type} atoms corresponding to {len(residues_with_valid_atom)} residues for contact calculations.")
    # Return map only for residues that were successfully included
    filtered_selected_atom_map = {res_idx: atom_idx for res_idx, atom_idx in selected_atom_map.items() if res_idx in residues_with_valid_atom}
    return valid_atom_indices, filtered_selected_atom_map

# --- Covariance Calculation Functions ---

def calculate_covariance_coordinate(traj, ca_indices):
    """Calculates the raw covariance matrix of C-alpha atom coordinates."""
    n_ca = len(ca_indices)
    if n_ca == 0:
        raise ValueError("No C-alpha indices provided for coordinate covariance.")
    if traj.n_frames < 2:
        raise ValueError("Trajectory must have at least 2 frames for covariance calculation.")
    if max(ca_indices) >= traj.n_atoms:
         raise ValueError(f"Invalid C-alpha index {max(ca_indices)} found for trajectory with {traj.n_atoms} atoms.")

    print(f"Calculating coordinate covariance for {n_ca} C-alpha atoms...")
    start_time = time.time()
    # Reshape coordinates: (n_frames, n_atoms * 3) -> (n_frames, n_ca * 3)
    ca_xyz = traj.xyz[:, ca_indices, :].reshape(traj.n_frames, -1) # Shape: (n_frames, n_ca * 3)

    # Calculate covariance matrix. Use rowvar=False because variables (coordinates) are columns.
    # np.cov expects variables as rows, so transpose ca_xyz
    cov_matrix_flat = np.cov(ca_xyz.T) # Shape: (n_ca*3, n_ca*3)

    # We need residue-based covariance. Average the covariance over the X, Y, Z components.
    cov_matrix_res = np.zeros((n_ca, n_ca))
    for i in range(n_ca):
        for j in range(i, n_ca): # Calculate only upper triangle
            # Extract the 3x3 submatrix for residues i and j
            sub_matrix = cov_matrix_flat[i*3:(i+1)*3, j*3:(j+1)*3]
            # Use the trace (sum of diagonal elements) as a measure of coupled fluctuation
            cov_matrix_res[i, j] = np.trace(sub_matrix) / 3.0 # Average variance/covariance
            cov_matrix_res[j, i] = cov_matrix_res[i, j] # Symmetric matrix

    elapsed = time.time() - start_time
    print(f"Coordinate covariance calculation finished ({elapsed:.2f}s).")
    return cov_matrix_res

def calculate_covariance_displacement_mean_of_dot(traj, ca_indices):
    """
    Compute raw covariance matrix based on C-alpha unit displacement vectors
    relative to the mean position. (Method 1: Mean of Dot Products)

    Calculates Cov(i, j) = <(e_i - <e_i>) ⋅ (e_j - <e_j>)>, where e_i is the unit vector
    of the displacement of atom i from its mean position (<pos_i>) at a given frame,
    and <...> denotes averaging over frames.

    Args:
        traj (mdtraj.Trajectory): Trajectory object (n_frames, n_atoms, 3).
        ca_indices (np.ndarray): Indices of C-alpha atoms to use.

    Returns:
        numpy.ndarray: Raw covariance matrix (n_ca x n_ca).
                       Needs normalization before use as correlation.
    """
    n_ca = len(ca_indices)
    if n_ca == 0:
        raise ValueError("No C-alpha indices provided.")
    if traj.n_frames < 2:
        raise ValueError("Trajectory must have at least 2 frames for covariance calculation.")

    print(f"Calculating displacement covariance (Mean-of-Dot) for {n_ca} C-alpha atoms...")
    start_time = time.time()
    ca_xyz = traj.xyz[:, ca_indices, :]  # (n_frames, n_ca, 3)
    mean_positions = np.mean(ca_xyz, axis=0)  # (n_ca, 3)
    displacements = ca_xyz - mean_positions  # (n_frames, n_ca, 3)
    displacement_norms = np.linalg.norm(displacements, axis=2, keepdims=True) + 1e-9 # (n_frames, n_ca, 1)
    unit_vectors = displacements / displacement_norms  # (n_frames, n_ca, 3) e_i(t)
    mean_unit_vectors = np.mean(unit_vectors, axis=0)  # (n_ca, 3) <e_i>
    deviations = unit_vectors - mean_unit_vectors  # (n_frames, n_ca, 3) e_i(t) - <e_i>

    # Calculate covariance matrix using einsum for vectorization
    # Cov(i, j) = mean over frames [ dot_product(deviation_i(frame), deviation_j(frame)) ]
    dot_products_over_frames = np.einsum('fad,fbd->fab', deviations, deviations, optimize=True) # (n_frames, n_ca, n_ca)
    cov_matrix = np.mean(dot_products_over_frames, axis=0) # (n_ca, n_ca)
    cov_matrix = (cov_matrix + cov_matrix.T) / 2.0 # Ensure symmetry

    elapsed = time.time() - start_time
    print(f"Displacement covariance (Mean-of-Dot) calculation finished ({elapsed:.2f}s).")
    return cov_matrix

def calculate_covariance_displacement_dot_of_mean(traj, ca_indices):
    """
    Compute raw covariance matrix based on C-alpha unit displacement vectors
    relative to the mean position, following Eq. 6 of the provided paper text.
    (Method 2: Dot Product of Mean Deviations)

    Eq. 6: Cov(i, j) = <dev_i> ⋅ <dev_j>
    where dev_i = e_i - <e_i>, and e_i is the unit vector of displacement
    from the mean position. <...> denotes averaging over frames.

    Args:
        traj (mdtraj.Trajectory): Trajectory object (n_frames, n_atoms, 3).
        ca_indices (np.ndarray): Indices of C-alpha atoms to use.

    Returns:
        numpy.ndarray: Raw covariance matrix (n_ca x n_ca) based on Eq. 6.
                       Needs normalization before use as correlation.
    """
    n_ca = len(ca_indices)
    if n_ca == 0:
        raise ValueError("No C-alpha indices provided.")
    if traj.n_frames < 2:
        raise ValueError("Trajectory must have at least 2 frames for covariance calculation.")

    print(f"Calculating displacement covariance (Dot-of-Mean / Eq.6) for {n_ca} C-alpha atoms...")
    start_time = time.time()
    ca_xyz = traj.xyz[:, ca_indices, :]  # Shape: (n_frames, n_ca, 3)
    mean_positions = np.mean(ca_xyz, axis=0)  # Shape: (n_ca, 3)
    displacements = ca_xyz - mean_positions  # Shape: (n_frames, n_ca, 3)
    displacement_norms = np.linalg.norm(displacements, axis=2, keepdims=True) + 1e-9
    unit_vectors = displacements / displacement_norms  # e_i(t), Shape: (n_frames, n_ca, 3)

    mean_unit_vectors = np.mean(unit_vectors, axis=0)  # <e_i>, Shape: (n_ca, 3)
    deviations = unit_vectors - mean_unit_vectors  # Shape: (n_frames, n_ca, 3) dev_i(t)
    mean_deviations = np.mean(deviations, axis=0) # Shape: (n_ca, 3) <dev_i>

    # Calculate covariance matrix: Cov(i, j) = <dev_i> ⋅ <dev_j>
    cov_matrix = np.einsum('ad,bd->ab', mean_deviations, mean_deviations, optimize=True) # Shape: (n_ca, n_ca)
    # cov_matrix = mean_deviations @ mean_deviations.T # Alternative formulation

    elapsed = time.time() - start_time
    print(f"Displacement covariance (Dot-of-Mean / Eq.6) calculation finished ({elapsed:.2f}s).")
    return cov_matrix

def normalize_covariance(cov_matrix):
    """Normalizes the raw covariance matrix to a correlation matrix."""
    print("Normalizing covariance matrix...")
    start_time = time.time()
    diag_cov = np.diag(cov_matrix)
    # Handle potential zero variance on diagonal (e.g., static atoms)
    # Add small epsilon to avoid division by zero and sqrt of negative (though latter shouldn't happen with cov)
    diag_sqrt = np.sqrt(np.maximum(diag_cov, 1e-12)) # Ensure non-negative before sqrt
    outer_prod_sqrt = np.outer(diag_sqrt, diag_sqrt)

    # Perform division only where denominator is non-zero
    norm_cov_matrix = np.divide(cov_matrix, outer_prod_sqrt,
                                out=np.zeros_like(cov_matrix),
                                where=(outer_prod_sqrt != 0))

    # Clip values to [-1, 1] to handle potential floating point inaccuracies
    norm_cov_matrix = np.clip(norm_cov_matrix, -1.0, 1.0)
    elapsed = time.time() - start_time
    print(f"Normalization finished ({elapsed:.2f}s).")
    return norm_cov_matrix

# --- Contact Frequency ---

def calculate_contact_frequency(traj, atom_indices, distance_cutoff=0.75):
    """Calculates contact frequency between selected atoms."""
    n_residues_analyzed = len(atom_indices) # Note: this is number of *atoms* selected
    n_frames = traj.n_frames
    contact_map = np.zeros((n_residues_analyzed, n_residues_analyzed), dtype=np.int32)

    # Ensure atom_indices are valid within the trajectory
    if max(atom_indices) >= traj.n_atoms:
        raise ValueError("Invalid atom index found in selection for contact calculation.")

    print(f"Calculating pairwise distances for {n_residues_analyzed} selected atoms across {n_frames} frames...")
    start_time = time.time()
    # Generate pairs of indices *within the selection* to compute distances for
    pairs = []
    for i in range(n_residues_analyzed):
        for j in range(i + 1, n_residues_analyzed):
             pairs.append((atom_indices[i], atom_indices[j]))

    if not pairs:
       print("Warning: No pairs found for distance calculation (only 1 atom selected?).")
       return np.zeros((n_residues_analyzed, n_residues_analyzed))

    # Calculate distances frame by frame or in chunks to manage memory if needed
    # For typical sizes, mdtraj handles this well
    distances = md.compute_distances(traj, pairs, periodic=True, opt=True) # Shape (n_frames, n_pairs)
    # Use periodic=True if simulation used PBC, False otherwise. Adjust as needed.

    # Count contacts (distance < cutoff)
    contacts = distances < distance_cutoff # Boolean array (n_frames, n_pairs)
    pair_contact_counts = np.sum(contacts, axis=0) # Shape (n_pairs,)

    # Map pair counts back to the residue contact matrix
    pair_idx = 0
    for i in range(n_residues_analyzed):
        for j in range(i + 1, n_residues_analyzed):
            if pair_idx < len(pair_contact_counts): # Ensure we don't exceed bounds
                count = pair_contact_counts[pair_idx]
                contact_map[i, j] = count
                contact_map[j, i] = count
            pair_idx += 1

    contact_frequency = contact_map / n_frames
    elapsed = time.time() - start_time
    print(f"Contact frequency calculation finished ({elapsed:.2f}s).")
    return contact_frequency

# --- Critical Cutoff Calculation (Original Method) ---

def find_original_critical_ec(
    n_analysis_nodes: int,
    raw_covariance_matrix: np.ndarray,
    contact_frequency: np.ndarray,
    contact_freq_cutoff: float
) -> float:
    """
    Find the critical covariance magnitude cutoff value (E_c) targeting ~50% graph connectivity.

    Args:
        n_analysis_nodes (int): Number of nodes (residues) in the analysis.
        raw_covariance_matrix (np.ndarray): Raw covariance matrix (n_nodes x n_nodes).
        contact_frequency (np.ndarray): Contact frequency matrix (n_nodes x n_nodes).
        contact_freq_cutoff (float): Minimum contact frequency to consider a pair.

    Returns:
        float: The critical E_c value (covariance magnitude threshold).
    """
    print("Calculating critical covariance cutoff (original method)...")
    start_time = time.time()

    # 1. Collect valid absolute covariance values for pairs meeting contact criteria
    valid_cov_values = []
    potential_edges = 0
    for i in range(n_analysis_nodes):
        for j in range(i + 1, n_analysis_nodes):
            if contact_frequency[i, j] >= contact_freq_cutoff:
                potential_edges += 1
                abs_cov_value = abs(raw_covariance_matrix[i, j])
                valid_cov_values.append((abs_cov_value, i, j)) # Store analysis indices

    print(f"Collected {len(valid_cov_values)} valid covariance values from {potential_edges} potential edges.")

    # 2. Sort unique covariance values to test as thresholds
    if not valid_cov_values:
        print("Warning: No valid covariance values found. Using default E_c = 0.1")
        return 0.1

    # Sort by covariance value (ascending) - we test thresholds from low to high
    valid_cov_values.sort()
    unique_cov_values = sorted(list(set(cov for cov, _, _ in valid_cov_values)))

    # Limit thresholds to test for large networks (e.g., ~100 steps)
    if len(unique_cov_values) > 1000: # Increased limit slightly
        step = max(1, len(unique_cov_values) // 1000)
        test_thresholds = unique_cov_values[::step]
        print(f"Testing {len(test_thresholds)} representative covariance thresholds...")
    else:
        test_thresholds = unique_cov_values
        print(f"Testing {len(test_thresholds)} unique covariance thresholds...")


    # 3. Iterate through thresholds and find the one closest to 50% connectivity
    best_ec = unique_cov_values[0]  # Default to minimum value
    min_target_distance = float('inf') # Target is 0.5 connectivity ratio

    max_possible_edges = n_analysis_nodes * (n_analysis_nodes - 1) // 2
    if max_possible_edges == 0: return 0.0 # Handle single node case

    # Build a base graph with only nodes
    G_base = nx.Graph()
    G_base.add_nodes_from(range(n_analysis_nodes))

    for threshold in tqdm(test_thresholds, desc="Testing Original Ec", disable=len(test_thresholds)<100):
        # Build graph with current threshold
        G_test = G_base.copy()
        edges_added = 0
        # Add edges with |covariance| >= threshold (and contact freq already met)
        for cov, node1, node2 in valid_cov_values:
            if cov >= threshold:
                G_test.add_edge(node1, node2)
                edges_added += 1

        # Calculate connectivity ratio
        connectivity_ratio = edges_added / max_possible_edges
        target_distance = abs(connectivity_ratio - 0.5)

        # Update best E_c if this one is closer to target 0.5
        # Using <= allows picking a higher threshold if distances are equal
        if target_distance <= min_target_distance:
            min_target_distance = target_distance
            best_ec = threshold
            best_ratio = connectivity_ratio # Store the ratio for reporting

    elapsed = time.time() - start_time
    print(f"Critical Covariance Cutoff (Original Method) Ec = {best_ec:.6f}")
    print(f"  (Resulting graph connectivity ratio: {best_ratio:.3f}, Target: 0.5)")
    print(f"  Calculation finished ({elapsed:.2f}s).")
    return best_ec


# --- Graph Building ---

def build_graph(
    n_analysis_nodes: int,
    analysis_idx_to_residue_obj_map: dict,
    contact_frequency: np.ndarray,
    normalized_cov_matrix: np.ndarray,
    contact_freq_cutoff: float = 0.5,
    filtering_mode: str = 'fragmentation_pruning', # Added
    raw_covariance_matrix: np.ndarray = None,     # Added
    original_ec_cutoff: float = 0.0               # Added
):
    """
    Builds the protein graph based on contacts and covariance/correlation.
    Applies filtering based on the specified mode.
    Nodes correspond to the indices used in contact_frequency and normalized_cov_matrix.
    """
    print(f"Building graph with {n_analysis_nodes} nodes (residues) using filtering mode: {filtering_mode}...")
    start_time = time.time()
    G = nx.Graph()
    edge_weights = [] # Store weights of edges that are added

    # Add nodes (using 0 to n_analysis_nodes-1)
    for i in range(n_analysis_nodes):
        res_info = analysis_idx_to_residue_obj_map[i] # Fetch Residue object for label
        G.add_node(i, label=f"{res_info.name}{res_info.resSeq}") # Store readable label

    # Add edges based on filtering criteria
    added_edges = 0
    skipped_low_freq = 0
    skipped_low_cov = 0

    for i in range(n_analysis_nodes):
        for j in range(i + 1, n_analysis_nodes):
            # 1. Check contact frequency threshold (always applied)
            if contact_frequency[i, j] < contact_freq_cutoff:
                skipped_low_freq += 1
                continue

            # 2. Apply covariance magnitude filter if using 'original_ec' mode
            passes_cov_filter = True
            if filtering_mode == 'original_ec':
                if raw_covariance_matrix is None:
                     raise ValueError("Raw covariance matrix needed for 'original_ec' filtering mode.")
                abs_cov = abs(raw_covariance_matrix[i, j])
                if abs_cov < original_ec_cutoff:
                    passes_cov_filter = False
                    skipped_low_cov += 1

            # 3. If all filters passed, add the edge and calculate weight
            if passes_cov_filter:
                correlation = normalized_cov_matrix[i, j]
                weight = 1.0 - abs(correlation)
                weight = max(weight, 1e-9) # Ensure non-negative/non-zero for Dijkstra
                G.add_edge(i, j, weight=weight, correlation=correlation)
                edge_weights.append(weight)
                added_edges += 1

    elapsed = time.time() - start_time
    print(f"Graph building finished: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({elapsed:.2f}s).")
    print(f"  Edges skipped due to contact frequency < {contact_freq_cutoff}: {skipped_low_freq}")
    if filtering_mode == 'original_ec':
        print(f"  Edges skipped due to |covariance| < {original_ec_cutoff:.6f}: {skipped_low_cov}")
    if G.number_of_edges() == 0:
        print("Warning: No edges added to the graph. Check contact frequency cutoff and simulation dynamics.")
    return G, edge_weights

# --- Graph Pruning (Paper Method) ---

def find_critical_cutoff_paper(graph):
    """
    Determines the critical edge weight cutoff (Ec) based on the paper's
    description: Ec is chosen such that removing edges with weight < Ec
    results in 50% of the original edges belonging to disconnected
    subgraphs (i.e., components other than the largest connected component).

    Args:
        graph (nx.Graph): The initial graph *after* contact frequency filtering,
                          with edges weighted by E_ij = 1 - |NormCov|.

    Returns:
        float: The critical cutoff value Ec. Returns 0.0 if no edges or trivial graph.
    """
    n_total_edges = graph.number_of_edges()
    if n_total_edges == 0:
        print("Warning: Graph has no edges. Cannot determine critical cutoff. Returning 0.0")
        return 0.0

    edge_weights_dict = nx.get_edge_attributes(graph, 'weight')
    if not edge_weights_dict: # Should be caught by n_total_edges == 0, but safety check
         return 0.0

    # Get unique edge weights and sort them ascendingly
    unique_weights = sorted(list(set(edge_weights_dict.values())))

    target_disconnected_edges = 0.5 * n_total_edges

    best_ec = 0.0 # Default if no fragmentation occurs or all edges pruned instantly
    min_diff = float('inf')

    print(f"Finding critical cutoff Ec targeting {target_disconnected_edges:.1f} disconnected edges...")
    start_time = time.time()

    # Iterate through possible cutoff values (the edge weights themselves)
    # Start from slightly above 0 up to max weight
    tested_ecs = 0
    for ec_candidate in tqdm(unique_weights, desc="Testing Ec candidates", disable=len(unique_weights)<100):
        # Create a temporary graph keeping only edges with weight >= ec_candidate
        temp_graph = nx.Graph(((u, v, d) for u, v, d in graph.edges(data=True) if d['weight'] >= ec_candidate))

        edges_kept = temp_graph.number_of_edges()
        num_disconnected_edges = 0

        if edges_kept > 0:
            components = list(nx.connected_components(temp_graph))
            if len(components) > 1:
                # Find the largest connected component (LCC)
                lcc = max(components, key=len)
                # Sum edges in all components EXCEPT the LCC
                for component in components:
                    if component != lcc:
                        num_disconnected_edges += temp_graph.subgraph(component).number_of_edges()
            # If only 1 component, num_disconnected_edges remains 0
        else:
             # If Ec is so high that no edges are kept, all original edges are "disconnected"
             # but this framework aims for fragmentation, not obliteration.
             # The loop should find a suitable Ec before this point if fragmentation occurs.
             # If no fragmentation occurs even at max weight, Ec=0 might be returned.
             pass # num_disconnected_edges remains 0

        current_diff = abs(num_disconnected_edges - target_disconnected_edges)
        tested_ecs += 1

        # Update best_ec if this candidate is closer to the target
        # Using <= allows us to potentially pick a higher Ec that achieves the same best diff
        # which might correspond to a slightly more fragmented state if differences are equal.
        if current_diff <= min_diff:
            min_diff = current_diff
            best_ec = ec_candidate

    elapsed = time.time() - start_time
    print(f"Determined critical cutoff Ec = {best_ec:.6f} (Tested {tested_ecs} values, minimal difference to target: {min_diff:.2f}) ({elapsed:.2f}s)")
    # Handle edge case where no fragmentation occurs
    if best_ec == 0.0 and n_total_edges > 0 and len(unique_weights) > 0:
         max_weight = max(unique_weights)
         # Check if even removing edges up to max_weight causes fragmentation
         temp_graph_max = nx.Graph(((u, v, d) for u, v, d in graph.edges(data=True) if d['weight'] >= max_weight + 1e-9 )) # Check at highest weight level
         if nx.number_connected_components(temp_graph_max) <= 1:
              print("Warning: Graph did not significantly fragment across tested weights. Critical Ec might not be meaningful.")


    return best_ec


def prune_graph_paper(graph, apply_critical_pruning):
    """
    Prunes the graph according to the paper's description:
    1. Calculates critical Ec if requested.
    2. Removes edges with weight < Ec.

    Args:
        graph (nx.Graph): The initial graph *after* contact frequency filtering.
        apply_critical_pruning (bool): Whether to calculate and use critical Ec.

    Returns:
        nx.Graph: The pruned graph.
    """
    if graph.number_of_edges() == 0:
        print("Graph has no edges, skipping pruning.")
        return graph

    prune_cutoff_ec = 0.0
    if apply_critical_pruning:
        prune_cutoff_ec = find_critical_cutoff_paper(graph)
    else:
        print("Skipping paper's critical Ec weight-based pruning step.")
        return graph # Return the unpruned graph if not applying critical pruning

    print(f"Pruning graph using paper's method. Removing edges with weight < {prune_cutoff_ec:.6f}")
    start_time = time.time()
    pruned_graph = graph.copy()
    edges_to_remove = []
    for u, v, data in pruned_graph.edges(data=True):
        # *** Critical change: remove if weight IS LESS THAN Ec ***
        if data['weight'] < prune_cutoff_ec:
            edges_to_remove.append((u, v))

    pruned_graph.remove_edges_from(edges_to_remove)
    elapsed = time.time() - start_time
    removed_count = len(edges_to_remove)
    print(f"Pruning complete. Removed {removed_count} edges. Remaining edges: {pruned_graph.number_of_edges()} ({elapsed:.2f}s).")

    # Check connectivity after pruning
    if pruned_graph.number_of_edges() > 0:
        num_components = nx.number_connected_components(pruned_graph)
        if num_components > 1:
             print(f"Graph is disconnected after pruning ({num_components} components). Pathfinding might fail between components.")
        else:
             print("Graph remains connected after pruning.")
    elif graph.number_of_edges() > 0 and removed_count == graph.number_of_edges() :
         print("Warning: All edges were pruned by the Ec cutoff.")

    return pruned_graph

# --- Pathfinding and Centrality ---

def find_optimal_path(graph, start_node_idx, end_node_idx):
    """Finds the optimal (shortest weighted) path using Dijkstra."""
    print(f"Finding optimal path between analysis nodes {start_node_idx} and {end_node_idx}...")
    start_time = time.time()
    try:
        # Check if nodes exist in graph first
        if start_node_idx not in graph:
             print(f"Error: Start node {start_node_idx} not in the graph (possibly pruned).")
             return None, float('inf')
        if end_node_idx not in graph:
             print(f"Error: End node {end_node_idx} not in the graph (possibly pruned).")
             return None, float('inf')

        path = nx.dijkstra_path(graph, source=start_node_idx, target=end_node_idx, weight='weight')
        path_length = nx.dijkstra_path_length(graph, source=start_node_idx, target=end_node_idx, weight='weight')

        # Calculate product of correlations along path for interest
        path_corr_product = 1.0
        path_avg_corr = 0.0
        corrs = []
        for i in range(len(path) - 1):
             u, v = path[i], path[i+1]
             edge_data = graph.get_edge_data(u, v)
             if edge_data and 'correlation' in edge_data:
                  corr = abs(edge_data['correlation']) # Use absolute correlation
                  corrs.append(corr)
                  # path_corr_product *= corr # Product can become very small
             else:
                  corrs.append(0) # Should not happen if graph built correctly
        if corrs: path_avg_corr = np.mean(corrs)

        elapsed = time.time() - start_time
        print(f"Optimal path found: {len(path)} nodes ({elapsed:.2f}s).")
        print(f"  Path length (sum of 1-|corr| weights): {path_length:.4f}")
        print(f"  Average |correlation| along path: {path_avg_corr:.4f}")
        return path, path_length
    except nx.NetworkXNoPath:
        elapsed = time.time() - start_time
        print(f"Error: No path found between nodes {start_node_idx} and {end_node_idx} ({elapsed:.2f}s).")
        # Check if they are in the same connected component
        if start_node_idx in graph and end_node_idx in graph:
             for component in nx.connected_components(graph):
                 if start_node_idx in component and end_node_idx not in component:
                     print("  Reason: Start and end nodes are in different connected components after pruning.")
                     break
        return None, float('inf')
    except Exception as e:
         print(f"An unexpected error occurred during path finding: {e}")
         return None, float('inf')


def find_critical_residues(graph, top_n=10):
    """Identifies critical residues using betweenness centrality."""
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        print("Graph is empty or has no edges, cannot calculate centrality.")
        return {}

    print(f"Calculating betweenness centrality for {graph.number_of_nodes()} nodes...")
    start_time = time.time()
    try:
        # Use weight=None for unweighted centrality (topology-based bottleneck)
        # Or use weight='weight' if inverse correlation strength should influence centrality
        # Paper description implies topological importance ("disconnect the largest subgraph")
        # Using k=None calculates exact centrality. For large graphs, consider k=int(...) sampling.
        centrality = nx.betweenness_centrality(graph, weight=None, normalized=True, k=None)

        # Sort residues by centrality in descending order
        sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        elapsed = time.time() - start_time
        print(f"Centrality calculation finished ({elapsed:.2f}s).")
        print(f"Top {top_n} critical residues (based on Betweenness Centrality):")
        top_residues = {}
        for i, (node_idx, score) in enumerate(sorted_centrality[:top_n]):
             # Ensure node exists in graph to get label (should always be true)
             if node_idx in graph.nodes:
                  node_label = graph.nodes[node_idx]['label']
                  print(f"  {i+1}. Residue {node_label} (Node {node_idx}): {score:.4f}")
                  top_residues[node_idx] = score
             else:
                  print(f"Warning: Node index {node_idx} from centrality not found in graph nodes.")
        return top_residues
    except Exception as e:
        print(f"Error calculating centrality: {e}")
        return {}

# --- Visualization ---

def visualize_network(graph, pos, optimal_path, critical_residues_idx, filename="protein_network.png"):
    """Visualizes the network graph."""
    if not graph or graph.number_of_nodes() == 0:
        print("Cannot visualize empty graph.")
        return

    print("Preparing network visualization...")
    start_time = time.time()
    plt.figure(figsize=(14, 14)) # Slightly larger figure
    ax = plt.gca()
    ax.set_title("Protein Residue Network (Pruned Graph)")

    node_colors = []
    node_sizes = []
    edge_colors = []
    edge_widths = []

    # Determine node properties based on path/criticality
    path_nodes = set(optimal_path) if optimal_path else set()
    critical_nodes = set(critical_residues_idx)

    max_size = 250
    med_size = 120
    min_size = 40

    for node in graph.nodes():
        is_path = node in path_nodes
        is_critical = node in critical_nodes
        if is_path and is_critical:
            node_colors.append('purple') # Critical node on path
            node_sizes.append(max_size)
        elif is_path:
            node_colors.append('red')    # Path node
            node_sizes.append(max_size * 0.8)
        elif is_critical:
            node_colors.append('orange') # Critical node (not on path)
            node_sizes.append(med_size)
        else:
            node_colors.append('lightblue') # Normal node
            node_sizes.append(min_size)

    # Determine edge properties
    path_edges = set()
    if optimal_path and len(optimal_path) > 1:
        path_edges.update(set(zip(optimal_path[:-1], optimal_path[1:])))
        path_edges.update(set(zip(optimal_path[1:], optimal_path[:-1]))) # Add reverse pairs

    min_width = 0.3
    max_width = 2.5
    path_width = 3.0

    # Normalize edge weights (0 to 1) for coloring/width, lower weight = stronger correlation = darker/thicker
    weights = [d['weight'] for u, v, d in graph.edges(data=True)]
    min_w = min(weights) if weights else 0
    max_w = max(weights) if weights else 1
    max_w = max(max_w, min_w + 1e-6) # Avoid division by zero if all weights are identical

    for u, v, data in graph.edges(data=True):
        is_path_edge = (u, v) in path_edges
        if is_path_edge:
            edge_colors.append('red')
            edge_widths.append(path_width)
        else:
            # Optional: color/width by correlation strength (lower weight = darker/thicker)
            # weight_norm = (data['weight'] - min_w) / (max_w - min_w) # Normalize 0-1
            # edge_colors.append(plt.cm.Greys(weight_norm * 0.8 + 0.1)) # Map to grey intensity
            # edge_widths.append(min_width + (1 - weight_norm) * (max_width - min_width)) # Thicker for lower weight
            edge_colors.append('grey')
            edge_widths.append(min_width)

    # Draw the network using precomputed lists
    print(f"Drawing network ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)...")
    nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.6)
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9)

    # Add labels only for specific nodes to avoid clutter
    labels = {}
    for node in graph.nodes():
         if node in path_nodes or node in critical_nodes:
             labels[node] = graph.nodes[node]['label'] # Use stored label (e.g., "TYR504")

    # Draw labels with slight offset
    label_pos = {k: (v[0], v[1] + 0.02) for k, v in pos.items()} # Adjust offset as needed
    nx.draw_networkx_labels(graph, pos=label_pos, labels=labels, font_size=7, ax=ax, font_weight='bold')

    plt.axis('off') # Hide axes
    plt.tight_layout()
    elapsed = time.time() - start_time
    print(f"Saving visualization to {filename} ({elapsed:.2f}s)")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate optimal paths and critical residues in protein networks based on MD trajectory analysis.")

    # Input files
    parser.add_argument("pdb_file", help="Path to the PDB file (topology).")
    parser.add_argument("dcd_file", help="Path to the DCD file (trajectory).")

    # Path definition
    parser.add_argument("start_resid", type=int, help="Residue sequence number (1-based) for the start of the path.")
    parser.add_argument("end_resid", type=int, help="Residue sequence number (1-based) for the end of the path.")

    # Analysis Parameters
    parser.add_argument("--cov_type", choices=['coordinate', 'displacement_mean_dot', 'displacement_dot_mean'],
                        default='displacement_mean_dot',
                        help="Type of raw covariance matrix calculation: "
                             "'coordinate' (C-alpha coordinates), "
                             "'displacement_mean_dot' (Mean of Dot Products of unit C-alpha displacement deviations from mean), "
                             "'displacement_dot_mean' (Dot Product of Mean unit C-alpha displacement deviations from mean - Eq.6 in paper text).")
    parser.add_argument("--contact_cutoff", type=float, default=0.75,
                        help="Distance cutoff (nm) for defining residue contacts (C-beta distance, C-alpha for Gly). Default: 0.75 nm = 7.5 Angstrom.")
    parser.add_argument("--contact_freq", type=float, default=0.5,
                        help="Minimum contact frequency threshold [0,1] to include an edge in the initial graph. Default: 0.5.")
    parser.add_argument("--contact_atoms", choices=['cbeta', 'calpha'], default='calpha',
                        help="Atom type to use for contact distance calculation ('cbeta' for C-beta/Gly-CA, 'calpha' for C-alpha). Default: calpha.")

    # Filtering/Pruning Options
    # Note: The old --pruning_method argument is deprecated in favor of --filtering_mode
    parser.add_argument("--filtering_mode", choices=['contact_only', 'original_ec', 'fragmentation_pruning'], default='original_ec',
                        help="Method for filtering/pruning graph edges: "
                             "'contact_only' (Apply only contact frequency filter), "
                             "'original_ec' (Apply contact freq filter + covariance magnitude filter during build), "
                             "'fragmentation_pruning' (Apply contact freq filter, then prune by weight targeting graph fragmentation). Default: fragmentation_pruning.")

    # Output Options
    parser.add_argument("-n", "--top_n_critical", type=int, default=10,
                        help="Number of top critical residues (by betweenness centrality) to report and highlight.")
    parser.add_argument("--out_image", default="protein_network.png",
                        help="Filename for the output network visualization graph.")

    args = parser.parse_args()

    print("--- Starting Protein Network Analysis ---")
    overall_start_time = time.time()

    # --- Step 1: Load Trajectory ---
    traj = load_trajectory(args.pdb_file, args.dcd_file)

    # --- Step 2: Setup Mappings ---
    # Map residue sequence numbers (1-based from PDB) to internal 0-based residue indices
    res_map_seq_to_idx = {}
    res_map_idx_to_res = {}
    for res in traj.topology.residues:
         if res.is_protein: # Consider only protein residues
            res_map_seq_to_idx[res.resSeq] = res.index
            res_map_idx_to_res[res.index] = res

    # Get 0-based residue indices for start/end nodes
    try:
        start_node_residue_idx = res_map_seq_to_idx[args.start_resid]
        end_node_residue_idx = res_map_seq_to_idx[args.end_resid]
        print(f"Mapping Start Residue {args.start_resid} to internal residue index {start_node_residue_idx}")
        print(f"Mapping End Residue {args.end_resid} to internal residue index {end_node_residue_idx}")
    except KeyError as e:
        print(f"Error: Residue sequence number {e} not found in the protein residues of the topology.")
        exit(1)

    # --- Step 3: Select Atoms and Create Analysis Index Mapping ---
    # Select atoms for contact frequency analysis based on user choice
    contact_atom_indices, selected_atom_map_resid_to_atomid = select_atoms_for_contact(
        traj,
        atom_type=args.contact_atoms
    )
    # These residues (keys of selected_atom_map_resid_to_atomid) form the basis of our analysis network
    analysis_residue_indices = sorted(selected_atom_map_resid_to_atomid.keys())
    n_analysis_nodes = len(analysis_residue_indices)
    if n_analysis_nodes == 0:
         print("Error: No valid residues found for analysis after atom selection.")
         exit(1)

    # Create mappings between the analysis index (0 to N-1) and the original residue index
    analysis_idx_to_resid_map = {i: res_idx for i, res_idx in enumerate(analysis_residue_indices)}
    resid_to_analysis_idx_map = {v: k for k, v in analysis_idx_to_resid_map.items()}
    # Also map analysis index directly to residue object for convenience
    analysis_idx_residue_obj_map = {i: res_map_idx_to_res[analysis_idx_to_resid_map[i]] for i in range(n_analysis_nodes)}

    # Select C-alpha atoms corresponding *only* to the residues included in the analysis network
    ca_indices_for_analysis = []
    for i in range(n_analysis_nodes):
         res_idx = analysis_idx_to_resid_map[i]
         ca_atom = next((atom for atom in res_map_idx_to_res[res_idx].atoms if atom.name == 'CA'), None)
         if ca_atom:
             ca_indices_for_analysis.append(ca_atom.index)
         else:
             # This should not happen if residue selection worked, but as a safeguard:
             print(f"Critical Error: Could not find CA atom for included residue index {res_idx}. Exiting.")
             exit(1)
    ca_indices_for_analysis = np.array(ca_indices_for_analysis)
    if len(ca_indices_for_analysis) != n_analysis_nodes:
         print("Critical Error: Mismatch between number of analysis residues and found C-alpha atoms. Exiting.")
         exit(1)

    # Get analysis indices for start/end nodes
    try:
        start_analysis_idx = resid_to_analysis_idx_map[start_node_residue_idx]
        print(f"Mapping Start Residue Index {start_node_residue_idx} to analysis index {start_analysis_idx}")
    except KeyError:
        print(f"Error: Start residue {args.start_resid} (index {start_node_residue_idx}) was excluded during atom selection for contacts (using {args.contact_atoms}). Cannot proceed.")
        exit(1)
    try:
        end_analysis_idx = resid_to_analysis_idx_map[end_node_residue_idx]
        print(f"Mapping End Residue Index {end_node_residue_idx} to analysis index {end_analysis_idx}")
    except KeyError:
        print(f"Error: End residue {args.end_resid} (index {end_node_residue_idx}) was excluded during atom selection for contacts (using {args.contact_atoms}). Cannot proceed.")
        exit(1)


    # --- Step 4: Calculate Contact Frequency ---
    # Use the indices of the CB/Gly-CA atoms selected earlier
    contact_frequency = calculate_contact_frequency(traj, contact_atom_indices, distance_cutoff=args.contact_cutoff)
    # contact_frequency matrix dimensions are (n_analysis_nodes, n_analysis_nodes)

    # --- Step 5: Calculate Raw Covariance ---
    # Use the indices of the CA atoms corresponding to the analysis residues
    if args.cov_type == 'coordinate':
        raw_cov_matrix = calculate_covariance_coordinate(traj, ca_indices_for_analysis)
    elif args.cov_type == 'displacement_mean_dot':
        raw_cov_matrix = calculate_covariance_displacement_mean_of_dot(traj, ca_indices_for_analysis)
    elif args.cov_type == 'displacement_dot_mean':
        raw_cov_matrix = calculate_covariance_displacement_dot_of_mean(traj, ca_indices_for_analysis)
    else: # Should not happen
        raise ValueError("Internal Error: Invalid covariance type specified.")
    # raw_cov_matrix dimensions are (n_analysis_nodes, n_analysis_nodes)

    # --- Step 6: Normalize Covariance -> Correlation Matrix ---
    norm_cov_matrix = normalize_covariance(raw_cov_matrix)
    # norm_cov_matrix dimensions are (n_analysis_nodes, n_analysis_nodes)

    # --- Step 7: Determine Filtering/Pruning Cutoff (if needed) ---
    original_ec_cutoff_value = 0.0
    if args.filtering_mode == 'original_ec':
        original_ec_cutoff_value = find_original_critical_ec(
            n_analysis_nodes,
            raw_cov_matrix,
            contact_frequency,
            args.contact_freq
        )
    # Note: The fragmentation_pruning cutoff is calculated later inside prune_graph_paper

    # --- Step 8: Build Graph (Applying Filters) ---
    # Uses contact_frequency and norm_cov_matrix, both indexed 0 to n_analysis_nodes-1
    # Also passes raw_cov_matrix and original_ec_cutoff if using that filtering mode
    graph_after_build, edge_weights = build_graph(
        n_analysis_nodes=n_analysis_nodes,
        analysis_idx_to_residue_obj_map=analysis_idx_residue_obj_map,
        contact_frequency=contact_frequency,
        normalized_cov_matrix=norm_cov_matrix,
        contact_freq_cutoff=args.contact_freq,
        filtering_mode=args.filtering_mode,
        raw_covariance_matrix=raw_cov_matrix if args.filtering_mode == 'original_ec' else None,
        original_ec_cutoff=original_ec_cutoff_value if args.filtering_mode == 'original_ec' else 0.0
    )

    # --- Step 9: Prune Graph (Fragmentation Method or None) ---
    # Only apply fragmentation pruning if selected and graph has edges
    if args.filtering_mode == 'fragmentation_pruning' and graph_after_build.number_of_edges() > 0:
         print("\n--- Applying Fragmentation Pruning ---")
         # Rename the function call to reflect its purpose
         final_graph = prune_graph_paper( # This function now specifically handles the fragmentation pruning
             graph_after_build,
             apply_critical_pruning=True # Always true if we reach here in this mode
         )
    elif args.filtering_mode == 'contact_only':
         print("\n--- Skipping Pruning (contact_only mode) ---")
         final_graph = graph_after_build
    elif args.filtering_mode == 'original_ec':
         print("\n--- Skipping Pruning (original_ec filtering applied during build) ---")
         final_graph = graph_after_build
    else: # Should not happen
         final_graph = graph_after_build # Default to the built graph

    # --- Step 10: Determine Optimal Path ---
    # Use the final graph after potential pruning
    optimal_path_analysis_idx, path_length = find_optimal_path(
        final_graph,
        start_analysis_idx,
        end_analysis_idx
    )

    # Convert path back to original residue sequence numbers for reporting
    optimal_path_resids = []
    if optimal_path_analysis_idx:
        optimal_path_resids = [analysis_idx_residue_obj_map[idx].resSeq for idx in optimal_path_analysis_idx]
        print(f"\nOptimal path (Residue IDs): {' -> '.join(map(str, optimal_path_resids))}")
    else:
        print("\nOptimal path: Not found.")

    # --- Step 11: Identify Critical Residues (Bottlenecks) ---
    # Uses the *final* graph after potential pruning
    critical_residues_analysis_idx = find_critical_residues(
        final_graph,
        args.top_n_critical
    )

    # Convert critical residues back to original residue info for reporting (already printed inside function)
    # critical_residues_resids = {analysis_idx_residue_obj_map[idx].resSeq : score for idx, score in critical_residues_analysis_idx.items()}

    # --- Step 12: Visualize Network ---
    # Get C-alpha positions from the first frame for layout (using only analysis atoms)
    if final_graph.number_of_nodes() > 0: # Only visualize if graph exists
         ca_pos_all_analysis = traj.xyz[0, ca_indices_for_analysis, :]
         # Create position dictionary mapping analysis index (0..N-1) to CA position
         node_positions = {i: ca_pos_all_analysis[i, 0:2] for i in range(n_analysis_nodes)} # Use only X, Y for 2D layout

         visualize_network(
             final_graph, # Use the final graph for visualization
             node_positions,
             optimal_path_analysis_idx,
             critical_residues_analysis_idx.keys(), # Pass only the indices
             filename=args.out_image
         )
    else:
         print("Skipping visualization as the pruned graph has no nodes.")


    overall_elapsed = time.time() - overall_start_time
    print(f"\n--- Analysis Finished ---")
    print(f"Total execution time: {overall_elapsed:.2f} seconds.")
    if optimal_path_resids:
        print(f"Optimal path found between {args.start_resid} and {args.end_resid}.")
    else:
        print(f"No optimal path found between {args.start_resid} and {args.end_resid}.")
    print(f"Top critical residues identified.")
    print(f"Visualization saved to: {args.out_image}")

# visualize_residues_schematic.py - clean wireframe backbone with translucent colored paths
from chimerax.core.commands import run


def visualize_residues(session,
                       path_residues: str,
                       all_residues: str,
                       bulk_radius: float = 0.06,
                       path_radius: float = 0.35):
    """
    Show protein as clean wireframe with translucent colored paths.
    White background, gray translucent backbone, no cartoons.

    * Entire backbone → gray, 85% transparent, very thin tubes
    * Path residues   → translucent colored, thicker tubes
    * All bonds shown, atoms hidden except special cases
    """

    print("=== VISUALIZE_RESIDUES FUNCTION CALLED ===")  # Debug
    
    run(session, "set bgColor white")     # White background
    run(session, "hide atoms")            # Hide all atom representations
    run(session, "hide cartoons")         # Remove ribbons completely
    
    # ── Set desired orientation ──────────────────────────────────
    run(session, "turn x 90")
    run(session, "turn z 90")
    run(session, "turn x 90")
    run(session, "turn z -10")
    
    # ── Show all alpha-carbons and create backbone tubes ─────────────────────────
    run(session, f"show {all_residues}@CA")
    
    # Create backbone tube - fixed syntax with proper quoting
    run(session, f'shape tube {all_residues}@CA radius {bulk_radius} color gray name backbone_tube')
    run(session, f"transparency 85 surfaces name backbone_tube")
    
    # ── Path residues: thicker, translucent colored tubes ─────────────
    # Create path tube with different name to avoid conflicts
    run(session, f'shape tube {path_residues}@CA radius {path_radius} color royalblue name path_tube')
    run(session, f"transparency 30 surfaces name path_tube")
    
    # Clean lighting, no silhouettes
    run(session, "graphics silhouettes false")
    run(session, "lighting soft")
    
    print("=== VISUALIZE_RESIDUES FUNCTION COMPLETE ===")  # Debug


# Alternative approach using cartoon style for hollow wireframe look
def visualize_residues_cartoon_style(session,
                                   path_residues: str,
                                   all_residues: str,
                                   backbone_width: float = 0.1,
                                   path_width: float = 0.4):
    """
    Alternative approach using cartoon representation for hollow wireframe look.
    This might give you the "hollow wire" appearance you're looking for.
    """
    
    print("=== VISUALIZE_RESIDUES CARTOON STYLE CALLED ===")
    
    run(session, "set bgColor white")
    run(session, "hide atoms")
    
    # ── Set desired orientation ──────────────────────────────────
    run(session, "turn x 90")
    run(session, "turn z 90")
    run(session, "turn x 90")
    run(session, "turn z -10")
    
    # Show cartoons for wireframe effect
    run(session, f"show {all_residues} cartoons")
    
    # Style the backbone as thin wireframe
    run(session, f"cartoon style {all_residues} width {backbone_width} thickness 0.05")
    run(session, f"color {all_residues} gray")
    run(session, f"transparency {all_residues} 85")
    
    # Style path residues as thicker wireframe
    run(session, f"cartoon style {path_residues} width {path_width} thickness 0.1")
    run(session, f"color {path_residues} royalblue")
    run(session, f"transparency {path_residues} 30")
    
    # Clean lighting
    run(session, "graphics silhouettes false")
    run(session, "lighting soft")
    
    print("=== VISUALIZE_RESIDUES CARTOON STYLE COMPLETE ===")
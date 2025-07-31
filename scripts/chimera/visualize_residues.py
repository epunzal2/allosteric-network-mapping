# visualize_residues.py - clean wireframe backbone with translucent colored paths
from chimerax.core.commands import run


def visualize_residues(session,
                       path_residues: str,
                       all_residues: str,
                       bulk_radius: float = 0.06,
                       path_radius: float = 0.35):
    """
    Show protein as clean wireframe with translucent colored paths.
    White background, gray translucent backbone, no cartoons.

    * Entire backbone → gray, 85% transparent, very thin sticks
    * Path residues   → translucent colored, thicker sticks
    * All bonds shown, atoms hidden except special cases
    """

    print("=== VISUALIZE_RESIDUES FUNCTION CALLED ===")  # Debug
    
    run(session, "set bgColor white")     # White background
    run(session, "hide atoms")            # Hide all atom representations
    run(session, "hide cartoons")         # Remove ribbons completely
    
    # ── Show all bonds in stick representation ─────────────────────────
    run(session, f"show {all_residues} bonds")
    run(session, f"style {all_residues} stick")
    
    # ── Whole protein: thin, translucent gray sticks ───────────────────
    run(session, f"size {all_residues} stickRadius {bulk_radius}")
    run(session, f"color {all_residues} gray")
    run(session, f"transparency {all_residues} 85")  # Very transparent
    
    # ── Path residues: thicker, translucent colored sticks ─────────────
    run(session, f"size {path_residues} stickRadius {path_radius}")
    run(session, f"transparency {path_residues} 30")  # Translucent
    
    # Clean lighting, no silhouettes
    run(session, "graphics silhouettes false")
    run(session, "lighting soft")
    
    print("=== VISUALIZE_RESIDUES FUNCTION COMPLETE ===")  # Debug
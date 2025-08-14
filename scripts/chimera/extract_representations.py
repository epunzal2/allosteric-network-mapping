from chimerax.core.commands import run
import numpy

def format_color(color):
    """Formats a color array into a user-friendly string."""
    if color is None:
        return "N/A"
    # Assuming color is a chimerax.core.colors.Color object or similar
    return f"({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}, {color[3]:.2f})"

# Open the session file
run(session, 'open "Data/arek_samples/Arek_0801_ChimeraX_representation_1Aug25.cxs"')

with open("analysis_results/cxs_representations.txt", "w") as f_reps:

    f_reps.write("ChimeraX Session Residue Representations\n")
    f_reps.write("======================================\n\n")

    if not session.models:
        f_reps.write("No models found in the session.\n")

    for m in session.models:
        f_reps.write(f"--- Model: {m.id_string} ({m.name}) ---\n")
        if hasattr(m, 'opened_as') and m.opened_as:
            f_reps.write(f"  Source File: {m.opened_as[0]}\n")
        else:
            f_reps.write(f"  Source File: Not available\n")

        if hasattr(m, "residues") and m.residues:
            f_reps.write(f"  Residue Ribbon Colors ({len(m.residues)} total):\n")
            for r in m.residues:
                if hasattr(r, 'ribbon_color'):
                     f_reps.write(f"    - Residue {r.number}: {format_color(r.ribbon_color)}\n")
        f_reps.write("\n")

    f_reps.write("Extraction complete.\n")
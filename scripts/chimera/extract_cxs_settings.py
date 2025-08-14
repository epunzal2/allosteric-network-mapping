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

# Get the view
view = session.view

with open("analysis_results/cxs_settings.txt", "w") as f_settings:

    # Write headers
    f_settings.write("ChimeraX Session Settings\n")
    f_settings.write("=========================\n\n")

    # --- Camera and View Settings ---
    f_settings.write("--- View Settings ---\n")
    if hasattr(view, 'camera') and hasattr(view.camera, 'position'):
        view_matrix = view.camera.position.matrix
        f_settings.write(f"View Camera Position (Matrix):\n{numpy.array2string(view_matrix, precision=4)}\n")
    else:
        f_settings.write("View Camera Position: Not available\n")
    f_settings.write(f"Background Color: {format_color(view.background_color)}\n\n")

    # --- Model Information ---
    f_settings.write("--- Model Details ---\n")
    if not session.models:
        f_settings.write("No models found in the session.\n")

    for m in session.models:
        # --- Write to settings file (original content) ---
        f_settings.write(f"Model ID: {m.id_string}, Name: {m.name}\n")
        if hasattr(m, 'opened_as') and m.opened_as:
            f_settings.write(f"  Source File: {m.opened_as[0]}\n")
        else:
            f_settings.write(f"  Source File: Not available\n")

        if hasattr(m, 'cartoon'):
            f_settings.write(f"  Cartoon Style: {m.cartoon.style}\n")
            f_settings.write(f"  Cartoon Color: {format_color(m.cartoon.color)}\n")

        if hasattr(m, 'surface') and m.surface.shown:
            f_settings.write(f"  Surface Representation: Visible\n")
            f_settings.write(f"  Surface Color: {format_color(m.surface.color)}\n")

        if hasattr(m, "atoms") and m.atoms:
            f_settings.write(f"  Atom Colors (first 5 of {len(m.atoms)}):\n")
            for a in m.atoms[:5]:
                f_settings.write(f"    - Atom {a.residue.number}{a.name}: {format_color(a.color)}\n")

        if hasattr(m, "residues") and m.residues:
            f_settings.write(f"  Residue Ribbon Colors (first 5 of {len(m.residues)}):\n")
            for r in m.residues[:5]:
                if hasattr(r, 'ribbon_color'):
                     f_settings.write(f"    - Residue {r.number}: {format_color(r.ribbon_color)}\n")

        if hasattr(m, 'pseudo_bonds') and m.pseudo_bonds:
            f_settings.write(f"  PseudoBonds ({len(m.pseudo_bonds)} total):\n")
            for pb in m.pseudo_bonds:
                atom1_str = f"{pb.atoms[0].residue.number}{pb.atoms[0].name}"
                atom2_str = f"{pb.atoms[1].residue.number}{pb.atoms[1].name}"
                f_settings.write(f"    - Bond: {atom1_str} to {atom2_str}, Color: {format_color(pb.color)}\n")
        f_settings.write("\n")

    f_settings.write("Extraction complete.\n")

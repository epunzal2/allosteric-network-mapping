import os
import pandas as pd
from pathlib import Path

def combine_optimal_paths():
    """
    This script combines optimal path data from various subdirectories into two summary Excel files.

    It iterates through subdirectories within a specified input directory, reading two types of CSV files:
    - 'paths_all.csv': Contains optimal path data.
    - 'paths_all_orig_label.csv': Contains optimal path data with original labels.

    The data from these files is then compiled into two separate Excel workbooks, with each subdirectory's
    data being placed on a new sheet named after the subdirectory.

    Input Directory Structure:
    analysis_results/reports/optimal_paths_csv/
    ├── subdirectory_1/
    │   ├── paths_all.csv
    │   └── paths_all_orig_label.csv
    ├── subdirectory_2/
    │   ├── paths_all.csv
    │   └── paths_all_orig_label.csv
    └── ...

    Output Files:
    - analysis_results/reports/optimal_paths_all.xlsx
    - analysis_results/reports/optimal_paths_all_orig_label.xlsx
    """
    # Define the input directory containing the subdirectories with CSV files
    input_dir = Path("analysis_results/reports/optimal_paths_csv")

    # Define the output directory for the combined Excel files
    output_dir = Path("analysis_results/reports/optimal_paths_csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the paths for the output Excel files
    output_excel_all = output_dir / "optimal_paths_all.xlsx"
    output_excel_all_orig_label = output_dir / "optimal_paths_all_orig_label.xlsx"

    # Create Excel writers to save multiple dataframes to different sheets
    with pd.ExcelWriter(output_excel_all, engine='openpyxl') as writer_all, \
         pd.ExcelWriter(output_excel_all_orig_label, engine='openpyxl') as writer_all_orig_label:

        # Iterate through all subdirectories in the input directory
        for subdir in sorted([d for d in input_dir.iterdir() if d.is_dir()]):
            # Get the name of the subdirectory to use as the sheet name
            sheet_name = subdir.name

            # --- Process paths_all.csv ---
            paths_all_csv = subdir / "paths_all.csv"
            if paths_all_csv.is_file():
                try:
                    # Read the CSV file into a pandas DataFrame
                    df_all = pd.read_csv(paths_all_csv)
                    # Write the DataFrame to a new sheet in the 'optimal_paths_all.xlsx' workbook
                    df_all.to_excel(writer_all, sheet_name=sheet_name, index=False)
                    print(f"Added sheet '{sheet_name}' to {output_excel_all.name} from {paths_all_csv}")
                except Exception as e:
                    print(f"Error processing {paths_all_csv}: {e}")
            else:
                print(f"File not found: {paths_all_csv}")

            # --- Process paths_all_orig_label.csv ---
            paths_all_orig_label_csv = subdir / "paths_all_orig_label.csv"
            if paths_all_orig_label_csv.is_file():
                try:
                    # Read the CSV file into a pandas DataFrame
                    df_all_orig_label = pd.read_csv(paths_all_orig_label_csv)
                    # Write the DataFrame to a new sheet in the 'optimal_paths_all_orig_label.xlsx' workbook
                    df_all_orig_label.to_excel(writer_all_orig_label, sheet_name=sheet_name, index=False)
                    print(f"Added sheet '{sheet_name}' to {output_excel_all_orig_label.name} from {paths_all_orig_label_csv}")
                except Exception as e:
                    print(f"Error processing {paths_all_orig_label_csv}: {e}")
            else:
                print(f"File not found: {paths_all_orig_label_csv}")

    print("\nScript finished.")
    print(f"Combined Excel file created at: {output_excel_all}")
    print(f"Combined Excel file with original labels created at: {output_excel_all_orig_label}")

    # --- Combine the two generated Excel files ---
    print("\nCombining the two Excel files into a single workbook...")
    output_excel_combined = output_dir / "optimal_paths_combined.xlsx"

    try:
        # Read the two Excel files
        xls_all = pd.ExcelFile(output_excel_all)
        xls_all_orig_label = pd.ExcelFile(output_excel_all_orig_label)

        # Get the sheet names from both files
        sheets_all = xls_all.sheet_names
        sheets_all_orig_label = xls_all_orig_label.sheet_names

        # Find the common sheets
        common_sheets = sorted(list(set(sheets_all) & set(sheets_all_orig_label)))

        if not common_sheets:
            print("No common sheets found between the two Excel files. Nothing to combine.")
        else:
            with pd.ExcelWriter(output_excel_combined, engine='openpyxl') as writer_combined:
                for sheet_name in common_sheets:
                    try:
                        # Read the data from each sheet
                        df1 = pd.read_excel(xls_all, sheet_name=sheet_name)
                        df2 = pd.read_excel(xls_all_orig_label, sheet_name=sheet_name)

                        # Concatenate the dataframes side-by-side
                        combined_df = pd.concat([df1, df2], axis=1)

                        # --- New data extraction feature ---
                        # Iterate through each column of the combined_df
                        for col in combined_df.columns:
                            # Check if the column's values are strings and start with 'a', 'b', or 'g'
                            if combined_df[col].dtype == 'object':
                                # Create a boolean mask for rows where the value is a string and starts with a, b, or g
                                mask = combined_df[col].str.startswith(('a', 'b', 'g'), na=False)
                                if mask.any():
                                    # Create the new column name
                                    new_col_name = f"{col}_prefix"
                                    # Create the new column with the first letter, or an empty string
                                    combined_df[new_col_name] = combined_df[col].apply(lambda x: x[0] if isinstance(x, str) and x and x[0] in ['a', 'b', 'g'] else None)

                        # Write the combined dataframe to the new Excel file
                        combined_df.to_excel(writer_combined, sheet_name=sheet_name, index=False)
                        print(f"Combined sheet '{sheet_name}' and added to {output_excel_combined.name}")
                    except Exception as e:
                        print(f"Error processing sheet '{sheet_name}': {e}")
            print(f"\nSuccessfully created combined Excel file: {output_excel_combined}")

    except FileNotFoundError as e:
        print(f"Error: One of the input Excel files was not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the combination step: {e}")


if __name__ == "__main__":
    combine_optimal_paths()
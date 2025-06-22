import pandas as pd
import os
import shutil

# Load the CSV files
file_a = 'GCNN/results_single/summary.csv'  # Replace with your actual filename
file_b = 'GCNN/results_single_alpha0/summary.csv'  # Replace with your actual filename

df_a = pd.read_csv(file_a)
df_b = pd.read_csv(file_b)

# Identify the bad rows to remove
bad_rows = (df_a['alpha'] == 0) & (df_a['learn_alpha'] == True)

# Get the list of run_ids to delete files/folders
run_ids_to_delete = df_a.loc[bad_rows, 'run_id'].tolist()

# Optional: base directory if files/folders are stored in a specific location
base_dir = 'GCNN/results_single/'  # Change to your target directory if needed

# Delete corresponding files/folders
for run_id in run_ids_to_delete:
    path = os.path.join(base_dir, str(run_id))
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)  # Remove folder and all contents
            print(f"Deleted folder: {path}")
        elif os.path.isfile(path):
            os.remove(path)  # Remove file
            print(f"Deleted file: {path}")
        else:
            print(f"No file or folder found: {path}")
    except Exception as e:
        print(f"Error deleting {path}: {e}")

# Remove the bad rows from df_a
filtered_df_a = df_a[~bad_rows]

# Append all rows from df_b
updated_df = pd.concat([filtered_df_a, df_b], ignore_index=True)

# Save the updated CSV
updated_df.to_csv('updates.csv', index=False)

print("File cleaned and updated successfully!")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv('updates.csv')  # Replace with your file name

# Get unique targets
targets = df['targets'].unique()

# Process each target group
for target in sorted(targets):
    print(f"\n=== Analysis for Target: {target} ===")

    # Filter for the current target
    df_target = df[df['targets'] == target].copy()

    # Sort by lowest test_mae
    df_target_sorted = df_target.sort_values(by='test_mae')
    print("\nTop 5 Performers for this Target:")
    print(df_target_sorted[['run_id', 'test_mae', 'alpha', 'learn_alpha', 'pooling']].head())


    g = sns.FacetGrid(df_target, col='pooling', row='learn_alpha', margin_titles=True)
    g.map_dataframe(sns.boxplot, x='alpha', y='test_mae')
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Test MAE Distribution per Pooling and Learn Alpha (Target: {target})')
    plt.show()



print("\n✅ Analysis Complete for All Targets.")

# Mean test_mae grouped by target and pooling
mean_mae_summary = df.groupby(['targets', 'pooling'])['test_mae'].mean().reset_index()

print("Mean Test MAE per Target and Pooling:")
print(mean_mae_summary)

import pandas as pd
import numpy as np
import os
import glob
import argparse
import json
from collections import defaultdict
from scipy.stats import gamma, norm

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not found. Please run 'pip install matplotlib' to use the --output_plot_dir feature.")
    plt = None

try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not found. Please run 'pip install seaborn' to use the --output_plot_dir feature.")
    sns = None

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    print("Warning: sklearn not found. Please run 'pip install scikit-learn' to use GMM fitting.")
    GaussianMixture = None


class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Handle potential NaN/Inf
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_all_scores(scores_dir: str) -> (pd.DataFrame, list):
    """
    Loads and concatenates all micro-segment score CSVs from a directory.
    
    It specifically excludes files ending in '_gt_segments.csv' to focus
    on the micro-segment analysis needed for beam search tuning.
    """
    all_files = glob.glob(os.path.join(scores_dir, "*.csv"))
    
    # --- IMPORTANT ---
    # We only want the micro-segment files, not the GT-segment files.
    micro_segment_files = [
        f for f in all_files if '_gt_segments' not in os.path.basename(f)
    ]
    
    if not micro_segment_files:
        print(f"Error: No micro-segment CSV files found in {scores_dir}")
        print("Did you run generate_segment_scores.py first?")
        return pd.DataFrame(), []

    print(f"Loading {len(micro_segment_files)} score files...")
    dfs = [pd.read_csv(f) for f in micro_segment_files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Identify which columns are actions and which are metadata
    known_meta_cols = [
        'segment_index', 'start_time_s', 'end_time_s', 
        'gt_actions', 'has_boundary'
    ]
    
    # Anything not in the metadata list is an action-score column
    all_action_names = sorted(
        [col for col in df.columns if col not in known_meta_cols]
    )
    
    print(f"Found {len(all_action_names)} actions: {', '.join(all_action_names)}")
    return df, all_action_names

def calculate_distributions(df: pd.DataFrame, all_action_names: list) -> dict:
    """
    Calculates the full score distribution matrix (mean, std, etc.)
    for every action against every other action.
    """
    
    # --- Data Cleaning ---
    # For this analysis, we only want segments that have a SINGLE,
    # unambiguous ground-truth label.
    # We exclude:
    #   - Boundaries (e.g., "action1;action2")
    #   - Background ("N/A", "BG", or NaN)
    df_clean = df[
        ~df['gt_actions'].str.contains(';', na=True) &  # Exclude boundaries
        df['gt_actions'].notna() &                       # Exclude NaNs
        (df['gt_actions'] != 'N/A') &                   # Exclude 'N/A'
        (df['gt_actions'] != 'BG')                      # Exclude explicit 'BG'
    ].copy()
    
    print(f"\nTotal segments: {len(df)}, Clean segments for analysis: {len(df_clean)}")

    # This will hold the final results
    # Format: all_stats[true_action][comparison_action] = {stats}
    all_stats = defaultdict(dict)
    
    # Loop over every "true" action (from the gt_actions column)
    for true_action in all_action_names:
        
        # Get all segments where this action was the ground truth
        segments_for_this_action = df_clean[df_clean['gt_actions'] == true_action]
        
        if segments_for_this_action.empty:
            print(f"  - No clean segments found for GT: '{true_action}'. Skipping.")
            continue
            
        # Now, for this set of segments, get the score distributions
        # against ALL action prototype sets.
        for comparison_action in all_action_names:
            
            # Get the column of scores (e.g., all 'stir' segments'
            # scores against 'add_water' prototypes)
            scores = segments_for_this_action[comparison_action]
            scores_to_fit = scores.dropna()

            dist_params = None
            # Calculate and store the statistics
            stats = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                '25_percentile': scores.quantile(0.25),
                '50_percentile (median)': scores.quantile(0.50),
                '75_percentile': scores.quantile(0.75),
                'count': int(scores.count()),
                'dist_fit': dist_params
            }

            if len(scores_to_fit) > 3: # Need a few data points to fit
                try:
                    # Fit a Gamma distribution
                    # floc=0 forces the distribution to start at 0 (DTW scores > 0)
                    shape_a, loc, scale = gamma.fit(scores_to_fit, floc=0)
                    dist_params = {
                        'dist': 'gamma',
                        'a': shape_a, # 'a' is the shape parameter for gamma
                        'loc': loc,
                        'scale': scale
                    }
                    stats['dist_fit'] = dist_params
                except Exception as e:
                    # Fit can fail (e.g., all values identical)
                    if true_action == comparison_action: # Only print for the important in-class one
                         print(f"    - WARNING: Gamma fit failed for '{true_action}' vs '{comparison_action}': {e}")
            
            if true_action == comparison_action and GaussianMixture is not None:
                if len(scores_to_fit) > 10: # Need a decent amount of data for GMM
                    try:
                        scores_for_gmm = scores_to_fit.values.reshape(-1, 1)
                        n_components_range = range(1, 4) # Test k=1, 2, 3
                        bics = []
                        gmms = []
                        
                        for n in n_components_range:
                            gmm = GaussianMixture(n_components=n,
                                                  random_state=42, 
                                                  covariance_type='full',
                                                  reg_covar=1e-3)
                            gmm.fit(scores_for_gmm)
                            gmms.append(gmm)
                            bics.append(gmm.bic(scores_for_gmm))
                        
                        best_k_idx = np.argmin(bics)
                        best_gmm = gmms[best_k_idx]
                        
                        stats['gmm_fit'] = {
                            'best_k': best_gmm.n_components,
                            'weights': best_gmm.weights_,
                            'means': best_gmm.means_,
                            'covariances': best_gmm.covariances_
                        }
                    except Exception as e:
                        print(f"    - WARNING: GMM fit failed for '{true_action}': {e}")

            all_stats[true_action][comparison_action] = stats
            
    return dict(all_stats), df_clean

def print_stats_summary(all_stats: dict):
    """
    Prints a human-readable summary of the statistics to the console.
    """
    print("\n" + "="*80)
    print("--- SCORE DISTRIBUTION ANALYSIS ---")
    print("="*80)

    for true_action, comparisons in all_stats.items():
        print(f"\n📊 --- Stats for Ground-Truth Action: '{true_action}' ---")
        
        # 1. Print In-Class scores first for easy comparison
        if true_action in comparisons:
            stats = comparisons[true_action]
            fit_indicator = ""
            if stats.get('dist_fit') and stats['dist_fit']['dist'] == 'gamma':
                fit_indicator = "✅ (Gamma Fit OK)"

            gmm_indicator = ""
            if stats.get('gmm_fit'):
                k = stats['gmm_fit']['best_k']
                gmm_indicator = f"✅ (GMM Fit OK, k={k})"
            else:
                gmm_indicator = "❌ (No GMM Fit)"

            print(f"  ✅ IN-CLASS vs '{true_action}': {fit_indicator} | {gmm_indicator}")
            print(f"     Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"     Min:  {stats['min']:.4f}, Median: {stats['50_percentile (median)']:.4f}, Max: {stats['max']:.4f}")
            print(f"     (Based on {stats['count']} segments)")

        # 2. Print all Out-of-Class scores
        for comparison_action, stats in comparisons.items():
            if comparison_action == true_action:
                continue # Already printed
                
            print(f"  ❌ OUT-OF-CLASS vs '{comparison_action}':")
            print(f"     Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")


def generate_distribution_plots(all_stats: dict, df_clean: pd.DataFrame, all_action_names: list, output_dir: str):
    """
    Generates and saves distribution plots for A-vs-A pairs, 
    overlaying the A-vs-Else distribution and fitted models.
    """
    if plt is None or sns is None or GaussianMixture is None:
        print("\nCannot generate plots. matplotlib, seaborn, or sklearn not installed.")
        return
        
    print("\n--- Generating Distribution Plots (A-vs-A vs. A-vs-Else) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define actions to ignore for the "Else" category
    ignore_actions = {'BG', 'N/A'}
    
    for true_action in all_action_names:
        
        # 1. Get A-vs-A scores
        comparison_action = true_action
        
        if true_action not in all_stats or comparison_action not in all_stats[true_action]:
            print(f"  - Skipping plot for '{true_action}' (no stats found).")
            continue

        scores_A_vs_A = df_clean[df_clean['gt_actions'] == true_action][comparison_action].dropna()
        
        if scores_A_vs_A.empty:
            print(f"  - Skipping plot for '{true_action}' (no raw A-vs-A scores found).")
            continue
            
        stats_A_vs_A = all_stats[true_action][comparison_action]
        
        # 2. Get A-vs-Everything_Else scores
        else_action_names = [a for a in all_action_names if a != true_action and a not in ignore_actions]
        if not else_action_names:
            print(f"  - No 'Else' actions found for '{true_action}'. Skipping overlay.")
            all_else_scores = np.array([])
        else:
            # Get the DataFrame of all A-vs-Else scores
            scores_A_vs_Else_df = df_clean[df_clean['gt_actions'] == true_action][else_action_names]
            # Flatten to a single array and drop NaNs
            all_else_scores = scores_A_vs_Else_df.values.flatten()
            all_else_scores = all_else_scores[~np.isnan(all_else_scores)]

        plt.figure(figsize=(12, 7))
        
        # 3. Plot A-vs-A Data
        # Plot the histogram WITHOUT its own KDE
        sns.histplot(scores_A_vs_A, kde=False, stat="density", label=f"A-vs-A (Data Hist)", bins=30, color='lightblue', alpha=0.6)
        # Plot the "true" KDE line for A-vs-A separately
        sns.kdeplot(scores_A_vs_A, label="A-vs-A (True KDE)", color='blue', linestyle=':', lw=2.5)

        
        # 4. Plot the A-vs-Else KDE
        if all_else_scores.size > 0:
            sns.kdeplot(all_else_scores, label=f"A-vs-Else (True KDE)", color='purple', linestyle='--', lw=2.5)

        # 5. Plot the fitted Gamma PDF on top
        dist_params = stats_A_vs_A.get('dist_fit')
        if dist_params and dist_params['dist'] == 'gamma':
            try:
                a, loc, scale = dist_params['a'], dist_params['loc'], dist_params['scale']
                # Create an x-axis range based on all data
                min_x = max(0, min(scores_A_vs_A.min(), all_else_scores.min() if all_else_scores.size > 0 else 0))
                # Get a reasonable max for the x-axis (99th percentile)
                all_data_for_range = np.concatenate((scores_A_vs_A, all_else_scores))
                max_x = np.percentile(all_data_for_range, 99.5) if all_data_for_range.size > 0 else scores_A_vs_A.max()
                
                x = np.linspace(min_x, max_x, 200)
                
                pdf = gamma.pdf(x, a, loc=loc, scale=scale)
                plt.plot(x, pdf, 'r-', lw=2.5, label=f"Fitted Gamma (a={a:.2f}, loc={loc:.2f}, scale={scale:.2f})")
            except Exception as e:
                print(f"  - Warning: Could not plot fitted Gamma for '{true_action}': {e}")
        
        # 6. Plot the fitted GMM PDF on top
        gmm_fit = stats_A_vs_A.get('gmm_fit')
        if gmm_fit:
            try:
                # Use the same x-axis as the Gamma plot
                gmm_pdf = np.zeros_like(x)
                for i in range(gmm_fit['best_k']):
                    weight = gmm_fit['weights'][i]
                    mean = gmm_fit['means'][i][0]
                    std = np.sqrt(gmm_fit['covariances'][i][0][0])
                    gmm_pdf += weight * norm.pdf(x, mean, std)
                
                plt.plot(x, gmm_pdf, 'g--', lw=2.5, label=f"Fitted GMM (k={gmm_fit['best_k']})")
            except Exception as e:
                print(f"  - Warning: Could not plot GMM for '{true_action}': {e}")
        
        plt.title(f"DTW Score Distribution: '{true_action}'")
        plt.xlabel("DTW Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(left=0, right=max_x if 'max_x' in locals() else None) # Use the 99.5th percentile as the max limit
        
        # Save the figure
        plot_path = os.path.join(output_dir, f"{true_action}_dist_comparison.png")
        try:
            plt.savefig(plot_path)
            print(f"  ✅ Saved plot to {plot_path}")
        except Exception as e:
            print(f"  ❌ Error saving plot {plot_path}: {e}")
        finally:
            plt.close() # Free memory
    
    print("--- Plot generation complete ---")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze aggregated score CSVs to find score distributions."
    )
    parser.add_argument(
        '--scores_dir', 
        type=str, 
        required=True,
        help="Directory containing the 'wide' score CSVs generated by generate_segment_scores.py"
    )
    parser.add_argument(
        '--output_stats_json', 
        type=str, 
        required=False,
        help="Optional path to save the full statistics dictionary as a JSON file."
    )
    parser.add_argument(
        '--output_plot_dir', 
        type=str, 
        required=False,
        help="Optional path to a directory to save A-vs-A distribution plots."
    )
    args = parser.parse_args()

    # 1. Load and combine all data
    df, all_action_names = load_all_scores(args.scores_dir)
    
    if df.empty:
        return

    # 2. Calculate the distributions
    all_stats, df_clean = calculate_distributions(df, all_action_names)

    # 3. Print the summary to the console
    print_stats_summary(all_stats)
    
    # 4. Save to JSON if requested
    if args.output_stats_json:
        try:
            with open(args.output_stats_json, 'w') as f:
                json.dump(all_stats, f, indent=2, cls=NpEncoder)
            print(f"\n✅ Successfully saved full statistics to: {args.output_stats_json}")
        except Exception as e:
            print(f"\n❌ Error saving stats to JSON: {e}")

    if args.output_plot_dir:
        generate_distribution_plots(all_stats, df_clean, all_action_names, args.output_plot_dir)

if __name__ == "__main__":
    main()
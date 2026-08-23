import pandas as pd
import numpy as np
import os
import json
import argparse
import sys
from collections import defaultdict
from typing import Dict, Any, Set, List
from scipy.stats import weibull_min

# Try to import matplotlib & seaborn for the plot feature
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not found. Please run 'pip install matplotlib' to use the --plot_histogram feature.")
    plt = None

try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not found. Please run 'pip install seaborn' to use the --plot_histogram feature (for KDE line plots).")
    sns = None

DEFAULT_RECIPES = ['coffee', 'tea', 'quesadilla', 'oatmeal', 'pinwheels']


class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for numpy types (copied from analyze_scores.py) """
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

def load_training_list(training_video_list_path: str, recipe: str) -> Set[str]:
    """
    Loads the list of training video filenames from a .txt file.
    (Logic adapted from prototypes.py)
    """
    video_filenames = set()
    try:
        with open(training_video_list_path, 'r') as f:
            video_filenames = {line.strip() for line in f if line.strip() and recipe in line}
        print(f"Loaded {len(video_filenames)} training video names.")
    except FileNotFoundError:
        print(f"Error: Training video list not found at {training_video_list_path}")
    return video_filenames

def load_recipe_segments(annotation_path: str, recipe_name: str) -> List[Dict[str, Any]]:
    """
    Loads the main annotation file and returns the list of segments
    for the specified recipe.
    """
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found at {annotation_path}")
        return []
    
    try:
        with open(annotation_path, 'r') as f:
            all_metadata = json.load(f)
        
        if recipe_name not in all_metadata:
            print(f"Error: Recipe '{recipe_name}' not found in annotation file.")
            return []
            
        recipe_data = all_metadata[recipe_name].get('segments', [])
        print(f"Found {len(recipe_data)} total video segments for recipe '{recipe_name}'.")
        return recipe_data
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing annotation file: {e}")
        return []

def get_action_durations_for_recipe(
    recipe_segments: List[Dict[str, Any]], 
    training_set: Set[str], 
    target_action_name: str
) -> List[float]:
    """
    Returns a list of durations for *only* the target action
    from the given recipe segments and training set.
    """
    action_durations = []
    for video_meta in recipe_segments:
        video_id = video_meta.get('video_id')
        
        if not video_id or video_id not in training_set:
            continue
            
        try:
            timestamps = video_meta['labels']['time_stamp']
            action_names = video_meta['labels']['error_description']
            
            for action_name, time_pair in zip(action_names, timestamps):
                if action_name == target_action_name:
                    duration = time_pair[1] - time_pair[0]
                    if duration > 0:
                        action_durations.append(duration)
                        
        except KeyError:
            continue # Skip malformed video
            
    return action_durations


def generate_action_kde_plot(
    durations_by_action: Dict[str, List[float]], 
    recipe_name: str, 
    output_path: str
):
    """
    Generates and saves a single plot with a Kernel Density Estimate (KDE)
    line for each action in the recipe.
    """
    if plt is None or sns is None:
        print("Error: Cannot generate plot. matplotlib and/or seaborn are not installed.")
        return

    print(f"  -> Generating per-action KDE plot for '{recipe_name}'...")
    
    plt.figure(figsize=(12, 7))
    
    # Find a reasonable max range for the plot
    all_durations = [d for durations in durations_by_action.values() for d in durations]
    if not all_durations:
        print("  -> No durations to plot.")
        plt.close()
        return
        
    max_duration = np.percentile(all_durations, 99) # Use 99th percentile
    
    # Sort actions by name for a consistent legend
    sorted_actions = sorted(durations_by_action.keys())
    
    for action_name in sorted_actions:
        durations = durations_by_action[action_name]
        if not durations:
            continue
        
        # Plot the Kernel Density Estimate line
        sns.kdeplot(
            durations, 
            label=f"{action_name} (n={len(durations)})", 
            clip=(0, None) # Don't plot density for negative durations
        )

    plt.title(f'Distribution of Action Durations for Recipe: {recipe_name}')
    plt.xlabel('Action Duration (seconds)')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside plot
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max(1.0, max_duration)) # Set x-axis limit
    plt.tight_layout() # Adjust layout to fit legend
    
    try:
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plt.savefig(output_path)
        print(f"  ✅ Successfully saved KDE plot to: {output_path}")
    except Exception as e:
        print(f"  ❌ Error saving KDE plot: {e}")
    finally:
        plt.close() # Free memory


def generate_cross_recipe_kde_plot(
    durations_by_recipe: Dict[str, List[float]], 
    action_name: str, 
    output_path: str
):
    """
    (Cross-Recipe Mode Plot)
    Generates a plot with a KDE line for *each recipe*, all for the *same action*.
    """
    if plt is None or sns is None:
        print("Error: Cannot generate plot. matplotlib and/or seaborn are not installed.")
        return

    print(f"\nGenerating cross-recipe KDE plot for action: '{action_name}'...")
    
    plt.figure(figsize=(12, 7))
    
    all_durations = [d for durations in durations_by_recipe.values() for d in durations]
    if not all_durations:
        print("  -> No durations found for this action in any recipe. Cannot plot.")
        plt.close()
        return
        
    max_duration = np.percentile(all_durations, 99)
    
    for recipe_name, durations in sorted(durations_by_recipe.items()):
        if not durations:
            print(f"  - No data for recipe: {recipe_name}")
            continue
        
        sns.kdeplot(
            durations, 
            label=f"{recipe_name} (n={len(durations)})", 
            clip=(0, None)
        )

    plt.title(f'Distribution of "{action_name}" Duration Across Recipes')
    plt.xlabel('Action Duration (seconds)')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max(1.0, max_duration))
    plt.tight_layout()
    
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path)
        print(f"  ✅ Successfully saved cross-recipe KDE plot to: {output_path}")
    except Exception as e:
        print(f"  ❌ Error saving KDE plot: {e}")
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze action durations from an annotation file for a single recipe."
    )
    
    # --- Common arguments ---
    parser.add_argument(
        '--annotation_file', 
        type=str, 
        required=True,
        help="Path to the main annotation JSON file (e.g., unified_metadata.json)."
    )
    parser.add_argument(
        '--cross_recipe_mode',
        action='store_true',
        help="Activates cross-recipe comparison mode for a single action."
    )

   # --- Group 1: Stats Mode Arguments (default if --cross_recipe_mode is NOT set) ---
    stats_group = parser.add_argument_group('Stats & Per-Action Plot Mode (Default)')
    stats_group.add_argument(
        '--recipe', 
        type=str, 
        help="The specific recipe name to process."
    )
    stats_group.add_argument(
        '--training_video_list', 
        type=str, 
        help="Path to a .txt file containing training video filenames (for this recipe)."
    )
    stats_group.add_argument(
        '--output_stats_json', 
        type=str, 
        help="Optional: Path to save the duration statistics as a JSON file."
    )
    stats_group.add_argument(
        '--plot_histogram',
        action='store_true',
        help="Optional: Add this flag to generate a duration plot (KDE lines per action)."
    )

    # --- Group 2: Cross-Recipe Mode Arguments ---
    cross_recipe_group = parser.add_argument_group('Cross-Recipe Plot Mode')
    cross_recipe_group.add_argument(
        '--action_name', 
        type=str, 
        help="The specific action to compare across recipes."
    )
    cross_recipe_group.add_argument(
        '--training_list_dir', 
        type=str, 
        help="Directory containing training files, named [recipe_name]_train.txt."
    )
    cross_recipe_group.add_argument(
        '--output_plot_path', 
        type=str, 
        help="Path to save the output plot (REQUIRED for --plot_histogram or --cross_recipe_mode)."
    )
    cross_recipe_group.add_argument(
        '--recipes',
        nargs='+',
        default=DEFAULT_RECIPES,
        help="List of recipe names to compare in cross-recipe mode."
    )
    
    args = parser.parse_args()

    if args.cross_recipe_mode:
        # --- Cross-Recipe Mode ---
        if not all([args.action_name, args.training_list_dir, args.output_plot_path]):
            parser.error("--action_name, --training_list_dir, and --output_plot_path are required for --cross_recipe_mode.")
        if plt is None or sns is None:
            parser.error("matplotlib and seaborn are required for --cross_recipe_mode.")
        
        print(f"--- Running in Cross-Recipe Mode for Action: '{args.action_name}' ---")
        
        durations_by_recipe = {}
        
        # Load the *entire* annotation file *once*
        try:
            with open(args.annotation_file, 'r') as f:
                all_metadata = json.load(f)
        except Exception as e:
            print(f"Fatal Error: Could not load annotation file {args.annotation_file}. {e}")
            sys.exit(1)

        for recipe_name in args.recipes:
            print(f"\nProcessing recipe: {recipe_name}")
            
            # 1. Load Training List
            train_file_path = os.path.join(args.training_list_dir, recipe_name, f"test.txt")
            training_set = load_training_list(train_file_path, recipe_name)
            if not training_set:
                continue
                
            # 2. Get Recipe Segments (from already-loaded data)
            if recipe_name not in all_metadata:
                print(f"  - Warning: Recipe '{recipe_name}' not found in annotation file. Skipping.")
                continue
            recipe_segments = all_metadata[recipe_name].get('segments', [])
            if not recipe_segments:
                print(f"  - Warning: No segments found for recipe '{recipe_name}'. Skipping.")
                continue

            # 3. Get Durations for the target action
            action_durations = get_action_durations_for_recipe(
                recipe_segments, 
                training_set, 
                args.action_name
            )
            
            if action_durations:
                print(f"  -> Found {len(action_durations)} instances of '{args.action_name}'.")
                durations_by_recipe[recipe_name] = action_durations
            else:
                print(f"  -> No instances of '{args.action_name}' found in training videos.")

        # 4. Generate the final plot
        generate_cross_recipe_kde_plot(durations_by_recipe, args.action_name, args.output_plot_path)

    else:
        # --- Default Stats Mode ---
        if not args.recipe or not args.training_video_list:
            parser.error("--recipe and --training_video_list are required (in default stats mode).")
        if not args.output_stats_json and not args.plot_histogram:
            parser.error("No output requested. Please specify --output_stats_json and/or --plot_histogram.")
        if args.plot_histogram and not args.output_plot_path:
            parser.error("--output_plot_path is required when --plot_histogram is set.")
        if args.plot_histogram and (plt is None or sns is None):
            parser.error("matplotlib and seaborn are required for --plot_histogram.")

        print(f"--- Processing Recipe (Stats Mode): {args.recipe} ---")
        
        training_set = load_training_list(args.training_video_list, args.recipe)
        if not training_set: sys.exit(1)
        
        recipe_segments = load_recipe_segments(args.annotation_file, args.recipe)
        if not recipe_segments: sys.exit(1)

        print("Collecting durations from training videos...")
        durations_by_action = defaultdict(list)
        videos_processed_count = 0
        
        for video_meta in recipe_segments:
            video_id = video_meta.get('video_id')
            if not video_id or video_id not in training_set:
                continue
            videos_processed_count += 1
            try:
                timestamps = video_meta['labels']['time_stamp']
                action_names = video_meta['labels']['error_description']
                for action_name, time_pair in zip(action_names, timestamps):
                    # if action_name in ["N/A", "BG"]: continue
                    duration = time_pair[1] - time_pair[0]
                    if duration > 0:
                        durations_by_action[action_name].append(duration)
            except KeyError:
                print(f"Warning: Skipping video '{video_id}' due to malformed metadata.")
        
        print(f"Processed {videos_processed_count} training videos.")
        print(f"Found durations for {len(durations_by_action)} unique actions.")

        if args.output_stats_json:
            print("\nCalculating statistics...")
            all_stats = {}
            for action_name, durations_list in durations_by_action.items():
                if not durations_list: continue
                durations_np = np.array(durations_list)

                dist_params = None
                # Need at least 3-4 samples to get a decent fit
                if len(durations_np) > 3:
                    try:
                        # Fit a Weibull distribution
                        # floc=0 forces the distribution to start at 0 (durations can't be negative)
                        shape, loc, scale = weibull_min.fit(durations_np, floc=0)
                        dist_params = {
                            'dist': 'weibull_min',
                            'shape': shape,
                            'loc': loc,
                            'scale': scale
                        }
                        print(f"  - Action: '{action_name}' (count={len(durations_np)}) ... successfully fit Weibull dist.")
                    except Exception as e:
                        print(f"  - Action: '{action_name}' (count={len(durations_np)}) ... Weibull fit failed: {e}")
                else:
                     print(f"  - Action: '{action_name}' (count={len(durations_np)}) ... skipping distribution fit (not enough data).")


                stats = {
                    'mean_s': durations_np.mean(), 'std_s': durations_np.std(),
                    'min_s': durations_np.min(), 'max_s': durations_np.max(),
                    '25_percentile_s': np.percentile(durations_np, 25),
                    '50_percentile_s (median)': np.percentile(durations_np, 50),
                    '75_percentile_s': np.percentile(durations_np, 75),
                    'count': int(len(durations_np)),
                    'dist_fit': dist_params
                }
                all_stats[action_name] = stats
                print(f"  - Action: '{action_name}' (count={stats['count']})")
                print(f"    - Mean: {stats['mean_s']:.2f}s, Median: {stats['50_percentile_s (median)']:.2f}s")
            
            if all_stats:
                try:
                    output_dir = os.path.dirname(args.output_stats_json)
                    if output_dir: os.makedirs(output_dir, exist_ok=True)
                    with open(args.output_stats_json, 'w') as f:
                        json.dump(all_stats, f, indent=2, cls=NpEncoder)
                    print(f"\n✅ Successfully saved duration statistics to: {args.output_stats_json}")
                except Exception as e:
                    print(f"\n❌ Error saving stats to JSON: {e}")
            else:
                print("No statistics were generated.")

        if args.plot_histogram:
            print("\nGenerating KDE plot...")
            if not durations_by_action:
                print("No duration data found. Cannot generate plot.")
            else:
                generate_action_kde_plot(durations_by_action, args.recipe, args.output_plot_path)

    print("\n--- Done ---")

if __name__ == "__main__":
    main()
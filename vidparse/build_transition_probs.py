import json
import os
import argparse
import re
from collections import defaultdict
from abc import ABC, abstractmethod

# --- 1. Modular Parsers (Copied from task_graphs.py) ---

class BaseDatasetParser(ABC):
    """
    Abstract base class to standardize data extraction from different datasets.
    """
    def __init__(self, data_dir, task_list=None, training_video_list=None):
        self.data_dir = data_dir
        self.task_list = task_list
        self.training_video_ids = self._load_training_ids(training_video_list)
        
    def _load_training_ids(self, split_file):
        if not split_file:
            return None 
        try:
            with open(split_file, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            print(f"Warning: Split file {split_file} not found. Using all videos.")
            return None

    @abstractmethod
    def parse(self):
        """
        Returns:
            task_sequences: { 'task_name': [[id1, id2], ...] }
            task_id_map: { 'task_name': { id: 'description' } }
        """
        pass

class EgoPerParser(BaseDatasetParser):
    def parse(self):
        metadata_path = os.path.join(self.data_dir, 'annotations.json')
        if not os.path.exists(metadata_path):
             metadata_path = self.data_dir 
        
        print(f"Loading EgoPer metadata from {metadata_path}...")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading EgoPer metadata: {e}")
            return {}, {}

        task_sequences = defaultdict(list)
        task_id_map = {}
        
        tasks_to_process = self.task_list if self.task_list else metadata.keys()

        for task_name in tasks_to_process:
            if task_name not in metadata:
                continue
            
            recipe_data = metadata[task_name]
            
            # Build ID Map (action2idx)
            if 'action2idx' not in recipe_data:
                continue
            
            # Create {id: name} map, filtering BG
            id_map = {
                idx: name for name, idx in recipe_data['action2idx'].items()
                if name.upper() != 'BG' and idx != 0
            }
            task_id_map[task_name] = id_map
            valid_step_ids = set(id_map.keys())

            # Extract Sequences
            if 'segments' in recipe_data:
                for segment in recipe_data['segments']:
                    video_id = segment.get('video_id')
                    
                    # Filter by training split
                    if self.training_video_ids is not None:
                        found = False
                        for train_id in self.training_video_ids:
                            if video_id in train_id: 
                                found = True
                                break
                        if not found: continue

                    labels = segment.get('labels', {})
                    step_names = labels.get('error_description', [])
                    
                    sequence = []
                    for name in step_names:
                        idx = recipe_data['action2idx'].get(name)
                        if idx in valid_step_ids:
                            sequence.append(idx)
                    
                    if sequence:
                        task_sequences[task_name].append(sequence)

        return task_sequences, task_id_map

class FineGymParser(BaseDatasetParser):
    def __init__(self, data_dir, action_mapping_file, label_mapping_file, task_list=None, training_video_list=None):
        super().__init__(data_dir, task_list, training_video_list)
        self.action_mapping_file = action_mapping_file
        self.label_mapping_file = label_mapping_file

    def _load_label_map(self):
        id_to_desc = {}
        print(f"Loading FineGym labels from {self.label_mapping_file}...")
        pattern = re.compile(r"Clabel:\s+(\d+);.*;\s+(.*)")
        try:
            with open(self.label_mapping_file, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        id_to_desc[int(match.group(1))] = match.group(2).strip()
        except FileNotFoundError:
            print("Error: Label mapping file not found.")
        return id_to_desc

    def _load_action_mapping(self):
        key_to_id = {}
        print(f"Loading FineGym action mappings from {self.action_mapping_file}...")
        try:
            with open(self.action_mapping_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        key_to_id[parts[0]] = int(parts[-1]) 
        except FileNotFoundError:
            print("Error: Action mapping file not found.")
        return key_to_id

    def parse(self):
        global_id_to_desc = self._load_label_map()
        segment_key_to_id = self._load_action_mapping()
        
        json_path = os.path.join(self.data_dir, 'annotations.json')
        if not os.path.exists(json_path):
             json_path = self.data_dir
             
        print(f"Loading FineGym structure from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        task_sequences = defaultdict(list)
        task_id_map = defaultdict(lambda: global_id_to_desc) 

        for video_id, events in data.items():
            if self.training_video_ids and video_id not in self.training_video_ids:
                continue

            for event_key, event_data in events.items():
                event_type_id = str(event_data.get('event'))
                
                if self.task_list and event_type_id not in self.task_list:
                    continue

                segments = event_data.get('segments', {})
                if not segments:
                    continue

                sorted_seg_keys = sorted(
                    segments.keys(),
                    key=lambda k: segments[k]['timestamps'][0][0] if segments[k].get('timestamps') else 0
                )

                sequence = []
                for seg_key in sorted_seg_keys:
                    lookup_key = f"{video_id}_{event_key}_{seg_key}"
                    if lookup_key in segment_key_to_id:
                        sequence.append(segment_key_to_id[lookup_key])
                
                if sequence:
                    task_sequences[event_type_id].append(sequence)

        return task_sequences, task_id_map


# --- 2. Core Logic: Build Probabilities ---

def build_transition_probabilities_from_sequences(task_sequences, task_id_map):
    """
    Builds a transition probability matrix (as a nested dict) from standardized sequences.
    
    Output Format:
    {
        "task_name": {
            "from_action_name": {
                "to_action_name": 0.75,
                ...
            },
            ...
        }
    }
    """
    recipe_transition_probs = defaultdict(lambda: defaultdict(dict))
    
    for task_name, sequences in task_sequences.items():
        print(f"Processing {task_name} with {len(sequences)} sequences...")
        
        # Get ID->Name map for this task
        # Fallback to string ID if name missing
        id_map = task_id_map.get(task_name, {})
        
        # Stores counts: {from_action_name: {to_action_name: count}}
        counts = defaultdict(lambda: defaultdict(int))
        
        for seq in sequences:
            if len(seq) < 2:
                continue
            
            # Convert IDs to Names immediately for output consistency
            # (If you prefer outputting IDs, remove this conversion step)
            names_seq = [id_map.get(sid, str(sid)) for sid in seq]
            
            for i in range(len(names_seq) - 1):
                from_action = names_seq[i]
                to_action = names_seq[i+1]
                counts[from_action][to_action] += 1

        # Normalize counts
        for from_action, to_map in counts.items():
            total_transitions = sum(to_map.values())
            if total_transitions > 0:
                for to_action, count in to_map.items():
                    prob = count / total_transitions
                    recipe_transition_probs[task_name][from_action][to_action] = prob
                    
    return recipe_transition_probs


# --- 3. Main CLI ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Build action transition probabilities (Modular).")
    
    # Dataset Selection
    parser.add_argument('--dataset', type=str, required=True, choices=['egoper', 'finegym', 'gtea'], help="Dataset parser to use.")
    parser.add_argument('--data_source', type=str, required=True, help="Path to main JSON or root folder.")
    parser.add_argument('--train_list', type=str, help="Path to txt file with training video IDs.")
    parser.add_argument('--tasks', nargs='+', help="Specific tasks to process.")
    
    # Output
    parser.add_argument('--output_json', type=str, default='recipe_transition_probs.json', help="Output path.")
    
    # FineGym specific
    parser.add_argument('--finegym_action_map', type=str, help="Path to FineGym action mapping txt.")
    parser.add_argument('--finegym_label_map', type=str, help="Path to FineGym label mapping txt.")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 1. Select Parser
    parser = None
    if args.dataset in ['egoper', 'gtea']:
        parser = EgoPerParser(
            data_dir=args.data_source, 
            task_list=args.tasks, 
            training_video_list=args.train_list
        )
    elif args.dataset == 'finegym':
        if not args.finegym_action_map or not args.finegym_label_map:
            print("Error: FineGym requires --finegym_action_map and --finegym_label_map")
            return
        parser = FineGymParser(
            data_dir=args.data_source,
            action_mapping_file=args.finegym_action_map,
            label_mapping_file=args.finegym_label_map,
            task_list=args.tasks,
            training_video_list=args.train_list
        )
        
    # 2. Extract Sequences
    print("Parsing dataset...")
    sequences, id_map = parser.parse()
    
    if not sequences:
        print("No sequences found. Check paths and filters.")
        return

    # 3. Build Probabilities
    print("\nBuilding transition probabilities...")
    transition_probs = build_transition_probabilities_from_sequences(sequences, id_map)
    
    # 4. Save
    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(transition_probs, f, indent=4)
        print(f"\nSuccessfully saved probabilities to {args.output_json}")
        
        # Sample Print
        if transition_probs:
            sample_recipe = next(iter(transition_probs))
            if transition_probs[sample_recipe]:
                sample_from = next(iter(transition_probs[sample_recipe]))
                sample_entry = transition_probs[sample_recipe][sample_from]
                print(f"\nSample Entry [{sample_recipe}]: {sample_from} -> {json.dumps(sample_entry)}")
            
    except Exception as e:
        print(f"Error saving results: {e}") 

if __name__ == "__main__":
    main()
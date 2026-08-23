import json
import os
import argparse
import re
from collections import defaultdict
import networkx as nx
import pygraphviz as pgv
from abc import ABC, abstractmethod

# --- Abstract Base Parser ---
class BaseDatasetParser(ABC):
    """
    Abstract base class to standardize data extraction from different datasets.
    """
    def __init__(self, data_dir, task_list=None, training_video_list=None):
        self.data_dir = data_dir
        self.task_list = task_list  # Optional filter (e.g., ['tea', 'coffee'] or ['4', '1'])
        self.training_video_ids = self._load_training_ids(training_video_list)
        
    def _load_training_ids(self, split_file):
        if not split_file:
            return None # None means use all videos
        try:
            with open(split_file, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            print(f"Warning: Split file {split_file} not found. Using all videos.")
            return None

    @abstractmethod
    def parse(self):
        """
        Must return a tuple: (task_sequences, task_id_map)
        
        1. task_sequences: dict
           {
             'task_name_or_id': [ 
                 [step_id_1, step_id_2, ...], # Video 1 sequence
                 [step_id_1, step_id_3, ...], # Video 2 sequence
             ]
           }
           
        2. task_id_map: dict
           {
             'task_name_or_id': { step_id (int): 'step_description' (str) }
           }
        """
        pass

# --- EgoPer Implementation ---
class EgoPerParser(BaseDatasetParser):
    def parse(self):
        metadata_path = os.path.join(self.data_dir, 'annotations.json') # Assuming standard name
        if not os.path.exists(metadata_path):
             # Fallback if user passed direct json path in previous usage
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
        
        # Determine which tasks to process
        tasks_to_process = self.task_list if self.task_list else metadata.keys()

        for task_name in tasks_to_process:
            if task_name not in metadata:
                continue
            
            recipe_data = metadata[task_name]
            
            # 1. Build ID Map (action2idx)
            if 'action2idx' not in recipe_data:
                continue
            
            # Create {id: name} map (Inverting action2idx)
            # Filter BG (usually 0)
            id_map = {
                idx: name for name, idx in recipe_data['action2idx'].items()
                if name.upper() != 'BG' and idx != 0
            }
            task_id_map[task_name] = id_map
            valid_step_ids = set(id_map.keys())

            # 2. Extract Sequences
            if 'segments' in recipe_data:
                for segment in recipe_data['segments']:
                    video_id = segment.get('video_id')
                    
                    # Filter by training split if provided
                    if self.training_video_ids is not None:
                        # EgoPer check: is the specific video ID in the list?
                        if video_id not in self.training_video_ids:
                            # Note: Original code had logic to check if video line contained recipe name
                            # We keep it simple here: exact match or containment
                            found = False
                            for train_id in self.training_video_ids:
                                if video_id in train_id: 
                                    found = True
                                    break
                            if not found: continue

                    labels = segment.get('labels', {})
                    step_names = labels.get('error_description', [])
                    
                    # Convert names to IDs
                    sequence = []
                    for name in step_names:
                        idx = recipe_data['action2idx'].get(name)
                        if idx in valid_step_ids:
                            sequence.append(idx)
                    
                    if sequence:
                        task_sequences[task_name].append(sequence)

        return task_sequences, task_id_map


# --- FineGym Implementation ---
class FineGymParser(BaseDatasetParser):
    def __init__(self, data_dir, action_mapping_file, label_mapping_file, task_list=None, training_video_list=None):
        super().__init__(data_dir, task_list, training_video_list)
        self.action_mapping_file = action_mapping_file
        self.label_mapping_file = label_mapping_file

    def _load_label_map(self):
        """Parses the text file: Clabel: 0; set: 1; Glabel: 1; description..."""
        id_to_desc = {}
        print(f"Loading FineGym labels from {self.label_mapping_file}...")
        
        # Regex to capture ID and the final description
        # Matches: "Clabel:   0; ... ; (VT) description"
        pattern = re.compile(r"Clabel:\s+(\d+);.*;\s+(.*)")
        
        try:
            with open(self.label_mapping_file, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        c_label = int(match.group(1))
                        desc = match.group(2).strip()
                        id_to_desc[c_label] = desc
        except FileNotFoundError:
            print("Error: Label mapping file not found.")
        return id_to_desc

    def _load_action_mapping(self):
        """Parses file: Video_Event_Action <space> ID"""
        key_to_id = {}
        print(f"Loading FineGym action mappings from {self.action_mapping_file}...")
        try:
            with open(self.action_mapping_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        key = parts[0]
                        # The ID might be the last element
                        action_id = int(parts[-1]) 
                        key_to_id[key] = action_id
        except FileNotFoundError:
            print("Error: Action mapping file not found.")
        return key_to_id

    def parse(self):
        # 1. Load Mappings
        global_id_to_desc = self._load_label_map()
        segment_key_to_id = self._load_action_mapping()

        breakpoint()
        
        # 2. Load Main Annotations
        json_path = os.path.join(self.data_dir, 'annotations.json') # Adjust name if needed
        if not os.path.exists(json_path):
             # Try assuming data_dir IS the json file
             json_path = self.data_dir
             
        print(f"Loading FineGym structure from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        task_sequences = defaultdict(list)
        # In FineGym, the "Task" is the Event ID (e.g., 4). 
        # All event 4s share the same step vocabulary (the global list).
        task_id_map = defaultdict(lambda: global_id_to_desc) 

        # 3. Traverse Structure
        # Video -> Event -> Segments
        for video_id, events in data.items():
            
            # Check split
            if self.training_video_ids and video_id not in self.training_video_ids:
                continue

            for event_key, event_data in events.items():
                event_type_id = str(event_data.get('event')) # This is our "Task" (e.g., "4")
                
                # Filter if specific tasks requested
                if self.task_list and event_type_id not in self.task_list:
                    continue

                segments = event_data.get('segments', {})
                if not segments:
                    continue

                # 4. Sort segments by timestamp to ensure correct order
                # segments structure: { "A_...": { "timestamps": [[start, end]] } }
                # We sort by start time
                sorted_seg_keys = sorted(
                    segments.keys(),
                    key=lambda k: segments[k]['timestamps'][0][0] if segments[k].get('timestamps') else 0
                )

                # 5. Build Sequence
                sequence = []
                for seg_key in sorted_seg_keys:
                    # Construct the lookup key: Video_EventKey_SegmentKey
                    # Example: A0xAXXysHUo_E_002184_002237_A_0035_0036
                    lookup_key = f"{video_id}_{event_key}_{seg_key}"
                    
                    if lookup_key in segment_key_to_id:
                        action_id = segment_key_to_id[lookup_key]
                        sequence.append(action_id)
                    else:
                        # Optional: Log missing keys if debugging
                        pass
                
                if sequence:
                    task_sequences[event_type_id].append(sequence)

        return task_sequences, task_id_map


# --- Core Graph Logic (Dataset Agnostic) ---

def build_graphs_from_sequences(task_sequences):
    """
    Builds task graphs given a dictionary of sequences.
    Input: { 'task_name': [ [1, 2], [1, 3] ... ] }
    """
    
    # Stores (u, v) edges
    first_visit_edges = defaultdict(set) 
    revisit_edges = defaultdict(set)
    
    # Stores prerequisites lists: {task: {step_id: [ {prereqs_video1}, ... ]}}
    prereq_accumulator = defaultdict(lambda: defaultdict(list))
    step_presence_counts = defaultdict(lambda: defaultdict(int)) # Count videos containing step X
    
    # 1. First Pass: Aggregate sequence data
    for task_name, sequences in task_sequences.items():
        total_videos = len(sequences)
        
        for sequence in sequences:
            # Track unique steps in this video for omittable calc
            unique_steps_in_video = set(sequence)
            for s in unique_steps_in_video:
                step_presence_counts[task_name][s] += 1
            
            steps_seen_so_far = set()
            first_occurrence_prereqs = defaultdict(set)
            
            for i, current_step in enumerate(sequence):
                # Sequential Edges
                if i > 0:
                    previous_step = sequence[i-1]
                    edge = (previous_step, current_step)
                    
                    if current_step in steps_seen_so_far:
                        revisit_edges[task_name].add(edge)
                    else:
                        first_visit_edges[task_name].add(edge)
                
                # Prerequisite tracking (First occurrence only)
                if current_step not in first_occurrence_prereqs:
                    prereqs = steps_seen_so_far - {current_step}
                    first_occurrence_prereqs[current_step] = prereqs
                    
                steps_seen_so_far.add(current_step)
            
            # Aggregate prereqs
            for step_id, p_set in first_occurrence_prereqs.items():
                prereq_accumulator[task_name][step_id].append(p_set)

    # 2. Final Assembly (Minimal Prerequisites)
    final_graphs = {}
    
    for task_name in task_sequences.keys():
        step_graph = {}
        start_nodes = []
        
        # Get all steps appearing in this task
        all_steps = set()
        for seq in task_sequences[task_name]:
            all_steps.update(seq)
        
        total_video_count = len(task_sequences[task_name])

        for step_id in sorted(list(all_steps)):
            prereq_sets = prereq_accumulator[task_name][step_id]
            
            # Minimal Prereq Logic
            if not prereq_sets:
                list_of_minimal_prereqs = [[]]
            else:
                unique_sets = [s for i, s in enumerate(prereq_sets) if s not in prereq_sets[:i]]
                sorted_sets = sorted(unique_sets, key=len)
                
                minimal_sets = []
                for current_set in sorted_sets:
                    is_superset = False
                    for existing in minimal_sets:
                        if existing.issubset(current_set):
                            is_superset = True
                            break
                    if not is_superset:
                        minimal_sets.append(current_set)
                
                list_of_minimal_prereqs = [sorted(list(s)) for s in minimal_sets]
                if not list_of_minimal_prereqs: list_of_minimal_prereqs = [[]]

            # Omittable Logic
            count = step_presence_counts[task_name][step_id]
            is_omittable = (count < total_video_count) and (total_video_count > 0)

            step_graph[step_id] = {
                "prerequisites": list_of_minimal_prereqs,
                "is_omittable": is_omittable
            }

            if list_of_minimal_prereqs == [[]]:
                start_nodes.append(step_id)

            step_graph[step_id] = {
                "prerequisites": list_of_minimal_prereqs,
                "is_omittable": is_omittable
            }
        
        # Fully connect all start nodes to each other
        # We use revisit_edges so they appear as dashed/red in the viz
        # to distinguish inferred connections from observed sequences.
        for i in range(len(start_nodes)):
            for j in range(len(start_nodes)):
                if i != j:
                    u, v = start_nodes[i], start_nodes[j]
                    first_visit_edges[task_name].add((u, v))

        final_graphs[task_name] = {
            "first_visit_edges": sorted(list(first_visit_edges[task_name])),
            "revisit_edges": sorted(list(revisit_edges[task_name])),
            "dependency_graph": step_graph
        }
        
    return final_graphs


def visualize_graphs(task_graphs, task_id_map, output_dir):
    """
    Generates PNGs using Graphviz.
    """
    print("\nGenerating Visualizations...")
    
    for task_name, graph_data in task_graphs.items():
        id_map = task_id_map.get(task_name, {})
        
        G = nx.DiGraph()
        
        dep_graph = graph_data['dependency_graph']
        fv_edges = graph_data['first_visit_edges']
        rv_edges = graph_data['revisit_edges']
        
        all_edges = fv_edges + rv_edges
        all_source_nodes = {u for u, v in all_edges}
        
        # Add Nodes
        for node_id in dep_graph.keys():
            # Label
            node_str = id_map.get(node_id, str(node_id))
            label = f"{node_id}:{node_str}"
            wrapped = label.replace(" ", "\n").replace("_", "\n")
            
            # Style
            color = 'whitesmoke'
            node_info = dep_graph[node_id]
            
            if node_info['prerequisites'] == [[]]:
                color = 'palegreen' # Start
            elif node_id not in all_source_nodes:
                color = 'lightcoral' # End
            elif node_info.get('is_omittable'):
                color = 'lightyellow'
                
            G.add_node(node_id, label=wrapped, shape='oval', style='filled', fillcolor=color, fontname='Helvetica')

        # Add Edges
        for u, v in fv_edges:
            G.add_edge(u, v, color='black', style='solid')
        for u, v in rv_edges:
            G.add_edge(u, v, color='red', style='dashed')
            
        # Draw
        try:
            if G.number_of_nodes() > 0:
                A = nx.nx_agraph.to_agraph(G)
                A.graph_attr.update({'layout': 'dot', 'splines': 'true', 'label': f"Task: {task_name}"})
                out_path = os.path.join(output_dir, f"{task_name}_flow.png")
                A.draw(out_path, format='png', prog='dot')
                print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Viz Error for {task_name}: {e}")

# --- Main CLI ---

def parse_args():
    parser = argparse.ArgumentParser(description="Modular Task Graph Builder")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['egoper', 'finegym', 'gtea'], help="Which dataset parser to use.")
    parser.add_argument('--data_source', type=str, required=True, help="Path to main JSON (EgoPer) or root folder (FineGym).")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_list', type=str, help="Path to txt file with training video IDs.")
    
    # Task filtering (Optional)
    parser.add_argument('--tasks', nargs='+', help="Specific tasks/recipes to process (e.g. 'tea' 'coffee' or '4' '1').")
    
    # FineGym specific
    parser.add_argument('--finegym_action_map', type=str, help="Path to FineGym action mapping txt.")
    parser.add_argument('--finegym_label_map', type=str, help="Path to FineGym label mapping txt.")

    return parser.parse_args()

def main():
    args = parse_args()
    
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
        
    # 2. Extract Data (Standardized)
    print("Parsing dataset...")
    sequences, id_map = parser.parse()
    
    if not sequences:
        print("No sequences found. Check paths and filters.")
        return
        
    print(f"Found sequences for {len(sequences)} tasks.")

    # 3. Build Graphs
    print("Building graphs...")
    graphs = build_graphs_from_sequences(sequences)

    print("Injecting step names into graph data...")
    for task_name, graph_data in graphs.items():
        if task_name in id_map:
            mapping = id_map[task_name]
            # Check the dependency graph (nodes)
            for step_id, node_data in graph_data['dependency_graph'].items():
                # step_id might be int or str in dict keys, normalize if needed
                name = mapping.get(int(step_id), str(step_id))
                node_data['name'] = name
    
    # 4. Save JSON
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    json_path = os.path.join(args.output_dir, f"{args.dataset}_task_graphs.json")
    with open(json_path, 'w') as f:
        json.dump(graphs, f, indent=4)
    print(f"JSON saved to {json_path}")

    # 5. Visualize
    visualize_graphs(graphs, id_map, args.output_dir)

if __name__ == "__main__":
    main()
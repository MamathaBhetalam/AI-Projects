import pickle
import json
from FOON_class import Object

# Importing necessary libraries and modules
# FOON_class is a custom module being imported

# -----------------------------------------------------------------------------------------------------------------------------#

# Checks if an ingredient exists in the kitchen

def check_if_exist_in_kitchen(kitchen_items, ingredient):
    """
    parameters: a list of all kitchen items,
                an ingredient to be searched in the kitchen
    returns: True if ingredient exists in the kitchen
    """
    # Function to check if an ingredient exists in the kitchen

    for item in kitchen_items:
        if item["label"] == ingredient.label \
                and sorted(item["states"]) == sorted(ingredient.states) \
                and sorted(item["ingredients"]) == sorted(ingredient.ingredients) \
                and item["container"] == ingredient.container:
            return True

    return False

# -----------------------------------------------------------------------------------------------------------------------------#

# Function to perform a Greedy Best-First Search (GBFS) with a heuristic function

def search_GBFS(heuristic_func, kitchen_items=[], goal_node=None):
    """
    parameters: heuristic_func - a function to determine which candidate unit to select,
                kitchen_items - a list of all kitchen items,
                goal_node - the target node to achieve
    returns: a list of functional units representing the chosen path
    """
    # list of indices of functional units
    reference_task_tree = []

    # list of object indices that need to be searched
    items_to_search = []

    # find the index of the goal node in object node list
    items_to_search.append(goal_node.id)

    # list of items already explored
    items_already_searched = []

    while len(items_to_search) > 0:
        current_item_index = items_to_search.pop(0)  # pop the first element
        if current_item_index in items_already_searched:
            continue

        else:
            items_already_searched.append(current_item_index)

        current_item = foon_object_nodes[current_item_index]

        if not check_if_exist_in_kitchen(kitchen_items, current_item):

            candidate_units = foon_object_to_FU_map[current_item_index]

            # selecting the first path
            # this is the part where you should use heuristic for Greedy Best-First search
            selected_candidate_idx = heuristic_func(candidate_units)

            # if an fu is already taken, do not process it again
            if selected_candidate_idx in reference_task_tree:
                continue

            reference_task_tree.append(selected_candidate_idx)

            # all input of the selected FU need to be explored
            for node in foon_functional_units[selected_candidate_idx].input_nodes:
                node_idx = node.id
                if node_idx not in items_to_search:

                    # if in the input nodes, we have bowl contains {onion} and onion, chopped, in [bowl]
                    # explore only onion, chopped, in bowl
                    flag = True
                    if node.label in utensils and len(node.ingredients) == 1:
                        for node2 in foon_functional_units[selected_candidate_idx].input_nodes:
                            if node2.label == node.ingredients[0] and node2.container == node.label:
                                flag = False
                                break
                    if flag:
                        items_to_search.append(node_idx)

    # reverse the task tree
    reference_task_tree.reverse()

    # create a list of functional unit from the indices of reference_task_tree
    task_tree_units = []
    for i in reference_task_tree:
        task_tree_units.append(foon_functional_units[i])

    return task_tree_units

# The function `search_GBFS` implements a Greedy Best-First Search algorithm. It starts with a goal node and iteratively explores related nodes using a heuristic function to select the most promising candidate.

# -----------------------------------------------------------------------------------------------------------------------------#

# Heuristic function for Greedy Best-First Search (GBFS)

def greed_bfs_heuristic2(candidate_units):
    """
    parameters: candidate_units - a list of candidate units
    returns: the selected candidate unit based on a heuristic
    """
    c_unit = candidate_units[0]
    min_inputs = len(foon_functional_units[candidate_units[0]].input_nodes)

    for candidate_unit in candidate_units:
        inputs = len(foon_functional_units[candidate_unit].input_nodes)
        if inputs < min_inputs:
            c_unit = candidate_unit
            min_inputs = inputs

    return c_unit

# The function `greed_bfs_heuristic2` is a heuristic function used for Greedy Best-First Search. It selects the candidate unit with the fewest input nodes, which is a heuristic strategy for making decisions.

# -----------------------------------------------------------------------------------------------------------------------------#

# Function to save the generated task tree to a file

def save_paths_to_file(task_tree, path):
    """
    parameters: task_tree - a list of functional units representing the chosen path,
                path - the file path to save the task tree
    """
    print('writing generated task tree to ', path)
    _file = open(path, 'w')

    _file.write('//\n')
    for FU in task_tree:
        _file.write(FU.get_FU_as_text() + "\n")
    _file.close()

# The function `save_paths_to_file` saves the generated task tree to a file specified by the `path`.

# -----------------------------------------------------------------------------------------------------------------------------#

# Function to read a universal FOON from a pickle file

def read_universal_foon(filepath='FOON.pkl'):
    """
    parameters: filepath - path of universal FOON (pickle file)
    returns: a map. key = object, value = list of functional units
    """
    # Function to read a universal FOON from a pickle file

    pickle_data = pickle.load(open(filepath, 'rb'))
    functional_units = pickle_data["functional_units"]
    object_nodes = pickle_data["object_nodes"]
    object_to_FU_map = pickle_data["object_to_FU_map"]

    return functional_units, object_nodes, object_to_FU_map

# -----------------------------------------------------------------------------------------------------------------------------#

# Main section of the code

if __name__ == '__main__':
    foon_functional_units, foon_object_nodes, foon_object_to_FU_map = read_universal_foon()

    utensils = []
    with open('utensils.txt', 'r') as f:
        for line in f:
            utensils.append(line.rstrip())

    kitchen_items = json.load(open('kitchen.json'))

    goal_nodes = json.load(open("goal_nodes.json"))

    motion = {}
    with open('motion.txt', 'r') as f:
        for line in f:
            if "\t" in line:
                name, rate = line.strip().split("\t")
                motion[name] = rate

    for node in goal_nodes:
        node_object = Object(node["label"])
        node_object.states = node["states"]
        node_object.ingredients = node["ingredients"]
        node_object.container = node["container"]

        for object2 in foon_object_nodes:
            if object2.check_object_equal(node_object):
                output_task_tree = search_GBFS(greed_bfs_heuristic2, kitchen_items, object2)
                save_paths_to_file(output_task_tree,
                                   'output_GBFS_heuristic2_{}.txt'.format(node["label"]))
                break

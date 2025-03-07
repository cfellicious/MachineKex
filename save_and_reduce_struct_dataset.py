import networkx as nx
import pickle
from classes import HeapGraph
import json
import math
import re
from datasets import Dataset, concatenate_datasets

# How many hex-values should make up one token.
# Due to a bug in the tokenizers library, we have to use a Whitespace tokenizer.
# Thus, we preprocess the data here and insert a whitespace into a struct wherever the tokenizer is supposed to make a new token.
token_size = 4

do_train = False  # Determines whether the dataset is created from the train folder, or from the validation folder
do_save_dataset = False  # Save dataset to HuggingFaceHub. You need to logged on to huggingface cli or provide a token to do this.
do_reduce = False  # Reduce by setting size of class 0 to three times the size of the second largest class.

# Set desired dataset name to be uploaded to hub
dataset_name = f"johannes-garstenauer/structs_token_size_{token_size}_labelled_{'train' if do_train else 'eval'}_{'balanced' if do_reduce else ''}"

print(f"Do Train {do_train}, Do reduce {do_reduce}, Do save {do_save_dataset}")

# Replace with path to Heap Dumps
if do_train:
    path = "../Smart-VMI/data/new"
else:
    path = "../validation"


# Get all raw heap files and corresponding json files
def get_dataset_file_paths(path, deploy=False):
    import glob
    import os
    paths = []

    file_paths = []
    key_paths = []

    sub_dir = os.walk(path)
    for directory in sub_dir:
        paths.append(directory[0])

    paths = set(paths)
    for path in paths:
        # print(os.listdir(path))
        files = glob.glob(os.path.join(path, '*.raw'), recursive=False)

        if len(files) == 0:
            continue

        for file in files:
            key_file = file[:-9] + ".json"

            if os.path.exists(key_file) and deploy is False:

                file_paths.append(file)
                key_paths.append(key_file)

            elif deploy is True:
                file_paths.append(file)

            else:
                print("Corresponding Key file does not exist for :%s" % file)

    return file_paths, key_paths


def load_json(json_path):
    with open(json_path, 'r') as fp:
        info = json.load(fp)

    return info


def collect_relevant_structs_addresses_and_label_options(json_path):
    """
    Returns all addresses of structures mentioned in the json metadata file.
    """

    relevant_structs_addresses = []
    label_options = ['SSH_STRUCT_ADDR', 'SESSION_STATE_ADDR', 'NEWKEYS_1_ADDR', 'NEWKEYS_2_ADDR',
                     'ENCRYPTION_KEY_1_NAME_ADDR', 'ENCRYPTION_KEY_2_NAME_ADDR']

    info = load_json(json_path)

    # Append all relevant struct addresses
    for struct_name in label_options:
        addr = info.get(struct_name, None)
        relevant_structs_addresses.append(addr)

    # Append all existing keys
    ascii_A = 65
    ascii_F = 70
    for ascii_value in range(ascii_A, ascii_F + 1):
        key_name_base = "KEY_" + chr(ascii_value)
        key_name_addr = key_name_base + "_ADDR"

        label_options.append(key_name_addr)  # Add the key names to the label options

        key_len_name = key_name_base + '_LEN'
        key_real_len_name = key_name_base + '_REAL_LEN'

        # Make sure, that key length is greater than 0, before appending.
        if int(info.get(key_len_name, -1)) > 0 and int(info.get(key_real_len_name, -1)) > 0:
            addr = info.get(key_name_addr, None)
            relevant_structs_addresses.append(addr)
        else:
            relevant_structs_addresses.append(None)

    # Convert addresses to uppercase
    return [x.upper() if x is not None else None for x in relevant_structs_addresses], label_options


# Open the raw file and create graph structure from the pointers
# Adapted from ExtractNewKeys.ipynb (associated with PointerKex paper)
def load_and_clean_heap(heap_path, json_path, relevant_structs_addresses, label_options):
    with open(heap_path, 'rb') as fp:
        heap = bytearray(fp.read())

    info = load_json(json_path)

    # construct graph of openssh' heap
    base_addr = info.get('HEAP_START', '00000000')
    ssh_struct_addr = str(info.get('SSH_STRUCT_ADDR', None))

    if ssh_struct_addr is None or ssh_struct_addr == 'None':
        return None, None, None, None

    ssh_struct_addr = ssh_struct_addr.upper()

    heap_obj = HeapGraph(heap=heap, base_address=base_addr,
                         relevant_structs_addresses=relevant_structs_addresses,
                         label_options=label_options, ssh_struct_addr=ssh_struct_addr)
    heap_obj.create_graph()

    return heap_obj, heap_obj.heap_graph, ssh_struct_addr, info


# %%


# Extract all connected structures from the raw heap
def extract_structs_and_labels(heap_obj, heap_graph):
    structs = []
    labels = []
    if heap_obj is None or heap_obj.formatted_heap is None:
        return structs, labels

    formatted_heap = heap_obj.formatted_heap

    # Assuming that the graph contains all the relevant struct addresses.
    for node in list(nx.nodes(heap_graph)):

        node_info = heap_graph.nodes.get(node)
        size = node_info.get('size', 0)  # Allocated chunk (struct) size in bytes.

        # Extract Sequence From Heap
        address = heap_obj.resolve_pointer_address(node)
        offset = int(address / 8)  # Get the 8-byte aligned offset for formatted heap

        # Append all formatted heap slices that contain at least one bit of the struct.
        struct = ""
        for i in range(math.ceil(size / 8)):
            struct = struct + formatted_heap[offset + i]
        structs.append(struct)

        # Find and insert label
        label = node_info.get('label', '-1')
        labels.append(label)

    return structs, labels


structs = []
labels = []
heap_paths, key_paths = get_dataset_file_paths(path)
assert len(heap_paths) == len(key_paths)
print(f"File paths collected, amount: {len(heap_paths)}")

# Process all heaps. Extract structures by using heap graph.
invalid_paths_set = set()
for i in range(len(heap_paths)):
    try:
        relevant_structs_addresses, label_options = collect_relevant_structs_addresses_and_label_options(
            json_path=key_paths[i])

        heap_obj, heap_graph, ssh_struct_addr, info = load_and_clean_heap(heap_path=heap_paths[i],
                                                                          json_path=key_paths[i],
                                                                          relevant_structs_addresses=relevant_structs_addresses,
                                                                          label_options=label_options)
    except Exception as ex:

        import os

        invalid_paths_set.add(os.path.dirname(heap_paths[i]))
    else:
        new_structs, new_labels = extract_structs_and_labels(heap_obj, heap_graph)
        structs.extend(new_structs)
        labels.extend(new_labels)

    if i % 500 == 0:
        print(f"Progress: {i} files scanned")
print(f"Structs: {len(structs)}")
print(f"Invalid paths: {invalid_paths_set}")


# Workaround for non-functional Split() pre-tokenizer. Insert a WhiteSpace where a sequence should be split,
# so that the WhiteSpace pre-tokenizer may actually split it there.
def whitespace_tokenization(token_length, sequence):
    pattern = '.{1,' + str(token_length) + '}'
    tokens = re.sub(pattern, lambda m: m.group(0) + " ",
                    sequence).strip()  # Replace a 4 long subsequence with itself and an added whitespace

    return tokens


# White space tokenize
print("White space tokenize start")
structs = [whitespace_tokenization(token_size, struct) for struct in structs]
print("White space tokenize done")


# Create dict from structs-list
structs_dict = {'struct': structs, 'label': labels}

# Convert dict into dataset
dataset = Dataset.from_dict(structs_dict)


# Map label names to integer value.
def label_to_int(label):
    if label == "SSH_STRUCT_ADDR" or label == "0":
        return 0
    elif label == "SESSION_STATE_ADDR":
        return 1
    elif label == "NEWKEYS_1_ADDR" or label == "NEWKEYS_2_ADDR":
        return 2
    elif label == "ENCRYPTION_KEY_1_NAME_ADDR" or label == "ENCRYPTION_KEY_2_NAME_ADDR":
        return 3
    elif label == "KEY_A_ADDR" or label == "KEY_B_ADDR" or label == "KEY_C_ADDR" or label == "KEY_D_ADDR" or label == "KEY_E_ADDR" or label == "KEY_F_ADDR":
        return 4
    else:
        raise ValueError(f"{label} is an illegal value for a label!")


# Process label names to integer values
def preprocess_function(examples):
    examples.data["label"] = [label_to_int(label) for label in examples["label"]]
    return examples


final_set = dataset.map(preprocess_function, batched=True, num_proc=8)

# Reduce class 0 samples.
if do_reduce:
    # Extracting examples of the separate labels
    label_0_examples = final_set.filter(lambda example: example['label'] == 0)  # , batched=True, num_proc=4)#.shuffle()
    label_1_examples = final_set.filter(lambda example: example['label'] == 1)  # , batched=True, num_proc=4)
    label_2_examples = final_set.filter(lambda example: example['label'] == 2)  # , batched=True, num_proc=4)
    label_3_examples = final_set.filter(lambda example: example['label'] == 3)  # , batched=True, num_proc=4)
    label_4_examples = final_set.filter(lambda example: example['label'] == 4)  # , batched=True, num_proc=4)

    # Find a suitable majority class dataset size
    largest_minority_class_size = max(len(label_1_examples), len(label_2_examples), len(label_3_examples),
                                      len(label_4_examples))
    majority_class_representation_factor = 3
    desired_majority_class_size = largest_minority_class_size * majority_class_representation_factor

    # Randomly selecting a subset of required size
    balanced_subset_class_0 = Dataset.from_dict(label_0_examples[:desired_majority_class_size])

    # Create the combined balanced subset
    balanced_subset = concatenate_datasets(
        [balanced_subset_class_0, label_1_examples, label_2_examples, label_3_examples,
         label_4_examples])

    final_set = balanced_subset.shuffle(seed=42)
    with open('validation_set.pkl', 'wb') as fp:
        pickle.dump(fp, final_set)
        print('Saved')
# Upload dataset
if do_save_dataset:
    final_set.push_to_hub(dataset_name)

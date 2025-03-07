from utils.utils import get_dataset_file_paths


def get_file_paths():
    train_root = '../new'
    test_root = '../validation'
    heap_paths, json_paths = get_dataset_file_paths(train_root)
    test_heap_paths, test_json_paths = get_dataset_file_paths(test_root)
    return heap_paths, json_paths, test_heap_paths, test_json_paths

def get_keys(info):
    const_keys = ['KEY_A', 'KEY_B', 'KEY_C', 'KEY_D', 'KEY_E', 'KEY_F']
    keys = []
    for key in const_keys:
        curr_key = info.get(key, None)
        if curr_key is None:
            continue
        if len(curr_key) > 12:
            keys.append(curr_key.upper())
    return keys
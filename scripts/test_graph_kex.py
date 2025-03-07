import json

from tqdm import tqdm
from classes.graphkex import GraphKex, TestGraphKex
from utils.utils import generate_dataset, train_and_test
from scripts.helper import get_file_paths, get_keys


def main():

    heap_paths, json_paths, test_heap_paths, test_json_paths = get_file_paths()

    dataset, labels = generate_dataset(heap_paths=heap_paths, json_paths=json_paths, obj_class=GraphKex,
                                       block_size=10000, feature_vector_size=9)
    test_dataset, test_labels = generate_dataset(heap_paths=test_heap_paths, json_paths=test_json_paths,
                                                 obj_class=GraphKex, block_size=20000, feature_vector_size=9)

    clf = train_and_test(dataset=dataset, labels=labels, test_dataset=test_dataset, test_labels=test_labels,
                         model_name='../models/graph_kex_model.pkl')

    total_keys = 0
    found_keys = 0
    total_files = 0
    for test_heap_path, test_json_path in tqdm(zip(test_heap_paths, test_json_paths)):

        with open(test_json_path, 'r') as fp:
            info = json.load(fp)
        test_obj = TestGraphKex(heap_path=test_heap_path, clf=clf, base_address=info.get('HEAP_START', '0x0000'))
        predicted_keys = test_obj.predict_keys()
        true_keys = get_keys(info=info)

        total_keys += len(true_keys)

        for true_key in true_keys:
            pred_true = [x for x in predicted_keys if true_key in x]
            if len(pred_true) > 0:
                found_keys += 1
        total_files += 1

    print('Found %d/%d of files: %d' % (found_keys, total_keys, total_files))


if __name__ == '__main__':
    main()

import json

from tqdm import tqdm
from classes.pointerkex import PointerKex
from utils.utils import generate_dataset, train_and_test
from scripts.helper import get_file_paths, get_keys


def main():

    heap_paths, json_paths, test_heap_paths, test_json_paths = get_file_paths()

    dataset, labels = generate_dataset(heap_paths=heap_paths, json_paths=json_paths, obj_class=PointerKex,
                                       block_size=10000, feature_vector_size=5)
    test_dataset, test_labels = generate_dataset(heap_paths=test_heap_paths, json_paths=test_json_paths,
                                                 obj_class=PointerKex, block_size=20000, feature_vector_size=5)

    clf = train_and_test(dataset=dataset, labels=labels, test_dataset=test_dataset, test_labels=test_labels,
                         model_name='../models/graph_kex_model.pkl')


if __name__ == '__main__':
    main()

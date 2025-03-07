from utils.utils import generate_dataset, train_and_test
from scripts.helper import get_file_paths
from classes.metakex import MetaKex


def main():
    heap_paths, json_paths, test_heap_paths, test_json_paths = get_file_paths()
    # 176 is the feature vector size as it is 128 for the memory chunk + 16 for the sum(8-byte words) + 16 for the ascii
    # + 16 for the non zeros
    dataset, labels = generate_dataset(heap_paths=heap_paths, json_paths=json_paths, obj_class=MetaKex,
                                       block_size=10000, feature_vector_size=176)
    test_dataset, test_labels = generate_dataset(heap_paths=test_heap_paths, json_paths=test_json_paths,
                                                 obj_class=MetaKex, block_size=20000, feature_vector_size=176)

    clf = train_and_test(dataset=dataset, labels=labels, test_dataset=test_dataset, test_labels=test_labels,
                         model_name='../models/metakex_model.pkl')


if __name__ == '__main__':
    main()

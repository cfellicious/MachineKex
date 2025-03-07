from utils.utils import generate_dataset, train_and_test
from scripts.helper import get_file_paths
from classes.smartkex import SmartKex


def main():
    heap_paths, json_paths, test_heap_paths, test_json_paths = get_file_paths()

    dataset, labels = generate_dataset(heap_paths=heap_paths, json_paths=json_paths, obj_class=SmartKex,
                                       block_size=5000, feature_vector_size=128)
    test_dataset, test_labels = generate_dataset(heap_paths=test_heap_paths, json_paths=test_json_paths,
                                                 obj_class=SmartKex, block_size=1000, feature_vector_size=128)

    clf = train_and_test(dataset=dataset, labels=labels, test_dataset=test_dataset, test_labels=test_labels,
                         model_name='../models/smart_kex_model.pkl')


if __name__ == '__main__':
    main()

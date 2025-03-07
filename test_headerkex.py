from utils.utils import generate_dataset, train_and_test
from scripts.helper import get_file_paths
from classes.header_kex import HeaderKex


def main():
    heap_paths, json_paths, test_heap_paths, test_json_paths = get_file_paths()

    dataset, labels = generate_dataset(heap_paths=heap_paths, json_paths=json_paths, obj_class=HeaderKex,
                                       block_size=10000, feature_vector_size=136)
    test_dataset, test_labels = generate_dataset(heap_paths=test_heap_paths, json_paths=test_json_paths,
                                                 obj_class=HeaderKex, block_size=16000, feature_vector_size=136)

    clf = train_and_test(dataset=dataset, labels=labels, test_dataset=test_dataset, test_labels=test_labels,
                         model_name='./models/header_kex_model.pkl')


if __name__ == '__main__':
    main()

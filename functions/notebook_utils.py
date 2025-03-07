from functions.utils import generate_dataset
from classes.slicedkex import SlicedKex
from classes.smartkex import SmartKex
from classes.metakex import MetaKex
from classes.headerkex import HeaderKex
from classes.graphkex import GraphKex
from classes.pointerkex import PointerKex

from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np


def generate_all_data(heap_paths, json_paths, classes, algorithms, block_size=1000, train_subset=True):
    test_dataset_dict = dict()

    for idx, curr_class in enumerate(classes):
        test_dataset, test_labels = generate_dataset(heap_paths=heap_paths,
                                                     json_paths=json_paths,
                                                     obj_class=curr_class, block_size=block_size,
                                                     train_subset=train_subset, feature_vector_size=128)
        test_dataset_dict[algorithms[idx]] = [deepcopy(test_dataset), deepcopy(test_labels)]


    test_dataset, test_labels = generate_dataset(heap_paths=heap_paths,
                                                 json_paths=json_paths,
                                                 obj_class=SlicedKex, block_size=block_size,
                                                 train_subset=train_subset, feature_vector_size=128)
    test_dataset_dict[algorithms[0]] = [deepcopy(test_dataset), deepcopy(test_labels)]

    """
    test_dataset, test_labels = generate_dataset(heap_paths=heap_paths,
                                                 json_paths=json_paths,
                                                 obj_class=SmartKex, block_size=block_size,
                                                 train_subset=train_subset, feature_vector_size=128)
    test_dataset_dict[algorithms[1]] = [deepcopy(test_dataset), deepcopy(test_labels)]

    test_dataset, test_labels = generate_dataset(heap_paths=heap_paths,
                                                 json_paths=json_paths,
                                                 obj_class=MetaKex, block_size=block_size,
                                                 train_subset=train_subset, feature_vector_size=176)
    test_dataset_dict[algorithms[2]] = [deepcopy(test_dataset), deepcopy(test_labels)]

    test_dataset, test_labels = generate_dataset(heap_paths=heap_paths,
                                                 json_paths=json_paths,
                                                 obj_class=HeaderKex, block_size=block_size,
                                                 train_subset=train_subset, feature_vector_size=136)
    test_dataset_dict[algorithms[3]] = [deepcopy(test_dataset), deepcopy(test_labels)]

    test_dataset, test_labels = generate_dataset(heap_paths=heap_paths,
                                                 json_paths=json_paths,
                                                 obj_class=GraphKex, block_size=block_size,
                                                 train_subset=train_subset, feature_vector_size=9)
    test_dataset_dict[algorithms[4]] = [test_dataset, test_labels]

    test_dataset, test_labels = generate_dataset(heap_paths=heap_paths,
                                                 json_paths=json_paths,
                                                 obj_class=PointerKex, block_size=block_size,
                                                 train_subset=train_subset, feature_vector_size=5)

    test_dataset_dict[algorithms[5]] = [test_dataset, test_labels]
    """
    return test_dataset_dict


def train_all_algorithms(dataset_dict, seed, algorithms, test_dataset_dict, results_dict):

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    print('Current Seed: %d' % seed)
    for idx, algorithm in enumerate(algorithms):

        clf = RandomForestClassifier(random_state=seed)
        dataset, labels = dataset_dict.get(algorithm)
        if len(labels) == 0:
            continue
        clf.fit(dataset, labels)

        test_dataset, test_labels = test_dataset_dict.get(algorithm)
        if len(test_labels) == 0:
            algorithm_result = results_dict.get(algorithm, dict())

            accuracy_list = algorithm_result.get(metrics[0], [])
            accuracy_list.append(0.0)
            algorithm_result[metrics[0]] = accuracy_list

            precision_list = algorithm_result.get(metrics[1], [])
            precision_list.append(0.0)
            algorithm_result[metrics[1]] = precision_list

            recall_list = algorithm_result.get(metrics[2], [])
            recall_list.append(0.0)
            algorithm_result[metrics[2]] = recall_list

            f1_list = algorithm_result.get(metrics[3], [])
            f1_list.append(0.0)
            algorithm_result[metrics[3]] = f1_list
            continue
        y_pred = clf.predict(test_dataset)
        precision = precision_score(y_true=test_labels, y_pred=y_pred)
        recall = recall_score(y_true=test_labels, y_pred=y_pred)
        f1_measure = f1_score(y_true=test_labels, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_labels, y_pred=y_pred)

        print('F1-Measure for %s for dataset: %f' % (algorithm, f1_measure))

        algorithm_result = results_dict.get(algorithm, dict())

        accuracy_list = algorithm_result.get(metrics[0], [])
        accuracy_list.append(accuracy)
        algorithm_result[metrics[0]] = accuracy_list

        precision_list = algorithm_result.get(metrics[1], [])
        precision_list.append(precision)
        algorithm_result[metrics[1]] = precision_list

        recall_list = algorithm_result.get(metrics[2], [])
        recall_list.append(recall)
        algorithm_result[metrics[2]] = recall_list

        f1_list = algorithm_result.get(metrics[3], [])
        f1_list.append(f1_measure)
        algorithm_result[metrics[3]] = f1_list

        results_dict[algorithm] = algorithm_result

    return results_dict


def merge_training_data_dicts(whole_dataset, curr_dataset):
    key = 'No Key Selected'
    try:
        for key, curr_dataset in curr_dataset.items():
            dataset, labels = whole_dataset.get(key, [None, None])
            if dataset is not None:
                dataset = np.vstack((dataset, curr_dataset[0]))
                labels = labels + curr_dataset[1]

            else:
                dataset = curr_dataset[0]
                labels = curr_dataset[1]

            whole_dataset[key] = [dataset, labels]
    except:
        print(key, curr_dataset)

    return whole_dataset

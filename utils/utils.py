import numpy as np
import pickle as pkl

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


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


def generate_dataset(heap_paths, json_paths, obj_class, train_subset=True, block_size=100, feature_vector_size=9):
    limit = len(heap_paths)
    if train_subset is True:
        limit = min(block_size, limit)

    total_files_found = 0
    total_size = 0

    dataset_dict = dict()
    labels_dict = dict()

    for idx in tqdm(range(limit), desc='Data Files'):

        # Read the raw heap and json information and create the graph
        heap_obj = obj_class(heap_path=heap_paths[idx], json_path=json_paths[idx])
        features, curr_labels = heap_obj.generate_data()
        if len(features) == 0:
            continue
        curr_labels = np.array(curr_labels)

        dataset_dict[idx] = features
        labels_dict[idx] = curr_labels
        total_size += len(curr_labels)

        total_files_found += 1

    curr_index = 0
    end_index = 0
    dataset = np.empty(shape=(total_size, feature_vector_size), dtype=int)
    labels = np.zeros(total_size, dtype=int)
    for key, item in dataset_dict.items():
        end_index += len(item)
        dataset[curr_index:end_index] = item
        labels[curr_index:end_index] = labels_dict[key]
        curr_index = end_index

    print('Total files found: %d' % total_files_found)
    return dataset.tolist(), labels.tolist()


def train_and_test(dataset, labels, test_dataset, test_labels, model_name='model.pkl'):
    clf = RandomForestClassifier(n_estimators=7)
    clf.fit(X=dataset, y=labels)
    with open(model_name, 'wb') as fp:
        pkl.dump(clf, fp)

    y_pred = clf.predict(test_dataset)
    precision = precision_score(y_true=test_labels, y_pred=y_pred)
    recall = recall_score(y_true=test_labels, y_pred=y_pred)
    f1_measure = f1_score(y_true=test_labels, y_pred=y_pred)
    accuracy = accuracy_score(y_true=test_labels, y_pred=y_pred)
    print('Accuracy: %f\nPrecision: %f\nRecall:%f\nF1-Measure:%f' % (accuracy, precision, recall, f1_measure))

    return clf

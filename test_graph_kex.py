import json
import pickle as pkl

from tqdm import tqdm
from classes.graphkex import GraphKex, TestGraphKex
from utils.utils import get_dataset_file_paths, generate_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


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

def main():

    train_root = '../new'
    test_root = '../validation'
    heap_paths, json_paths = get_dataset_file_paths(train_root)
    test_heap_paths, test_json_paths = get_dataset_file_paths(test_root)

    dataset, labels = generate_dataset(heap_paths=heap_paths, json_paths=json_paths, obj_class=GraphKex,
                                       block_size=100000)
    clf = RandomForestClassifier(n_estimators=7)
    clf.fit(X=dataset, y=labels)
    with open('model.pkl', 'wb') as fp:
        pkl.dump(clf, fp)

    test_dataset, test_labels = generate_dataset(heap_paths=test_heap_paths, json_paths=test_json_paths,
                                                 obj_class=GraphKex, block_size=20000)
    y_pred = clf.predict(test_dataset)
    precision = precision_score(y_true=test_labels, y_pred=y_pred)
    recall = recall_score(y_true=test_labels, y_pred=y_pred)
    f1_measure = f1_score(y_true=test_labels, y_pred=y_pred)
    accuracy = accuracy_score(y_true=test_labels, y_pred=y_pred)
    print('Accuracy: %f\nPrecision: %f\nRecall:%f\nF1-Measure:%f' % (accuracy, precision, recall, f1_measure))
    
    total_keys = 0
    found_keys = 0
    total_files = 0
    for test_heap_path, test_json_path in tqdm(zip(test_heap_paths, test_json_paths)):
        # print('Testing heap: %s\nTesting JSON: %s' % (test_heap_path, test_json_path))
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

    print('Found %d/%d from %d files' % (found_keys, total_keys, total_files))


if __name__ == '__main__':
    main()


import numpy as np

from classes.kex import Kex


class SlicedKex(Kex):

    max_key_length = 0
    window_size = 0

    def __init__(self, heap_path, json_path, window_size=128, max_key_length=64):
        super().__init__(heap_path=heap_path, json_path=json_path)
        self.max_key_length = max_key_length
        self.window_size = window_size

    def convert_keys(self):
        keys = []
        for key in self.keys:
            keys.append(bytearray.fromhex(key))
        self.keys = keys

    def generate_data(self):

        self.read_data()
        self.convert_keys()
        idx = 0
        labels = []
        dataset = []
        while idx + self.window_size < self.heap_size:

            curr_data = self.heap[idx:idx+128]
            idx += self.max_key_length
            if sum(curr_data) == 0:
                continue

            found = [l_idx if self.keys[l_idx] in curr_data
                     else 0 for l_idx in range(len(self.keys))]

            dataset.append(curr_data)
            if any(found) is True:
                labels.append(1)
            else:
                labels.append(0)

        return dataset, labels


class TestSlicedKex(SlicedKex):
    clf = None

    def __init__(self, heap_path, clf, window_size=128, max_key_length=64):

        super().__init__(heap_path=heap_path, json_path=None, window_size=window_size, max_key_length=max_key_length)
        self.clf = clf
        self.read_data()

    def generate_data(self):
        dataset = []
        self.read_data()
        idx = 0
        while idx + 128 < self.heap_size:

            curr_data = self.heap[idx:idx + 128]
            if sum(curr_data) == 0:
                continue
            dataset.append(curr_data)
        return np.array(dataset)

    def predict_keys(self):

        dataset = self.generate_data()
        predictions = dataset[self.clf.predict(dataset).nonzero()[0]]
        return predictions




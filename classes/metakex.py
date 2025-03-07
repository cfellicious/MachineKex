import numpy as np
from classes.smartkex import SmartKex


class MetaKex(SmartKex):

    concatenate = True

    def __init__(self, heap_path, json_path, max_key_size=64, window_size=128, concatenate=True):
        super().__init__(heap_path=heap_path, json_path=json_path, max_key_size=max_key_size, window_size=window_size)
        self.concatenate = concatenate

    def generate_data(self):
        dataset, labels = super().generate_data()
        dataset = np.array(dataset, dtype=np.uint8)
        dataset_reshaped = np.reshape(np.array(dataset, dtype=np.uint), (dataset.shape[0], 16, 8))
        non_zero_counts = np.count_nonzero(dataset_reshaped, axis=2)
        slice_sum = np.sum(np.log10(dataset_reshaped.astype(int) + 1).astype(np.uint8), axis=2)
        ascii_values = np.sum(((dataset_reshaped < 127) & (dataset_reshaped > 32)).astype(int), axis=2).astype(np.uint8)

        if self.concatenate is True:
            dataset = np.hstack((dataset, non_zero_counts, ascii_values, slice_sum))

        else:
            dataset = np.hstack((dataset, ascii_values, slice_sum))

        return dataset, labels



import numpy as np

from classes.smartkex import SmartKex


class HeaderKex(SmartKex):

    def __init__(self, heap_path, json_path, max_key_size=64, window_size=136):
        super().__init__(heap_path=heap_path, json_path=json_path, max_key_size=64, window_size=window_size)

    def generate_data(self):

        dataset = []
        labels = []

        characters, counts, cum_sum = self.get_possible_key_locations()

        # c = [counts[idx] for idx in range(len(counts)) if characters[idx] == 1]
        # print(max(c))

        viable_offsets = [cum_sum[idx] for idx in range(len(cum_sum)) if characters[idx] == 1]
        # Modify the offsets so that we get the header information as well
        viable_offsets = np.subtract(viable_offsets, 8)
        viable_offsets[viable_offsets < 0] = 0
        slices, curr_labels = self.get_slices(offsets=viable_offsets)

        dataset = dataset + slices
        labels = labels + curr_labels

        return dataset, labels

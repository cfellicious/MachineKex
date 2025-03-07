import numpy as np
from classes.kex import Kex


class SmartKex(Kex):
    max_key_size = 0
    window_size = 128

    def __init__(self, heap_path, json_path, max_key_size=64, window_size=128):
        super().__init__(heap_path=heap_path, json_path=json_path)
        self.max_key_size = max_key_size
        self.window_size = window_size

    def convert_keys(self):
        keys = []
        for key in self.keys:
            keys.append(bytearray.fromhex(key))
        self.keys = keys

    def get_possible_key_locations(self):
        self.read_data()
        self.convert_keys()

        # Here we take reshaped array and compute the numerical row and column wise gradient and count the number
        # of zeroes in each row. If there are more than 4 zeros which means there is a pattern repeating and
        # is not a key. This is a very conservative estimate for better recall
        # x_grad = np.abs(np.diff(reshaped.astype(int), axis=1, append=np.zeros((num_row, 1)))).astype(bool)
        # y_grad = np.abs(np.diff(reshaped.astype(int), axis=0, append=np.zeros((1, 8)))).astype(bool)
        # The above numerical gradient computation is transformed into a single step below
        aligned_size = int(self.heap_size/8)
        aligned_heap = np.reshape(self.heap, (aligned_size, 8)).astype(int)

        poss_key_locs = (np.count_nonzero(
            np.logical_and(
                np.abs(np.diff(aligned_heap, axis=1, append=np.zeros((aligned_size, 1)))).astype(bool),
                np.abs(np.diff(aligned_heap, axis=0, append=np.zeros((1, 8))).astype(bool))), axis=1) >= 4).astype(int)

        # This part addresses the issue of 12 byte keys. There could be two identical characters next to each other in
        # the last 4 bytes which would make it impossible for a key loc.
        # We modify that if there is a possibility for a key
        idx = 1
        while idx < len(poss_key_locs):
            # Last 4 characters must be zeros and first four should have at least 3 unique characters
            if poss_key_locs[idx] == 0 and poss_key_locs[idx - 1] == 1 and \
                    all(aligned_heap[idx][4:]) == 0 and len(set(aligned_heap[idx][:4])) > 2:
                poss_key_locs[idx] = 1
            idx += 1

        # Roll the data to the left
        rolled = np.roll(poss_key_locs, -1)
        # The key cannot start at the last byte and then the block contain the whole key.
        # So the last value is set to False
        rolled[-1] = False
        poss_key_locs = (poss_key_locs & rolled).astype(int)

        # Roll right and OR it. The whole operation is similar to the opening morphological operation
        rolled = np.roll(poss_key_locs, 1)
        rolled[0] = False

        poss_key_locs = poss_key_locs | rolled

        characters, counts = self.get_run_length_encoded(poss_key_locs)

        cum_sum = [0]
        for idx in range(len(counts)):
            cum_sum.append(cum_sum[idx] + counts[idx])

        cum_sum = [x * 8 for x in cum_sum]

        # The last offset is not required for the cumulative sum
        return characters, counts, cum_sum[:-1]

    @staticmethod
    def get_run_length_encoded(data_block):

        idx = 1
        characters = []
        counts = []
        count = 1
        curr_char = data_block[0]
        while idx < len(data_block):
            if data_block[idx] == curr_char:
                idx += 1
                count += 1
                continue

            else:
                characters.append(curr_char)
                counts.append(count)

                count = 1
                curr_char = data_block[idx]

            idx += 1

        # Append the last character and count
        characters.append(curr_char)
        counts.append(count)

        return bytearray(characters), counts

    def generate_data(self):

        dataset = []
        labels = []

        characters, counts, cum_sum = self.get_possible_key_locations()

        # c = [counts[idx] for idx in range(len(counts)) if characters[idx] == 1]
        # print(max(c))

        viable_offsets = [cum_sum[idx] for idx in range(len(cum_sum)) if characters[idx] == 1]
        slices, curr_labels = self.get_slices(offsets=viable_offsets)

        dataset = dataset + slices
        labels = labels + curr_labels

        return dataset, labels

    def get_slices(self, offsets):
        data_blocks = []
        labels = []
        last_frame_added = False
        key_count = [0] * len(self.keys)
        for idx, offset in enumerate(offsets):
            if offset + self.max_key_size > len(self.heap):
                curr_data = self.heap[-self.window_size:]
                last_frame_added = True
            else:

                # Check if the current slice + window size overs the next offset too and if it covers, set the data as 0
                # The aim is not to have overlaps between slices
                if (idx + 1 < len(offsets) and offset + self.window_size <= offsets[idx + 1]) or (
                        (idx + 1) >= len(offsets)):
                    curr_data = self.heap[offset:offset + self.window_size]

                else:
                    # Get the data until the next offset and then replace the rest with 0's
                    curr_data = self.heap[offset:offsets[idx + 1]]
                    curr_data = curr_data + bytearray([0] * (self.window_size - len(curr_data)))

            data_blocks.append(np.array(curr_data))

            found = [l_idx for l_idx in range(len(self.keys)) if self.keys[l_idx] in curr_data]

            if len(found) > 0:
                labels.append(1)
                for key_idx in set(found):
                    key_count[key_idx] += 1

            else:
                labels.append(0)

            if last_frame_added is True:
                break

        return data_blocks, labels







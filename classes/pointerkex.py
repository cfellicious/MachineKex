import numpy as np
from classes.graphkex import GraphKex


class PointerKex(GraphKex):
    sizes = None
    pointer_sizes_dict = dict()

    def __init__(self, heap_path, json_path):
        super().__init__(heap_path=heap_path, json_path=json_path)
        self.sizes = None
        self.pointer_sizes_dict = dict()

    def get_pointers(self):

        pointer_list = dict()
        feature_list_count = 0
        # Get the pointer candidates
        self.get_pointer_candidate_indices()

        # Create a mask
        mask = np.zeros(self.aligned_heap.shape[0])
        mask[self.pointer_candidate_indices] = 1
        mask = mask.astype(int)

        # Translate the base address into the corresponding heap offset address to compute the sizes
        base_address_translated = self.aligned_heap - (mask * self.base_address_int)

        # Select the malloc header
        heap_offsets = base_address_translated[self.pointer_candidate_indices].astype(int) // 8 - 1

        # Convert it to int
        self.sizes = self.aligned_heap[heap_offsets].astype(int)

        # We set this as 8 as the next instruction is to subtract 8 and thus setting the values as 0
        self.sizes[self.sizes <= 0] = 8
        self.sizes = self.sizes - 8
        self.sizes[self.sizes > self.heap_size] = 0

        # Convert the aligned heap to list for easier accessibility
        self.aligned_heap = self.aligned_heap.tolist()
        self.aligned_heap_size = len(self.aligned_heap)

        # Create a dictionary with the corresponding sizes for each pointer
        self.pointer_sizes_dict = dict()
        for idx in range(len(self.sizes)):
            self.pointer_sizes_dict[self.aligned_heap[self.pointer_candidate_indices[idx]]] = self.sizes[idx]

        self.pointer_list = None
        feature_list = np.empty(shape=(len(self.pointer_candidate_indices), 5), dtype=np.int32)

        for dataset_idx, idx in enumerate(self.pointer_candidate_indices.tolist()):
            # curr_row = self.formatted_heap[idx]
            # curr_row = ''.join(hex_dict[x] for x in self.heap[idx*8:(idx+1)*8])
            # if self.base_address_int <= self.aligned_heap[idx] <= self.end_of_heap:
            # address = ''.join(format(x, '02x') for x in self.aligned_heap[idx][::-1]).upper()
            # address.lstrip('0')
            # address = curr_row[-6] + curr_row[-5] + \
            #           curr_row[-8] + curr_row[-7] + curr_row[-10] + \
            #           curr_row[-9] + curr_row[-12] + \
            #           curr_row[-11] + curr_row[-14] + \
            #           curr_row[-13] + curr_row[-16] + curr_row[-15]

            address = self.aligned_heap[idx]
            # data_addr = self.resolve_pointer_address(address=address)
            data_addr = self.aligned_heap[idx] - self.base_address_int
            if data_addr <= 0:
                continue

            # size = self.get_allocation_size(heap_offset=data_addr)
            size = self.sizes[dataset_idx]

            data_addr = int(data_addr / 8)
            indices_to_check = int(size / 8) + 1
            out_degree = 0
            pointer_count = 0

            final_pointer_offset = -1
            final_valid_pointer_offset = -1

            # Get the heap offset by resolving the pointer
            for idx_range in range(indices_to_check):
                if self.is_pointer_candidate_int(self.aligned_heap[data_addr + idx_range]) is True:
                    pointer_count += 1
                    final_pointer_offset = idx_range
                    # Add size here
                    if self.is_heap_address_valid(data_addr + idx_range) is True:
                        final_valid_pointer_offset = idx_range
                        if self.pointer_sizes_dict.get(self.aligned_heap[data_addr + idx_range], 0) > 0:
                            out_degree += 1
            # feature_list.append([size, pointer_count, out_degree, final_pointer_offset,
            #                      final_valid_pointer_offset])
            feature_list[dataset_idx] = [size, pointer_count, out_degree, final_pointer_offset,
                                         final_valid_pointer_offset]
            # Sometimes there are multiple pointers that point to the NEWKEYS. So these all have to be considered
            # as positive samples. Therefore, we need an array to store the indices
            # pointer_list[address.lstrip('0')] = feature_list_count
            # feature_list_count += 1
            idx_list = pointer_list.get(address, [])
            idx_list.append(feature_list_count)
            pointer_list[address] = idx_list
            feature_list_count += 1

        return feature_list, pointer_list

    def generate_data(self):
        relevant_nodes = []
        if self.info.get('NEWKEYS_1_ADDR', None) is not None:
            relevant_nodes = [int(self.info.get('NEWKEYS_1_ADDR').upper(), 16)]
        if self.info.get('NEWKEYS_2_ADDR', None) is not None:
            relevant_nodes = relevant_nodes + [int(self.info.get('NEWKEYS_2_ADDR').upper(), 16)]
        if len(relevant_nodes) == 0:
            return [], []
        features, pointer_list = self.get_pointers()
        curr_labels = np.zeros(features.shape[0], dtype=int)
        # There should be two new keys.
        curr_labels[pointer_list.get(relevant_nodes[0])] = 1
        curr_labels[pointer_list.get(relevant_nodes[1])] = 1
        return features, curr_labels

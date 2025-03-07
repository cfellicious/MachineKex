import networkx as nx
import numpy as np
import re
from classes.kex import Kex


class GraphKex(Kex):

    heap_graph = None
    i_mask = 0x0F0000000000
    MASK = np.uint64(i_mask)
    base_address = None
    pointer_candidate_indices = None

    pointer_list = None
    heap = None
    ssh_struct_address = None
    aligned_heap = None
    formatted_heap = None
    length = 0
    heap_size = 0
    pointer_offsets = None

    def __init__(self, heap_path, json_path):
        super().__init__(heap_path=heap_path, json_path=json_path)

        # initialize the heap and info
        self.read_data()
        self.base_address = self.info.get('HEAP_START', '0x0000')
        self.base_address_int = int(self.base_address, 16)
        self.aligned_heap = np.frombuffer(self.heap, dtype=np.uint64)
        self.aligned_heap_size = len(self.aligned_heap)
        self.end_of_heap = self.base_address_int + self.heap_size

    def get_pointer_candidate_indices(self):
        self.pointer_candidate_indices = np.asarray(np.logical_and
                                                    (np.logical_and(self.aligned_heap >= self.base_address_int,
                                                                    self.aligned_heap < self.end_of_heap),
                                                     np.mod(self.aligned_heap, 8) == 0)).nonzero()[0]

    @staticmethod
    def convert_to_big_endian(data):
        """
        Function converts hex little endian to big endian
        :param data:
        :return:
        """
        # temp = bytearray.fromhex(data)
        # temp.reverse()
        # return ''.join(format(x, '02x') for x in temp).upper()
        return data[-2] + data[-1] + data[-4] + data[-3] + data[-6] + data[-5] + data[-8] + data[-7] + data[-10] + \
            data[-9] + data[-12] + data[-11] + data[-14] + data[-13] + data[-16] + data[-15]

    def is_heap_address_valid(self, idx):
        if idx > self.aligned_heap_size:
            return False

        # address = ''.join(format(x, '02x') for x in self.aligned_heap[idx][::-1]).upper()
        address = self.aligned_heap[idx]
        if address <= self.base_address_int or address > self.end_of_heap:
            return False

        return True

    def get_allocation_size(self, heap_offset):
        # Get the 8-byte aligned offset for formatted heap
        print(heap_offset)
        heap_offset = int(heap_offset/8)

        # Header info is in the previous byte in little endian form
        header_offset = heap_offset - 1
        size = self.aligned_heap[header_offset]
        if size <= 0:
            return 0

        # 8 bytes for the malloc header
        size = size - 8

        if size > self.heap_size:
            return 0
        return size

    def get_pointer_allocation_size(self, address):
        # address = self.aligned_heap[idx]
        if address <= self.base_address_int or \
                address > self.end_of_heap:
            return 0

        # Get the heap offset by resolving the pointer
        # self.resolve_pointer_address(pointer)
        heap_offset = address - self.base_address_int

        # Get the 8-byte aligned offset for formatted heap
        heap_offset = int(heap_offset / 8)

        # Header info is in the previous byte in little endian form
        header_offset = heap_offset - 1
        size = self.aligned_heap[header_offset]

        if size <= 0:
            return 0

        # 8 bytes for the malloc header
        size = size - 8

        if size > self.heap_size:
            return 0
        return size

    def resolve_pointer_address(self, address):
        """
        Gets the actual offset from the starting of the heap
        :param address: Valid address in the heap
        :return: Actual offset of the heap in base10
        """
        # diff = int(address, 16) - self.base_address_int
        diff = address - self.base_address_int
        if diff <= 0 or diff > self.heap_size:
            return 0
        return diff

    def count_pointers(self, pointer):
        # The number of pointers for the allocated memory
        num_pointers = 0
        allocation_size = int(self.get_pointer_allocation_size(address=pointer) / 8) + 1
        starting_addr = int(self.resolve_pointer_address(address=pointer) / 8)
        idx = 0
        last_pointer_offset = -1
        last_valid_pointer_offset = -1
        while idx < allocation_size:
            if self.is_pointer_candidate_int(self.aligned_heap[starting_addr + idx]) is True:
                num_pointers += 1
                last_pointer_offset = idx
                if self.is_heap_address_valid(starting_addr + idx) is True:
                    last_valid_pointer_offset = idx

            idx += 1
        return num_pointers, last_pointer_offset, last_valid_pointer_offset

    def is_pointer_candidate_int(self, address):
        # A bit strange, but is the same as the original regex. Should be modified to be more logical
        if address <= 0xFFFFFFFFFFFF and address & self.i_mask > 0:
            return True

        return False

    def build_graph(self):

        self.heap_graph = nx.DiGraph()
        self.get_pointer_candidate_indices()
        pointers = list(set(self.aligned_heap[self.pointer_candidate_indices]))
        self.aligned_heap = self.aligned_heap.tolist()

        # Big endian valid pointers
        # pointers = [self.aligned_heap[x] for x in self.pointer_candidate_indices]

        # Select only pointers which have addresses within the heap
        # pointers = [x.lstrip("0") for x in pointers if self.is_heap_address_valid(x) is True]

        # Take only unique pointers. Multiple unique addresses are not necessary
        # pointers = set(pointers)

        # pointer that points to a data structure should be 16 bytes align (malloc in 64bit is 16 bytes aligned)
        # https://www.gnu.org/software/libc/manual/html_node/Aligned-Memory-Blocks.html
        # pointers = [x for x in pointers if int(x, 16) % 16 == 0]
        # print(pointers)

        # Create all the nodes
        for pointer in pointers:
            allocated_size = self.get_pointer_allocation_size(address=pointer)
            if allocated_size <= 0:
                continue

            num_pointers, last_pointer_offset, last_valid_pointer_offset = self.count_pointers(pointer)
            self.heap_graph.add_node(pointer, size=allocated_size,
                                     pointer_count=num_pointers, offset=0,
                                     last_pointer_offset=last_pointer_offset,
                                     last_valid_pointer_offset=last_valid_pointer_offset)

        # Start adding edges
        for idx, pointer in enumerate(pointers):

            # Find how many rows are contained in the heap
            struct_start_addr = int(self.resolve_pointer_address(address=pointer) / 8)
            # Get the search boundary from the allocated size stored in the node
            allocated_size = self.heap_graph.nodes.get(pointer, {}).get('size', 0)
            if allocated_size == 0:
                continue
            struct_ending_addr = struct_start_addr + int(allocated_size / 8)

            if struct_ending_addr >= self.aligned_heap_size:
                continue

            inner_idx = struct_start_addr
            while inner_idx <= struct_ending_addr:
                if inner_idx >= self.aligned_heap_size:
                    print(pointer)
                    print(self.info.get('PATH'))
                if self.is_pointer_candidate_int(self.aligned_heap[inner_idx]) is True:
                    # We found a pointer, we add it as an edge from pointer to the identified address
                    # Convert the v edge to big endian and set the offset as the edge property
                    # Remove self loops by checking whether both the pointers are identical
                    target_pointer = self.aligned_heap[inner_idx]
                    # self.convert_to_big_endian(self.formatted_heap[inner_idx]).lstrip("0")
                    if self.heap_graph.has_node(pointer) and self.heap_graph.has_node(target_pointer) and \
                            pointer != target_pointer:
                        # set offset on the node as well
                        # We cannot add the offset to a node as there are multiple pointers to a node
                        # self.heap_graph.nodes.get(target_pointer, {}).update(
                        #     {"offset": (inner_idx - struct_start_addr) * 8})

                        self.heap_graph.add_edge(u_of_edge=pointer,
                                                 v_of_edge=target_pointer,
                                                 offset=(inner_idx - struct_start_addr) * 8)
                inner_idx += 1

        # remove nodes that do not have any incoming or outgoing edges
        self.heap_graph.remove_nodes_from(list(nx.isolates(self.heap_graph)))

        # Remove isolated pairs of nodes
        starting_points = [x for x in self.heap_graph.nodes if self.heap_graph.in_degree(x) == 0]
        nodes_to_be_removed = []
        for node in starting_points:
            if self.heap_graph.out_degree(node) == 1 and \
                    len(list(self.heap_graph.neighbors(list(self.heap_graph.neighbors(node))[0]))) == 0:
                nodes_to_be_removed.append(node)

        # Remove the nodes
        self.heap_graph.remove_nodes_from(nodes_to_be_removed)

        # Remove the neighbors of the deleted nodes which are now isolated nodes
        self.heap_graph.remove_nodes_from(list(nx.isolates(self.heap_graph)))

        # Update the starting points
        # self.starting_points = [x for x in self.heap_graph.nodes if self.heap_graph.in_degree(x) == 0]

    def get_feature_vector(self, node):
        properties = self.heap_graph.nodes.get(node)
        size = properties.get('size')
        pointer_count = properties.get('pointer_count')
        last_pointer_offset = properties.get('last_pointer_offset')
        last_valid_pointer_offset = properties.get('last_valid_pointer_offset')
        out_degree = self.heap_graph.out_degree[node]
        predecessor = list(self.heap_graph.predecessors(node))
        if len(predecessor) > 0:
            predecessor_node_info = self.heap_graph.nodes.get(predecessor[0])
            predecessor_size = predecessor_node_info.get('size', 0)
            predecessor_pointer_count = predecessor_node_info.get('pointer_count', 0)
            # predecessor_offset = predecessor_node_info.get('offset', 0)
            offset = self.heap_graph.edges.get((predecessor[0], node)).get('offset', 0)
            predecessor_out_degree = self.heap_graph.out_degree[predecessor[0]]
        else:
            predecessor_size = 0
            predecessor_pointer_count = 0
            offset = 0
            predecessor_out_degree = 0

        curr_feature_vector = [size, pointer_count, out_degree, offset, last_pointer_offset,
                               last_valid_pointer_offset, predecessor_size, predecessor_pointer_count,
                               predecessor_out_degree]

        return curr_feature_vector

    def generate_data(self):
        if len(self.key_address) == 0:
            return [], []

        self.build_graph()
        total_nodes = self.heap_graph.number_of_nodes()
        labels = [0] * total_nodes
        dataset = [[0] * 9] * total_nodes
        idx = 0

        for node in self.heap_graph.nodes:
            dataset[idx] = self.get_feature_vector(node=node)
            labels[idx] = 0
            if node in self.key_address:
                labels[idx] = 1
            idx += 1
        return dataset, labels


class TestGraphKex(GraphKex):

    clf = None

    def __init__(self, heap_path, clf, base_address, feature_size=9):

        super().__init__(heap_path=heap_path, json_path=None)
        self.clf = clf
        self.base_address = base_address
        self.base_address_int = int(self.base_address, 16)
        self.end_of_heap = self.base_address_int + self.heap_size
        self.feature_size = feature_size

    def predict_keys(self):
        self.build_graph()
        dataset = np.empty((self.heap_graph.number_of_nodes(), self.feature_size))
        pointers = np.zeros(self.heap_graph.number_of_nodes(), dtype=np.uint64)

        for idx, node in enumerate(self.heap_graph.nodes):
            dataset[idx] = self.get_feature_vector(node=node)
            pointers[idx] = node

        y_pred = pointers[self.clf.predict(dataset).nonzero()[0]]
        keys = []
        for index in y_pred.tolist():
            heap_offset = self.resolve_pointer_address(index)
            key = ''.join(format(x, '02x') for x in self.heap[heap_offset:heap_offset + 64])
            keys.append(key.upper())

        return keys




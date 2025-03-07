import json


class Kex:

    heap_path = None
    json_path = None
    heap = None
    info = None
    keys = None
    key_address = None
    heap_size = 0

    def __init__(self, heap_path, json_path=None):
        self.heap_path = heap_path
        self.json_path = json_path

    def read_data(self):
        with open(self.heap_path, 'rb') as fp:
            self.heap = bytearray(fp.read())
            self.heap_size = len(self.heap)

        if self.json_path is not None:
            with open(self.json_path, 'r') as fp:
                self.info = json.load(fp)
                self.info['PATH'] = self.json_path

            keys = ['KEY_A', 'KEY_B', 'KEY_C', 'KEY_D', 'KEY_E', 'KEY_F']
            self.keys = []
            self.key_address = []
            for key in keys:
                curr_key = self.info.get(key, None)
                if curr_key is None:
                    continue
                if len(curr_key) > 12:
                    self.keys.append(curr_key.upper())
                    curr_key_addr = self.info.get(key+'_ADDR', None)
                    if curr_key_addr is not None:
                        self.key_address.append(int(curr_key_addr, 16))

        else:
            # Create an empty dictionary for testing purposes
            self.info = dict()

    def generate_data(self):
        pass


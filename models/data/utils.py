import json


def load_char_to_num(path):
    with open (path, "r") as f:
        char_to_num = json.load(f)
    num_to_char = {j:i for i,j in char_to_num.items()}
    return num_to_char

def decode_fn():
    pass


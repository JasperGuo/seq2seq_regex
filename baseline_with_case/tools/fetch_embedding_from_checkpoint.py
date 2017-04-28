# coding=utf8

import argparse
import numpy as np
from tensorflow.python import pywrap_tensorflow


def fetch(checkpoint_path, tensor_name, save_path=None):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    target_tensor = None
    for key in var_to_shape_map:
        if tensor_name == key:
            target_tensor = reader.get_tensor(key)
            break
    print(target_tensor)
    if save_path and isinstance(target_tensor, np.ndarray):
        np.save(save_path, target_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", help="Checkpoint path", required=True)
    parser.add_argument("--tensor", help="Tensor name", required=True)
    parser.add_argument("--save", help="Save path", required=False)
    args = parser.parse_args()

    fetch(args.checkpoint, args.tensor, args.save)

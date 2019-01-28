
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
"""Simple benchmark for testing data loader performance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import dummy_dataset

import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops


class DummyDataset(dataset_ops.DatasetSource):

    def __init__(self, n):
        super(DummyDataset, self).__init__()
        self.n = n

    def _as_variant_tensor(self):
        dummy_op = load_custom_tf_op("op_dummy_dataset.so")
        return dummy_op.dummy_dataset(self.n)

    @property
    def output_classes(self):
        "output_classes method."
        return tuple(tf.Tensor for _ in range(self.n))

    @property
    def output_shapes(self):
        "output_shapes method."
        return tuple(tf.TensorShape([]) for _ in range(self.n))

    @property
    def output_types(self):
        "output_types method."
        return tuple(tf.int32 for _ in range(self.n))


def benchmark(n):
    dataset = dummy_dataset.DummyDataset(n)

    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()
    num_rows = 0

    start_time = time.time()

    with tf.Session() as sess:
        while num_rows < 20000:
            _row = sess.run(get_next)
            num_rows += 1
            if num_rows % 2000 == 0:
                print(_row)
                print("n={}: Read {} rows, {} rows / sec"
                      .format(n, num_rows, num_rows / (time.time() - start_time)))


    print("Finished (n=): read {} rows, {} rows / sec"
          .format(n, num_rows, num_rows / (time.time() - start_time)))


def main(args=None):
    """Benchmark dataset reading."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_columns",
        type=int,
        default=30,
        help="Number of cols"
    )
    args = parser.parse_args()

    benchmark(args.num_columns)


if __name__ == '__main__':
    main()

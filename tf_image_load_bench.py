from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys

import numpy as np
import tensorflow as tf

def imagename_gen():
    while True:
        yield "/data/testimage.fp16"

def load_one_image(path):
    data = tf.read_file(path)
    image = tf.decode_raw(data, tf.float16)
    image = tf.reshape(image, [3, 504, 960])

    # TOGGLE THIS:
    #image = tf.cast(image, tf.float32)
    return {'image': image, 'path': path}

def run():
    batch_size = 32
    dataset = tf.data.Dataset.from_generator(
        imagename_gen, (tf.string), output_shapes=tf.TensorShape(None))

    dataset = dataset.map(load_one_image)
    dataset = dataset.batch(batch_size)
    get_next = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        start_time = time.time()

        for batch_count in range(200):
            row = sess.run(get_next)

            if batch_count % 20 == 19:
                print(row)
                print("-- {} images/sec".format(
                    (batch_count * batch_size) / (time.time() - start_time)))

        print("Finished {} batches in {} secs".format(
            batch_count, time.time() - start_time))





def main():
    run()

if __name__ == '__main__':
    main()

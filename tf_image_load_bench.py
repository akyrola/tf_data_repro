from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys

import numpy as np
import tensorflow as tf

def imagename_gen():
    while True:
        yield "testimage.fp16"

def load_one_image(path):
    data = tf.read_file(path)
    image = tf.decode_raw(data, tf.float16)
    image = tf.reshape(image, [3, 504, 960])

    # TOGGLE THIS:
    image = tf.cast(image, tf.float32)
    return {'image': image, 'path': path}


def run():
    tfdata = True
    dataset = tf.data.Dataset.from_generator(
        imagename_gen, (tf.string), output_shapes=tf.TensorShape(None))

    if tfdata:
        print("---- use tf.data.map() ----")
        dataset = dataset.map(load_one_image)
        get_next = dataset.make_one_shot_iterator().get_next()
    else:
        print("---- use plain tf ----")
        ds_next = dataset.make_one_shot_iterator().get_next()
        get_next = load_one_image(ds_next)

    with tf.Session() as sess:
        start_time = time.time()

        for i in range(5000):
            _ = sess.run(get_next['image'].op)

            if i % 200 == 199:
                print("-- {} images/sec".format(i / (time.time() - start_time)))

        print("{}: Finished {} images in {} secs".format(
            "tfdata.map" if tfdata else "plain tf",
            i, time.time() - start_time))





def main():
    run()

if __name__ == '__main__':
    main()

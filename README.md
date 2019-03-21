## tf.data map issues

See: [tf_image_load_bench.py](tf_image_load_bench.py).

This benchmark surfaces strange behaviour of using `tf.data.map` with a `cast` call.
The below table shows the problem: adding a cast in our `tf.data.map` operation
is surprisingly slow, and uses a lot of CPU resources.

|       | fp16           | +cast(fp32)    |
|-------|----------------|----------------|
| map   | 11s (120% CPU) | 23s (**350% CPU**) |
| plain | 27s (100% CPU) | 26s (100% CPU) |


## Dummy Benchmark

Repro issue with TF.data performance:

dummy_dataset.cc is a tf.data operator that return a specified number of integer scalar tensors (parameter "n").

dummy_dataset.py is the python test to execute to the test.


Benchmarks on Ubuntu 16.04, Tensorflow r1.12. Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz

Results (approximate):
- n=1:   8300 rows/sec
- n=2:   6800 rows/sec
- n=5:   4800 rows/sec
- n=10:  3200 rows/sec
- n=20:  2000 rows/sec
- n=40:  1100 rows/sec



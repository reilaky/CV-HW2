import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
'''
(1) input (load) the final network parameters from prog7
(2) process images in the test batch (10,000 images)
(3) estimate the performance (error rate)
	using the final network parameters on the test images only and output the error rate.
'''


# implement a five layer CNN
'''
layer 1: CNN(relu, max pooling)
layer 2: CNN(relu, max pooling)
layer 31: CNN(relu)
layer 32: CNN(relu, max pooling)
layer 4: FNN
layer 5: output layer
'''

with tf.Session() as session:
    saver = tf.train.import_meta_graph(CURRENT_PATH + '/model.ckpt')
    saver.restore(sess, tf.train.latest_checkpoint(savedir))

    graph = tf.get_default_graph()
    input_x = graph.get_tensor_by_name("input_x:0")
    result = graph.get_tensor_by_name("result:0")

    feed_dict = {input_x: x_data,}

    predictions = result.eval(feed_dict=feed_dict)

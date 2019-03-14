import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pickle

'''
(1) input (load) the final network parameters from prog7
(2) process images in the test batch (10,000 images)
(3) estimate the performance (error rate)
	using the final network parameters on the test images only and output the error rate.
'''

os.environ['KMP_DUPLICATE_LIB_OK']='True'
TEST_DATASET = 'test_batch'
FOLDER_NAME = 'cifar-10-batches-py/'
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3


# load testset
# dataset: CIFAR10
# 10 classes: (airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck)
def one_hot_encode(np_array, num_label):
	temp = (np.arange(num_label) == np_array[:,None]).astype(np.float32)
	return temp

def reformat_data(dataset, label):
	np_dataset_ = dataset.reshape((len(dataset), IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT)).transpose(0, 2, 3, 1)
	num_label = len(np.unique(label))
	np_label_ = one_hot_encode(np.array(label, dtype=np.float32), num_label)
	np_dataset, np_label = randomize(np_dataset_, np_label_)
	return np_dataset, np_label

def randomize(dataset, label):
	permutation = np.random.permutation(label.shape[0])
	shuffled_dataset = dataset[permutation, :, :]
	shuffled_label = label[permutation]
	return shuffled_dataset, shuffled_label

def data_process():

	file_path = CURRENT_PATH + '/' + FOLDER_NAME
	with open(file_path + TEST_DATASET, 'rb') as f1:
		test_dict = pickle.load(f1, encoding = 'bytes')
		test_dataset, test_label = test_dict[b'data'], test_dict[b'labels']
	
	test_dataset, test_label = reformat_data(test_dataset, test_label)
	print("test dataset contains {} images, each of size {}".format(test_dataset[:,:,:,:].shape[0], test_dataset[:,:,:,:].shape[1:]))
	imgplot = plt.imshow(test_dataset[100,:,:,:])
	print(test_label[100])
	plt.show()
	return test_dataset, test_label

test_dataset, test_label = data_process()

tf_test_dataset = tf.constant(test_dataset, tf.float32)

with tf.Session() as session:
	saver = tf.train.import_meta_graph(CURRENT_PATH + '/model.ckpt')
	saver.restore(sess, tf.train.latest_checkpoint(savedir))

	graph = tf.get_default_graph()
	input_x = graph.get_tensor_by_name("input_x:0")
	result = graph.get_tensor_by_name("result:0")

	feed_dict = {input_x: x_data,}

	predictions = result.eval(feed_dict=feed_dict)

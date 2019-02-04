import tensorflow as tf
import numpy as np
import numpy.random as npr
import scipy
import os
import scipy.io
import sys
import gzip
import matplotlib.pyplot as plt
try:
	import cPickle
except:
	import _pickle as cPickle


# sess = tf.Session()
# with gzip.open('mnist.pkl.gz', 'rb') as f:
#   # python3
#   #((traind,trainl),(vald,vall),(testd,testl))=pickle.load(f, encoding='bytes')
#   # python2
#   ((traind,trainl),(vald,vall),(testd,testl))=cPickle.load(f, encoding="bytes") ;
#   traind = traind.astype("float32").reshape(-1,784) ;
#   trainl = trainl.astype("float32") ;
#   testd = testd.astype("float32").reshape(-1,784) ;
#   testl = testl.astype("float32") ;
# data_placeholder = tf.placeholder(tf.float32,[None,784]) ;
# label_placeholder = tf.placeholder(tf.float32,[None,10]) ;
# keep_prob = tf.placeholder(tf.float32)

# batch_size = 64
# training_iters = 10
# display_step = 20
# dropout = 0.8


# N = 10000 ;
# fd = {data_placeholder: traind[0:N], label_placeholder : trainl[0:N], keep_prob: dropout } ;

# dataReshaped=tf.reshape(data_placeholder, (-1,28,28,1)) ;


def max_pool(name, l_input, k):
	return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def dropout(x, keepPro, name = None):
	return tf.nn.dropout(x, keepPro, name)


def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)





def conv2d(x, kHeight, kWidth, in_channels,
			  filters, name, padding = "SAME"): #group=2 means the second part of AlexNet
	with tf.variable_scope(name):
		w       = tf.get_variable("W", shape=[kHeight, kWidth, in_channels, filters], dtype=tf.float32, collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
		b       = tf.get_variable("b", shape=[filters], dtype=tf.float32, collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
		conv    = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=padding)
		out     = tf.nn.bias_add(conv,b, name='convolution')
		relu    = tf.nn.relu(out, name='relu')

	return relu;





def AlexNet(inp, _dropout, phase, num_outputs=1000):
	dataReshaped=tf.reshape(inp, (-1,28,28,1)) ;

	conv1   = conv2d(dataReshaped, 5, 5, 1, 32, 'conv1')

	pool1   = max_pool('pool1', conv1, k=2)
	norm1   = norm('norm1', pool1, lsize=4)
	#norm1 = tf.nn.dropout(norm1, _dropout)

	conv2 = conv2d(norm1, 3, 3, 32, 64, 'conv2')
	pool2 = max_pool('pool2', conv2, k=2)
	norm2 = norm('norm2', pool2, lsize=4)
	#norm2 = tf.nn.dropout(norm2, _dropout)

	conv3 = conv2d(norm2, 3, 3, 64, 128, 'conv3')
	pool3 = max_pool('pool3', conv3, k=2)
	norm3 = norm('norm3', pool3, lsize=4)
	#norm3 = tf.nn.dropout(norm3, _dropout)

	#w1  = tf.random_normal([4*4*128, 1000])
	#b   = tf.random_normal([1000])
	Z3 = 1000

	w1v = npr.uniform(-0.01,0.01, [4*4*128,Z3])

	b1v = npr.uniform(-0.01,0.01, [1,Z3])
	w1Init = tf.constant_initializer(w1v)
	b1Init = tf.constant_initializer(b1v)

	
	#w1 = tf.Variable(npr.uniform(-0.01,0.01, [4*4*128,Z3]),dtype=tf.float32, name ="W3") ;
	#b = tf.Variable(npr.uniform(-0.01,0.01, [1,Z3]),dtype=tf.float32, name ="b3") ;

	w1 = tf.get_variable(initializer=w1Init,dtype=tf.float32, name ="W3", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],shape=[2048,1000] ) ;
	b = tf.get_variable(initializer=b1Init,dtype=tf.float32, name ="b3", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES], shape=[1,1000]) ;


	dense1 = tf.reshape(norm3, [-1, w1.get_shape().as_list()[0]])
	dense1 = tf.nn.relu(tf.matmul(dense1, w1) + b, name='fc1')

	w2v = npr.uniform(-0.1,0.1, [Z3,10])
	b2v = npr.uniform(-0.01,0.01, [10])

	w2Init = tf.constant_initializer(w2v)
	b2Init = tf.constant_initializer(b2v)

	#w2 = tf.Variable(npr.uniform(-0.1,0.1, [Z3,10]),dtype=tf.float32, name ="W4") ;
	#b2 = tf.Variable(npr.uniform(-0.01,0.01, [10]),dtype=tf.float32, name ="b4") ;
	w2 = tf.get_variable(initializer=w2Init,dtype=tf.float32, name ="W4", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES], shape=[1000,10]) ;
	b2 = tf.get_variable(initializer=b2Init,dtype=tf.float32, name ="b4", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES], shape=[10,]) ;




	# w2  = tf.random_normal([1000, 1000])
	# b2  = tf.random_normal([1000])
	#dense2 = tf.nn.relu(tf.matmul(dense1, w2) + b2, name='fc2')
	out = tf.matmul(dense1, w2) + b2
	#wOut = tf.Variable(tf.random_normal([1000, 10]))
	#bOut = tf.Variable(tf.random_normal([10]))

	# wOut = tf.Variable(npr.uniform(-0.1,0.1, [Z3,10]),dtype=tf.float32, name ="wOut") ;
	# bOut = tf.Variable(npr.uniform(-0.01,0.01, [1,10]),dtype=tf.float32, name ="bOut") ;
	# out = tf.matmul(dense2, wOut) + bOut

	return out;

# pred = AlexNet(data_placeholder, 'train', keep_prob)
# print(pred)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels=label_placeholder))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(label_placeholder,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# tMax = 10;


# # expects 1D array
# def sm(arr):
# 	num = np.exp(arr) ;
# 	den = num.sum() ;
# 	return num/den ;


# def test_cb(self):
# 	global testit ;
# 	ax1.cla();
# 	ax2.cla();
# 	ax3.cla();
# 	ax1.imshow(testd[testit].reshape(28,28),cmap=plt.get_cmap("bone")) ;
# 	confs =sm(testout[testit]) ;
# 	ax2.bar(range(0,10),confs);
# 	ax2.set_ylim(0,1.)
# 	ce = -(confs*np.log(confs+0.00000001)).sum() ;
# 	ax3.text(0.5,0.5,str(ce),fontsize=20)
# 	testit = testit + 1;
# 	f.canvas.draw();
# 	print ("--------------------") ;
# 	print("logits", testout[testit], "probabilities", sm(testout[testit]), "decision", testout[testit].argmax(), "label", testl[testit].argmax()) ;






# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer()) ;

# 	for iteration in range(0,tMax):
# 		sess.run(optimizer, feed_dict = fd) ;

# 		acc= sess.run(accuracy, feed_dict = fd) ;
# 		loss = sess.run(cost, feed_dict = fd)
# 		#testacc = sess.run(accuracy, feed_dict = {data_placeholder: testd, label_placeholder: testl, keep_prob: 1.})
# 		#print ("epoch ", iteration, "acc=", float(correct), "loss=", lossVal, "testacc=",testacc) ;
# 		print("epoch ", iteration, "Training Accuracy= " + "{:.5f}".format(acc))
# 		print ("Testing Accuracy:", sess.run(accuracy, feed_dict={data_placeholder: testd, label_placeholder: testl}))

# 	print ("Optimization Finished!")
# 	print ("Testing Accuracy:", sess.run(accuracy, feed_dict={data_placeholder: testd, label_placeholder: testl}))


# 	testout = sess.run(pred, feed_dict = {data_placeholder : testd}) ;

# 	testit = 0 ;



# 	f,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3) ;
# 	f.canvas.mpl_connect('button_press_event', test_cb)
# 	plt.show();
#ax = f.gca() 




def get_weight_initializer(params):
    
    initializer = []
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    for layer, value in params.items():
        op = tf.get_variable('%s' % layer).assign(value)
        initializer.append(op)
    return initializer


def save_model(name, scope, sess):
    variables = tf.get_collection(tf.GraphKeys.WEIGHTS, scope=scope)
    print(variables)
    d = [(v.name.split(':')[0], sess.run(v)) for v in variables]
    cPickle.dump(d, open(name, 'wb'))
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
	if (name == 'pool5'):
		return tf.nn.avg_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
	else:
		return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def dropout(x, keepPro, name = None):
	return tf.nn.dropout(x, keepPro, name)


def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)





# def conv2d(x, kHeight, kWidth, in_channels,
# 			  filters, name, padding = "SAME"):
# 	with tf.variable_scope(name):
# 		w       = tf.get_variable("W", shape=[kHeight, kWidth, in_channels, filters], dtype=tf.float32, collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
# 		b       = tf.get_variable("b", shape=[filters], dtype=tf.float32, collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
# 		conv    = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=padding)
# 		out     = tf.nn.bias_add(conv,b, name='convolution')
# 		relu    = tf.nn.relu(out, name='relu')

# 	return relu;


def conv2d(name,x, W, B, strides=1):
    # Conv2D wrapper, with bias and relu activation
    with tf.variable_scope(name):
    	w = tf.get_variable("W", initializer=tf.random_normal(W), dtype=tf.float32, collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    	b = tf.get_variable("b", initializer=tf.random_normal(B), dtype=tf.float32, collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    	x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    	x = tf.nn.bias_add(x, b, name='convolution')
    	relu = tf.nn.relu(x,name='relu') 
    return relu

weights = {
    'wc1': [11, 11, 1, 96],
    'wc2': [5, 5, 96, 256],
    'wc3': [3, 3, 256, 384],
    'wc4': [3, 3, 384, 384],
    'wc5': [3, 3, 384, 256],
    'wd1': [4*4*256, 4096],
    'wd2': [4096, 1024],
    'out': [1024, 10]
}
biases = {
    'bc1': [96],
    'bc2': [256],
    'bc3': [384],
    'bc4': [384],
    'bc5': [256],
    'bd1': [4096],
    'bd2': [1024],
    'out': [10]
}




def AlexNet(inp, _dropout, phase, num_outputs=1000):

	dataReshaped=tf.reshape(inp, (-1,28,28,1)) ;

	conv1 = conv2d('conv1', dataReshaped, weights['wc1'], biases['bc1'])
	pool1 = max_pool('pool1', conv1, k=2)
	norm1 = norm('norm1', pool1, lsize=4)

	conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
	pool2 = max_pool('pool2', conv2, k=2)
	norm2 = norm('norm2', pool2, lsize=4)

	conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
	norm3 = norm('norm3', conv3, lsize=4)

	conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

	conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
	pool5 = max_pool('pool5', conv5, k=2)
	norm5 = norm('norm5', pool5, lsize=4)


	w1 = tf.get_variable(initializer=tf.random_normal(weights['wd1']),dtype=tf.float32, name ="W3", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES]) ;
	b = tf.get_variable(initializer=tf.random_normal(biases['bd1']),dtype=tf.float32, name ="b3", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES]) ;


	fc1 = tf.reshape(norm5, [-1, w1.get_shape().as_list()[0]])
	fc1 =tf.add(tf.matmul(fc1, w1),b)
	fc1 = tf.nn.relu(fc1)
	fc1=tf.nn.dropout(fc1,0.75)


	w2 = tf.get_variable(initializer=tf.random_normal(weights['wd2']),dtype=tf.float32, name ="W4", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES] ) ;
	b2 = tf.get_variable(initializer=tf.random_normal(biases['bd2']),dtype=tf.float32, name ="b4", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES]) ;
	
	fc2 = tf.reshape(fc1, [-1, w2.get_shape().as_list()[0]])
	fc2 =tf.add(tf.matmul(fc2, w2),b2)
	fc2 = tf.nn.relu(fc2)
	fc2=tf.nn.dropout(fc2,0.75)

	wout = tf.get_variable(initializer=tf.random_normal(weights['out']),dtype=tf.float32, name ="W5", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES] ) ;
	bout = tf.get_variable(initializer=tf.random_normal(biases['out']),dtype=tf.float32, name ="B5", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES] ) ;

	out = tf.add(tf.matmul(fc2, wout) ,bout)
	return out



# def AlexNet(inp, _dropout, phase, num_outputs=1000):
# 	dataReshaped=tf.reshape(inp, (-1,28,28,1)) ;

# 	conv1   = conv2d(dataReshaped, 5, 5, 1, 32, 'conv1')

# 	pool1   = max_pool('pool1', conv1, k=2)
# 	norm1   = norm('norm1', pool1, lsize=4)
# 	#norm1 = tf.nn.dropout(norm1, _dropout)

# 	conv2 = conv2d(norm1, 3, 3, 32, 64, 'conv2')
# 	pool2 = max_pool('pool2', conv2, k=2)
# 	norm2 = norm('norm2', pool2, lsize=4)
# 	#norm2 = tf.nn.dropout(norm2, _dropout)

# 	conv3 = conv2d(norm2, 3, 3, 64, 128, 'conv3')
# 	pool3 = max_pool('pool3', conv3, k=2)
# 	norm3 = norm('norm3', pool3, lsize=4)
# 	#norm3 = tf.nn.dropout(norm3, _dropout)

# 	#w1  = tf.random_normal([4*4*128, 1000])
# 	#b   = tf.random_normal([1000])
# 	Z3 = 1000

# 	w1v = npr.uniform(-0.01,0.01, [4*4*128,Z3])

# 	b1v = npr.uniform(-0.01,0.01, [1,Z3])
# 	w1Init = tf.constant_initializer(w1v)
# 	b1Init = tf.constant_initializer(b1v)

	
# 	#w1 = tf.Variable(npr.uniform(-0.01,0.01, [4*4*128,Z3]),dtype=tf.float32, name ="W3") ;
# 	#b = tf.Variable(npr.uniform(-0.01,0.01, [1,Z3]),dtype=tf.float32, name ="b3") ;

# 	w1 = tf.get_variable(initializer=w1Init,dtype=tf.float32, name ="W3", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],shape=[2048,2048] ) ;
# 	b = tf.get_variable(initializer=b1Init,dtype=tf.float32, name ="b3", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES], shape=[1,2048]) ;


# 	dense1 = tf.reshape(norm3, [-1, w1.get_shape().as_list()[0]])
# 	dense1 = tf.nn.relu(tf.matmul(dense1, w1) + b, name='fc1')

# 	w2v = npr.uniform(-0.1,0.1, [Z3,10])
# 	b2v = npr.uniform(-0.01,0.01, [10])

# 	w2Init = tf.constant_initializer(w2v)
# 	b2Init = tf.constant_initializer(b2v)

# 	#w2 = tf.Variable(npr.uniform(-0.1,0.1, [Z3,10]),dtype=tf.float32, name ="W4") ;
# 	#b2 = tf.Variable(npr.uniform(-0.01,0.01, [10]),dtype=tf.float32, name ="b4") ;
# 	w2 = tf.get_variable(initializer=w2Init,dtype=tf.float32, name ="W4", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES], shape=[2048,1000]) ;
# 	b2 = tf.get_variable(initializer=b2Init,dtype=tf.float32, name ="b4", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES], shape=[1,1000]) ;




# 	# w2  = tf.random_normal([1000, 1000])
# 	# b2  = tf.random_normal([1000])
# 	dense2 = tf.nn.relu(tf.matmul(dense1, w2) + b2, name='fc2')

# 	#out = tf.matmul(dense1, w2) + b2
	
# 	#wOut = tf.Variable(tf.random_normal([1000, 10]))
# 	#bOut = tf.Variable(tf.random_normal([10]))

# 	wOut = npr.uniform(-0.1,0.1, [Z3,10]) ;
# 	bOut = npr.uniform(-0.01,0.01, [10]) ;

# 	w3Init = tf.constant_initializer(wOut)
# 	b3Init = tf.constant_initializer(bOut)

# 	w3 = tf.get_variable(initializer=w3Init,dtype=tf.float32, name ="W5", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES], shape=[1000,10]) ;
# 	b3 = tf.get_variable(initializer=b3Init,dtype=tf.float32, name ="b5", collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES], shape=[10,]) ;


# 	out = tf.matmul(dense2, w3) + b3

# 	return out;

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
    d = [(v.name.split(':')[0], sess.run(v)) for v in variables]
    cPickle.dump(d, open(name, 'wb'))
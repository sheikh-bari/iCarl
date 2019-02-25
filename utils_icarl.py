import tensorflow as tf
import numpy as np
import os
import scipy.io
import sys
import utils_data
import utils_resnet
import alexnet
try:
    import cPickle
except:
    import _pickle as cPickle

def reading_data_and_preparing_network(index_of_files, files_from_cl, gpu, itera, batch_size, train_path, labels_dic, mixing, nb_groups, nb_cl, save_path, trainl, labels_from_cl, keep_prob):
    
    image_train, label_train,file_string       = utils_data.read_data_test_mnist(index_of_files, train_path,labels_dic, mixing,trainl, labels_from_cl, files_from_cl=files_from_cl)
    
    image_batch, label_batch,file_string_batch = tf.train.batch([image_train, label_train,file_string], batch_size=batch_size, num_threads=8)

    label_batch_one_hot = tf.one_hot(label_batch,nb_cl)
    
    ### Network and loss function  
    #mean_img = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    
    with tf.variable_scope('ResNet18'):
        with tf.device('/cpu:'+gpu):
            #scores         = utils_resnet.ownNet(image_batch, phase='test',num_outputs=nb_cl*nb_groups)
            scores         = alexnet.AlexNet(image_batch, keep_prob, phase='test', num_outputs=nb_cl)

            graph          = tf.get_default_graph()
            
            #op_feature_map = graph.get_operation_by_name('ResNet18/pool_last/avg').outputs[0]
            #op_feature_map = graph.get_operation_by_name('ResNet18/average_pooling2d/AvgPool').outputs[0]

            op_feature_map = graph.get_operation_by_name('ResNet18/pool3').outputs[0]
    
    loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch_one_hot, logits=scores))

    ### Initilization
    params = dict(cPickle.load(open(save_path+'model-iteration'+str(nb_cl)+'-%i.pickle' % itera, 'rb')))

    inits  = alexnet.get_weight_initializer(params)

    return inits,scores,label_batch,loss_class,file_string_batch,op_feature_map

def load_class_in_feature_space(files_from_cl,batch_size,scores, label_batch,loss_class,file_string_batch,op_feature_map,sess):
    processed_files=[]
    label_dico=[]
    Dtot=[]

    for i in range(int(np.ceil(len(files_from_cl)/batch_size)+1)):

        sc, l , loss,files_tmp,feat_map_tmp = sess.run([scores, label_batch,loss_class,file_string_batch,op_feature_map])
       
        processed_files.extend(files_tmp)
        label_dico.extend(l)
        feat_map_tmp_reshape = feat_map_tmp.reshape(128,1,1,2048)
        mapped_prototypes = feat_map_tmp_reshape[:,0,0,:]

        Dtot.append((mapped_prototypes.T)/np.linalg.norm(mapped_prototypes.T,axis=0))
    
    Dtot            = np.concatenate(Dtot,axis=1)
    processed_files = np.array(processed_files)
    label_dico      = np.array(label_dico)
    return Dtot,processed_files,label_dico

def prepare_networks(gpu,image_batch, nb_cl, nb_groups, keep_prob):
  #mean_img = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1], name='img_mean')

  scores   = []
  with tf.variable_scope('ResNet18'):
    with tf.device('/cpu:' + gpu):
        #score = utils_resnet.ownNet(image_batch, phase='train',num_outputs=nb_cl*nb_groups)
        score         = alexnet.AlexNet(image_batch, keep_prob, phase='test', num_outputs=nb_cl)
        scores.append(score)
    
    scope = tf.get_variable_scope()
    scope.reuse_variables()
  # First score and initialization
  variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='ResNet18')
  scores_stored   = []
  with tf.variable_scope('store_ResNet18'):
    with tf.device('/cpu:' + gpu):
        #score = utils_resnet.ownNet(image_batch, phase='train',num_outputs=nb_cl*nb_groups)
        score         = alexnet.AlexNet(image_batch, keep_prob, phase='test', num_outputs=nb_cl)
        scores_stored.append(score)
    
    scope = tf.get_variable_scope()
    scope.reuse_variables()
  variables_graph2 = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='store_ResNet18')
  
  return variables_graph,variables_graph2,scores,scores_stored



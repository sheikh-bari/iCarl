import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import scipy
import os
from scipy.spatial.distance import cdist
import scipy.io
import sys
import gzip
try:
    import cPickle
except:
    import _pickle as cPickle
# Syspath for the folder with the utils files
#sys.path.insert(0, "/data/sylvestre")

import utils_resnet
import utils_icarl
import utils_data

with gzip.open('mnist.pkl.gz', 'rb') as f:
    ((traind, trainl), (vald, vall), (testd, testl)) = cPickle.load(f, encoding="latin-1")
    traind = traind.astype("float32").reshape(-1,784)
    trainl = trainl.astype("float32")
    testd = testd.astype("float32").reshape(-1,784)
    testl = testl.astype("float32")

######### Modifiable Settings ##########
batch_size = 128             # Batch size
nb_cl      = 10              # Classes per group 
nb_groups  = 1              # Number of groups
top        = 5               # Choose to evaluate the top X accuracy 
itera      = 0               # Choose the state of the network : 0 correspond to the first batch of classes
eval_groups= np.array(range(itera+1)) # List indicating on which batches of classes to evaluate the classifier
gpu        = '0'             # Used GPU
########################################

######### Paths  ##########
# Working station 
# devkit_path = '/ssd_disk/ILSVRC2012/ILSVRC2012_devkit_t12'
# train_path  = '/ssd_disk/ILSVRC2012/train'
# save_path   = '/media/data/srebuffi/'

devkit_path = ''
#train_path  = '../../../images1'
save_path   = 'result/'

###########################

# Load ResNet settings
str_mixing = str(nb_cl)+'mixing.pickle'
with open(str_mixing,'rb') as fp:
    mixing = cPickle.load(fp)

str_settings_resnet = str(nb_cl)+'settings_resnet.pickle'
with open(str_settings_resnet,'rb') as fp:
    order       = cPickle.load(fp)
    files_valid = cPickle.load(fp)
    files_train = cPickle.load(fp)
    file_labels = cPickle.load(fp)
    file_indexes = cPickle.load(fp)
    labels_valid = cPickle.load(fp)
    all_file_indexes = cPickle.load(fp)
print(len(files_valid))
# Load class means
str_class_means = str(nb_cl)+'class_means.pickle'
with open(str_class_means,'rb') as fp:
      class_means = cPickle.load(fp)

# Loading the labels
#labels_dic, label_names, validation_ground_truth = utils_data.parse_devkit_meta(devkit_path)

define_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels_dic = {k: v for v, k in enumerate(define_class)}

# Initialization
acc_list = np.zeros((nb_groups,3))

    
# Load the evaluation files
print("Processing network after {} increments\t".format(itera))
print("Evaluation on batches {} \t".format(eval_groups))
files_from_cl = []
labels_from_cl = []
indexs_of_files = []
for i in eval_groups:
    files_from_cl.extend(files_valid[i])
    labels_from_cl.extend(labels_valid[i])
    indexs_of_files.extend(all_file_indexes[i])

inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(indexs_of_files, files_from_cl, gpu, itera, batch_size, traind, labels_dic, mixing, nb_groups, nb_cl, save_path, trainl, labels_from_cl) 

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    # Launch the prefetch system
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(inits)
    
    # Evaluation routine
    stat_hb1     = []
    stat_icarl = []
    stat_ncm     = []
    
    for i in range(int(np.ceil(len(files_from_cl)/batch_size))):
        sc, l , loss,files_tmp,feat_map_tmp = sess.run([scores, label_batch,loss_class,file_string_batch,op_feature_map])
        mapped_prototypes = feat_map_tmp[:,0,0,:]
        pred_inter    = (mapped_prototypes.T)/np.linalg.norm(mapped_prototypes.T,axis=0)
        sqd_icarl     = -cdist(class_means[:,:,0,itera].T, pred_inter.T, 'sqeuclidean').T
        sqd_ncm       = -cdist(class_means[:,:,1,itera].T, pred_inter.T, 'sqeuclidean').T
        stat_hb1     += ([ll in best for ll, best in zip(l, np.argsort(sc, axis=1)[:, -top:])])
        stat_icarl   += ([ll in best for ll, best in zip(l, np.argsort(sqd_icarl, axis=1)[:, -top:])])
        stat_ncm     += ([ll in best for ll, best in zip(l, np.argsort(sqd_ncm, axis=1)[:, -top:])])

    coord.request_stop()
    coord.join(threads)

print('Increment: %i' %itera)
print('Hybrid 1 top '+str(top)+' accuracy: %f' %np.average(stat_hb1))
print('iCaRL top '+str(top)+' accuracy: %f' %np.average(stat_icarl))
print('NCM top '+str(top)+' accuracy: %f' %np.average(stat_ncm))
acc_list[itera,0] = np.average(stat_icarl)
acc_list[itera,1] = np.average(stat_hb1)
acc_list[itera,2] = np.average(stat_ncm)

# Reset the graph to compute the numbers ater the next increment
tf.reset_default_graph()


np.save('results_top'+str(top)+'_acc_cl'+str(nb_cl),acc_list)

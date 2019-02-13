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
import matplotlib.pyplot as plt
import scikitplot as skplt

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
keep_prob = tf.placeholder(name="keep_prob", dtype=tf.float32)

######### Modifiable Settings ##########
batch_size = 128             # Batch size
nb_cl      = 5              # Classes per group 
nb_groups  = 2              # Number of groups
total_nb_cl = 10
top        = 1               # Choose to evaluate the top X accuracy 
itera      = 1               # Choose the state of the network : 0 correspond to the first batch of classes
eval_groups= np.array(range(itera+1)) # List indicating on which batches of classes to evaluate the classifier
gpu        = '0'             # Used GPU
########################################

######### Paths  ##########
# Working station 
# devkit_path = '/ssd_disk/ILSVRC2012/ILSVRC2012_devkit_t12'
# train_path  = '/ssd_disk/ILSVRC2012/train'
# save_path   = '/media/data/srebuffi/'

devkit_path = '50epochs300protoCustom9,1/'
#train_path  = '../../../images1'
save_path   = './50epochs300protoCustom9,1/result/'

###########################

mean_acc     = []

# Load ResNet settings
str_mixing = devkit_path+str(total_nb_cl)+'mixing.pickle'
with open(str_mixing,'rb') as fp:
    mixing = cPickle.load(fp)

str_settings_resnet = devkit_path+ str(total_nb_cl)+'settings_resnet.pickle'
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
str_class_means = devkit_path+str(total_nb_cl)+'class_means.pickle'
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

inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(indexs_of_files, files_from_cl, gpu, itera, batch_size, traind, labels_dic, mixing, nb_groups, total_nb_cl, save_path, trainl, labels_from_cl,keep_prob) 

label_batch_one_hot = tf.one_hot(label_batch, 10)

correct_pred = tf.equal(tf.argmax(scores,1), tf.argmax(label_batch_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def sm(arr):
  num = np.exp(arr) ;
  den = num.sum() ;
  return num/den ;

def test_cb(self):
  global testit ;
  print(testit)
  ax1.cla();
  ax2.cla();
  ax3.cla();
  ax1.imshow(files_from_cl[b+testit].reshape(28,28),cmap=plt.get_cmap("bone")) ;
  confs =sm(sc[testit]) ;
  ax2.bar(range(0,10),confs);
  ax2.set_ylim(0,1.)
  ce = -(confs*np.log(confs+0.00000001)).sum() ;
  ax3.text(0.5,0.5,str(ce),fontsize=20)
  testit = testit + 1;
  f.canvas.draw();
  print(b, 'value of b')
  print ("--------------------") ;
  print("logits", sc[testit], "probabilities", sm(sc[testit]), "decision", sc[testit].argmax(), "label", labels_from_cl[b].argmax()) ;



with tf.Session(config=config) as sess:
    # Launch the prefetch system
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(inits)
    
    # Evaluation routine
    stat_hb1     = []
    stat_icarl = []
    stat_ncm     = []
    b = 0
    lbl_list = []
    sc_list = []
    icarl_list = []
    ncm_list = []

    for i in range(int(np.ceil(len(files_from_cl)/batch_size))):
        acc, sc, l , loss,files_tmp,feat_map_tmp = sess.run([accuracy,scores, label_batch,loss_class,file_string_batch,op_feature_map])
        lbl_list.extend(l)
        sc_list.extend(np.argmax(sc,1))
        
        
        mapped_prototypes = feat_map_tmp[:,0,0,:]

        pred_inter    = (mapped_prototypes.T)/np.linalg.norm(mapped_prototypes.T,axis=0)
        sqd_icarl     = -cdist(class_means[:,:,0,itera].T, pred_inter.T, 'sqeuclidean').T
        sqd_ncm       = -cdist(class_means[:,:,1,itera].T, pred_inter.T, 'sqeuclidean').T

        icarl_list.extend(np.argmax(sqd_icarl,1))
        ncm_list.extend(np.argmax(sqd_ncm,1))

        stat_hb1     += ([ll in best for ll, best in zip(l, np.argsort(sc, axis=1)[:, -top:])])
        stat_icarl   += ([ll in best for ll, best in zip(l, np.argsort(sqd_icarl, axis=1)[:, -top:])])
        stat_ncm     += ([ll in best for ll, best in zip(l, np.argsort(sqd_ncm, axis=1)[:, -top:])])

        # testit = 0 ;    
        # f,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3) ;
        # f.canvas.mpl_connect('button_press_event', test_cb)
        # plt.show();
        # b = b + 128


    skplt.metrics.plot_confusion_matrix(lbl_list, icarl_list, normalize=True, title="iCaRL")
    skplt.metrics.plot_confusion_matrix(lbl_list, sc_list, normalize=True, title="Hybrid-1")
    skplt.metrics.plot_confusion_matrix(lbl_list, ncm_list, normalize=True, title="NCM")
    
    
  
    coord.request_stop()
    coord.join(threads)

print('Increment: %i' %itera)
print('Hybrid 1 top '+str(top)+' accuracy: %f' %np.average(stat_hb1))
print('iCaRL top '+str(top)+' accuracy: %f' %np.average(stat_icarl))
print('NCM top '+str(top)+' accuracy: %f' %np.average(stat_ncm))
acc_list[itera,0] = np.average(stat_icarl)
acc_list[itera,1] = np.average(stat_hb1)
acc_list[itera,2] = np.average(stat_ncm)
plt.show()
# Reset the graph to compute the numbers ater the next increment
tf.reset_default_graph()


np.save('results_top'+str(top)+'_acc_cl'+str(total_nb_cl),acc_list)

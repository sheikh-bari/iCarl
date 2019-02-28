import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import scipy
import os
import scipy.io
import sys
import gzip
import matplotlib.pyplot as plt
from matplotlib.widgets import Button ;
import math ;
from tensorflow import keras
try:
    import cPickle
except:
    import _pickle as cPickle
# Syspath for the folder with the utils files
#sys.path.insert(0, "/media/data/srebuffi")

import utils_resnet
import utils_icarl
import utils_data
import alexnet

# with gzip.open('mnist.pkl.gz', 'rb') as f:
#     ((traind, trainl), (vald, vall), (testd, testl)) = cPickle.load(f,encoding="latin-1")
#     traind = traind.astype("float32").reshape(-1,784)
#     trainl = trainl.astype("float32")
#     testd = testd.astype("float32").reshape(-1,784)
#     testl = testl.astype("float32")

fashion_mnist = keras.datasets.fashion_mnist
((traind, trainlabel), (testd, testlabel)) = fashion_mnist.load_data()
traind = traind.astype("float32").reshape(-1,784)
trainlabel = trainlabel.astype("uint8")

trainl = np.zeros((60000, 10))
trainl[np.arange(60000), trainlabel] = 1

testd = testd.astype("float32").reshape(-1,784)
testlabel = testlabel.astype("uint8")

testl = np.zeros((10000, 10))
testl[np.arange(10000), testlabel] = 1


# for i in range(10):

#   plt.imshow(traind[i].reshape(28,28))
#   plt.show()
#   print(trainl[i])
# exit()

data_placeholder2 = tf.placeholder(tf.float32,[None,784],  name='data_placeholder2') ;
label_placeholder2 = tf.placeholder(tf.float32,[None,10], name='label_placeholder2') ;



######### Modifiable Settings ##########
batch_size = 128            # Batch size
nb_val     = 200             # Validation samples per class
nb_cl      = [1,1,1,1,1,1,1,1,1,1]             # Classes per group
total_nb_cl = 10
nb_groups  = 10            # Number of groups
nb_proto   = 100             # Number of prototypes per class: total protoset memory/ total number of classes
epochs     = 10             # Total number of epochs 
lr_old     = 0.1             # Initial learning rate
lr_strat   = [20,30,40]  # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
gpu        = '0'            # Used GPU
wght_decay = 0.00001        # Weight Decay
#dropout    = 0.8            # Dropout, probability to keep units
########################################

######### Paths  ##########
# Working station 
devkit_path = ''
#train_path  = '../../../images1'
save_path   = 'result/'

###########################

#####################################################################################################

### Initialization of some variables ###
class_means    = np.zeros((2048,total_nb_cl,2,nb_groups))
loss_batch     = []
files_protoset = []
labels_protoset = []
for _ in range(total_nb_cl):
    files_protoset.append([])
    labels_protoset.append([])


### Preparing the files for the training/validation ###

# Random mixing
print("Mixing the classes and putting them in batches of classes...")
np.random.seed(1993)
order  = np.arange(total_nb_cl)
mixing = np.arange(total_nb_cl)
#np.random.shuffle(mixing)

# Loading the labels
#labels_dic, label_names, validation_ground_truth = utils_data.parse_devkit_meta(devkit_path)
# Or you can just do like this
define_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels_dic = {k: v for v, k in enumerate(define_class)}

# Preparing the files per group of classes
print("Creating a validation set ...")
#files_train, files_valid = utils_data.prepare_files(train_path, mixing, order, labels_dic, nb_groups, nb_cl, nb_val)
files_train, files_valid, file_labels, file_indexes, labels_valid, all_file_indexes = utils_data.prepare_data(traind, trainl, mixing, order, labels_dic, nb_groups, nb_cl, nb_val, testd, testl)


# Pickle order and files lists and mixing
with open(str(total_nb_cl)+'mixing.pickle','wb') as fp:
    cPickle.dump(mixing,fp)

with open(str(total_nb_cl)+'settings_resnet.pickle','wb') as fp:
    cPickle.dump(order,fp)
    cPickle.dump(files_valid,fp)
    cPickle.dump(files_train,fp)
    cPickle.dump(file_labels,fp)
    cPickle.dump(file_indexes,fp)
    cPickle.dump(labels_valid,fp)
    cPickle.dump(all_file_indexes,fp)


### Start of the main algorithm ###

for itera in range(nb_groups):
  
  # Files to load : training samples + protoset
  print('Batch of classes number {0} arrives ...'.format(itera+1))
  # Adding the stored exemplars to the training set
  if itera == 0:
    files_from_cl = files_train[itera]
    labels_from_cl = file_labels[itera]
    indexs_of_files = file_indexes[itera]
    testFiles       = np.asarray(files_valid[itera])
    testLabels      = np.asarray(labels_valid[itera])
  else:

    files_from_cl = files_train[itera][:]
    labels_from_cl = file_labels[itera][:]
    indexs_of_files = file_indexes[itera][:]

    print('itera', itera)
    for i in range(sum(nb_cl[:itera])):
      nb_protos_cl  = int(np.ceil(nb_proto*nb_groups*1./itera)) # Reducing number of exemplars of the previous classes
      tmp_var = files_protoset[i]
      files_from_cl   = np.vstack((files_from_cl,traind[tmp_var[0:min(len(tmp_var),nb_protos_cl)]]))
      #files_from_cl   += tmp_var[0:min(len(tmp_var),nb_protos_cl)]
      labels_from_cl  = np.vstack((labels_from_cl,trainl[tmp_var[0:min(len(tmp_var),nb_protos_cl)]]))

    # if(itera == 9):
    #   for k in range(int(np.floor(len(files_from_cl)/100))):
    #     plt.imshow(files_from_cl[k*100].reshape(28,28))
    #     plt.show()
    #     print(np.argmax(labels_from_cl[k*100]))

  print('shape of labells:',np.asarray(labels_from_cl).shape)

  ## Import the data reader ##
  #image_train, label_train   = utils_data.read_data(train_path, labels_dic, mixing, files_from_cl=files_from_cl)  
  image_train, label_train   = utils_data.read_data_mnist(traind, trainl, labels_dic, labels_from_cl, mixing, files_from_cl=files_from_cl)

  image_batch, label_batch_0 = tf.train.batch([image_train, label_train], batch_size=batch_size, num_threads=8)

  label_batch = tf.one_hot(label_batch_0,total_nb_cl)

  ## Define the objective for the neural network ##
  if itera == 0:
    keep_prob = tf.placeholder(name="keep_prob", dtype=tf.float32)
    # No distillation
    variables_graph,variables_graph2,scores,scores_stored = utils_icarl.prepare_networks(gpu,image_batch, total_nb_cl, nb_groups, keep_prob)

    # Define the objective for the neural network: 1 vs all cross_entropy
    with tf.device('/cpu:0'):
        scores        = tf.concat(scores,0)

        l2_reg        = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))

        loss_class    = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=scores)) 

        ## classification accuracy
        #nrCorrect     = tf.reduce_mean(tf.cast(tf.equal (tf.argmax(scores,axis=1), tf.argmax(label_batch,axis=1)), tf.float32)) ;
        loss          = loss_class + l2_reg

        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt           = tf.train.MomentumOptimizer(learning_rate, 0.9)
        #opt           = tf.train.GradientDescentOptimizer(learning_rate = 0.1)

        train_step    = opt.minimize(loss,var_list=variables_graph)

        label_placeholder = tf.placeholder(tf.float32,[None,10], name='lbl1') ;
        #data_placeholder = tf.placeholder(tf.float32,[None,784], name='dp1') ;
       

        correct_pred = tf.equal(tf.argmax(scores,1), tf.argmax(label_batch,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  if itera > 0:
    # Distillation
    keep_prob = tf.placeholder(name="keep_prob2", dtype=tf.float32)

    data_placeholder2 = tf.placeholder(tf.float32,[None,784],  name='data_placeholder1') ; #image_batch
    label_placeholder2 = tf.placeholder(tf.float32,[None,10], name='label_placeholder2') ; #label_old_classes
    label_placeholder = tf.placeholder(tf.float32,[None,5], name='label_placeholder1') ; #label_new_classes

    variables_graph,variables_graph2,scores,scores_stored = utils_icarl.prepare_networks(gpu,image_batch, total_nb_cl, nb_groups, keep_prob)
    
    # Copying the network to use its predictions as ground truth labels
    op_assign = [(variables_graph2[i]).assign(variables_graph[i]) for i in range(len(variables_graph))]
    
    # Define the objective for the neural network : 1 vs all cross_entropy + distillation
    with tf.device('/cpu:0'):
      #label_placeholder = tf.placeholder(tf.float32,[None,10], name='lblplace2') ;
      #data_placeholder = tf.placeholder(tf.float32,[None,784], name='dp2') ;
      

      scores            = tf.concat(scores,0)
      scores_stored     = tf.concat(scores_stored,0)
      old_cl            = (order[range(sum(nb_cl[:itera]))]).astype(np.int32)
      new_cl            = (order[range(sum(nb_cl[:itera]),sum(nb_cl[:itera+1]))]).astype(np.int32)
      label_old_classes = tf.sigmoid(tf.stack([scores_stored[:,i] for i in old_cl],axis=1))
      label_new_classes = tf.stack([label_batch[:,i] for i in new_cl],axis=1)
      pred_old_classes  = tf.stack([scores[:,i] for i in old_cl],axis=1)
      pred_new_classes  = tf.stack([scores[:,i] for i in new_cl],axis=1)
      l2_reg            = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
      loss_class        = tf.reduce_mean(tf.concat([tf.nn.sigmoid_cross_entropy_with_logits(labels=label_old_classes, logits=pred_old_classes),tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes, logits=pred_new_classes)],1)) 
      loss              = loss_class + l2_reg
      learning_rate     = tf.placeholder(tf.float32, shape=[])
      #opt               = tf.train.MomentumOptimizer(learning_rate, 0.9)
      opt               = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
      train_step        = opt.minimize(loss,var_list= variables_graph)

      correct_pred = tf.equal(tf.argmax(scores,1), tf.argmax(label_batch,1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


  ## Run the learning phase ##
  with tf.Session(config=config) as sess:
    # Launch the data reader 
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    lr      = lr_old

    # Run the loading of the weights for the learning network and the copy network
    if itera > 0:
      void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
      void1 = sess.run(op_assign)

    for epoch in range(epochs):
        print("Batch of classes {} out of {} batches".format(
                itera + 1, nb_groups))
        print('Epoch %i' % epoch)
        print(int(np.ceil(len(files_from_cl)/batch_size)))

        for i in range(int(np.ceil(len(files_from_cl)/batch_size))):
            # lbl_batch1, imgBatch = sess.run([label_batch,image_batch])

            # if(itera == 0):
              
              #fd = {label_placeholder2:lbl_batch1, data_placeholder2:imgBatch, learning_rate: lr, keep_prob: dropout}
              #fd1 = { data_placeholder2: testFiles, label_placeholder2: testLabels, keep_prob: 1.}
            # else:
              
            #   lblNewClasses, imgBatch = sess.run([label_new_classes, image_batch])
            #   fd = {label_placeholder:lblNewClasses, data_placeholder2:imgBatch, learning_rate: lr, keep_prob: dropout}
            #   fd1 = {label_placeholder:lblNewClasses, data_placeholder2: testFiles, label_placeholder2: testLabels, keep_prob: 1.}

            acc, loss_class_val, _ ,sc, lab = sess.run([accuracy, loss_class, train_step,scores, label_batch_0], feed_dict={learning_rate: lr})
            loss_batch.append(loss_class_val)
            # Plot the training error every 10 batches
            if len(loss_batch) == 30:
                #print('training error: loss: ',np.mean(loss_batch))
                loss_batch = []

            # Plot the training top 1 accuracy every 80 batches
            if (i+1)%15 == 0:
                stat = []
                stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
                stat =np.average(stat)
                print('Training accuracy %f' %stat)
    
        #testacc = sess.run(nrCorrect, feed_dict = {data_placeholder: testd, label_placeholder: testl})
        #print('my test accuracy %f' %testacc)
        # Decrease the learning by 5 every 10 epoch after 20 epochs at the first learning rate
        if epoch in lr_strat:
            lr /= lr_factor
    
    #acc = sess.run(accuracy, feed_dict=fd1)
    #print('my trainig accuracy', acc)
    coord.request_stop()
    coord.join(threads)

    # copy weights to store network
    save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])
    alexnet.save_model(save_path+'model-iteration'+str(total_nb_cl)+'-%i.pickle' % itera, scope='ResNet18', sess=sess)

    # testout = sess.run(scores, feed_dict = {data_placeholder2 : testFiles}) ;
    # print(testout, 'testout')
    # testit = 0 ;


    # def sm(arr):
    #   num = np.exp(arr) ;
    #   den = num.sum() ;
    #   return num/den ;

    # def test_cb(self):
    #   global testit ;
    #   ax1.cla();
    #   ax2.cla();
    #   ax3.cla();
    #   ax1.imshow(testFiles[testit].reshape(28,28),cmap=plt.get_cmap("bone")) ;
    #   confs =sm(testout[testit]) ;
    #   ax2.bar(range(0,10),confs);
    #   ax2.set_ylim(0,1.)
    #   ce = -(confs*np.log(confs+0.00000001)).sum() ;
    #   ax3.text(0.5,0.5,str(ce),fontsize=20)
    #   testit = testit + 1;
    #   f.canvas.draw();
    #   print ("--------------------") ;
    #   print("logits", testout[testit], "probabilities", sm(testout[testit]), "decision", testout[testit].argmax(), "label", testLabels[testit].argmax()) ;


    # f,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3) ;
    # f.canvas.mpl_connect('button_press_event', test_cb)
    # plt.show();




  # Reset the graph 
  tf.reset_default_graph()
  
  ## Exemplars management part  ##
  nb_protos_cl  = int(np.ceil(nb_proto*nb_groups*1./(itera+1))) # Reducing number of exemplars for the previous classes
  files_from_cl = files_train[itera]
  indexs_of_files = file_indexes[itera]
  labels_from_cl = file_labels[itera]
  #inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(files_from_cl, gpu, itera, batch_size, train_path, labels_dic, mixing, nb_groups, nb_cl, save_path)

  inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(indexs_of_files, files_from_cl, gpu, itera, batch_size, traind, labels_dic, mixing, nb_groups, total_nb_cl, save_path, trainl, labels_from_cl, keep_prob)

  with tf.Session(config=config) as sess:
    

    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    void3   = sess.run(inits)
    
    # Load the training samples of the current batch of classes in the feature space to apply the herding algorithm
    Dtot,processed_files,label_dico = utils_icarl.load_class_in_feature_space(files_from_cl, batch_size, scores, label_batch, loss_class, file_string_batch, op_feature_map, sess)

    #processed_files = np.array([x.decode() for x in processed_files])
    
    # Herding procedure : ranking of the potential exemplars
    print('Exemplars selection starting ...')
    for iter_dico in range(nb_cl[itera]):

        ind_cl     = np.where(label_dico == order[iter_dico+sum(nb_cl[:itera])])[0]
        D          = Dtot[:,ind_cl]
        
        files_iter = processed_files[ind_cl]
        labels_iter = label_dico[ind_cl]
        mu         = np.mean(D,axis=1)
        w_t        = mu
        step_t     = 0
        while not(len(files_protoset[sum(nb_cl[:itera])+iter_dico]) == nb_protos_cl) and step_t<1.1*nb_protos_cl:
            tmp_t   = np.dot(w_t,D)
            ind_max = np.argmax(tmp_t)
            w_t     = w_t + mu - D[:,ind_max]
            step_t  += 1
            if files_iter[ind_max] not in files_protoset[sum(nb_cl[:itera])+iter_dico]:
              files_protoset[sum(nb_cl[:itera])+iter_dico].append(files_iter[ind_max])
              labels_protoset[sum(nb_cl[:itera])+iter_dico].append(labels_iter[ind_max])

    coord.request_stop()
    coord.join(threads)

  # Reset the graph
  tf.reset_default_graph()
  
  # Class means for iCaRL and NCM 
  print('Computing theoretical class means for NCM and mean-of-exemplars for iCaRL ...')
  for iteration2 in range(itera+1):
      files_from_cl = files_train[iteration2]
      labels_from_cl = file_labels[iteration2]
      indexs_of_files = file_indexes[iteration2]
      inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(indexs_of_files, files_from_cl, gpu, itera, batch_size, traind, labels_dic, mixing, nb_groups, total_nb_cl, save_path, trainl, labels_from_cl, keep_prob)
      
      with tf.Session(config=config) as sess:
          coord   = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(coord=coord)
          void2   = sess.run(inits)
          
          Dtot,processed_files,label_dico = utils_icarl.load_class_in_feature_space(files_from_cl, batch_size, scores, label_batch, loss_class, file_string_batch, op_feature_map, sess)
          #processed_files = np.array([x.decode() for x in processed_files])
          
          for iter_dico in range(nb_cl[iteration2]):
              ind_cl     = np.where(label_dico == order[iter_dico+sum(nb_cl[:iteration2])])[0]
              D          = Dtot[:,ind_cl]
              files_iter = processed_files[ind_cl]
              labels_iter= label_dico[ind_cl]
              current_cl = order[range(sum(nb_cl[:iteration2]),sum(nb_cl[:iteration2+1]))]

              # Normal NCM mean
              class_means[:,order[iter_dico+sum(nb_cl[:iteration2])],1,itera] = np.mean(D,axis=1)
              class_means[:,order[iter_dico+sum(nb_cl[:iteration2])],1,itera] /= np.linalg.norm(class_means[:,order[iter_dico+sum(nb_cl[:iteration2])],1,itera])
              
              # iCaRL approximated mean (mean-of-exemplars)
              # use only the first exemplars of the old classes: nb_protos_cl controls the number of exemplars per class
              ind_herding = np.array([np.where(files_iter == files_protoset[iter_dico+sum(nb_cl[:iteration2])][i])[0][0] for i in range(min(nb_protos_cl,len(files_protoset[iter_dico+sum(nb_cl[:iteration2])])))])
              D_tmp       = D[:,ind_herding]
              class_means[:,order[iter_dico+sum(nb_cl[:iteration2])],0,itera] = np.mean(D_tmp,axis=1)
              class_means[:,order[iter_dico+sum(nb_cl[:iteration2])],0,itera] /= np.linalg.norm(class_means[:,order[iter_dico+sum(nb_cl[:iteration2])],0,itera])

          coord.request_stop()
          coord.join(threads)

      # Reset the graph
      tf.reset_default_graph()
  
  # Pickle class means and protoset
  with open(str(nb_cl)+'class_means.pickle','wb') as fp:
      cPickle.dump(class_means,fp)
  with open(str(nb_cl)+'files_protoset.pickle','wb') as fp:
      cPickle.dump(files_protoset,fp)



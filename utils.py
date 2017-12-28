from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import csv
import copy
try:
	import xml.etree.cElementTree as ET
except ImportError:
	import xml.etree.ElementTree as ET

import os
from os import listdir
from os.path import isfile, join
import sys
import time

import argparse
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import Model

#=========#
#   Run   #
#=========#
def run_training(
    Hyperparameter       , 
    # Info               
    FLAGs                ,
    Epoch                ,
    IS_HYPERPARAMETER_OPT,
    # Path               
    Dataset_Path         ,
    Y_pre_Path           ,
    trained_model_path   ,
    trained_model        
    ):

    Dataset           = FLAGs.Dataset
    Model_first_name  = FLAGs.Model_1st
    Model_second_name = FLAGs.Model_2nd
    EPOCH             = Epoch
    BATCH_SIZE        = FLAGs.BatchSize
    
    Model_Name = Model_first_name + '_' + Model_second_name
    
    #---------------#
    #   Data Info   #
    #---------------#
    if Dataset=='CamVid':   # original : [360, 480, 12]
        H_Resize = 360
        W_Resize = 480
        class_num = 12
    elif Dataset=='ade20k': # original : All Different
        H_Resize = 224
        W_Resize = 224
        class_num = 151
    elif Dataset=='mnist':  # original : [28, 28, 10]
        H_Resize = 32
        W_Resize = 32
        class_num = 10
    elif Dataset=='cifar10': # original : [28, 28, 10]
        H_Resize = 32
        W_Resize = 32
        class_num = 10
        
    #----------------#
    #   Model_dict   #
    #----------------#
    Model_dict = Model_dict_Generator('Model/' + Model_first_name + '_Model/' + Model_Name + '.csv', class_num)
    
    #-----------------------------------#
    #   Hyperparameter : User Defined   #
    #-----------------------------------#
    if not IS_HYPERPARAMETER_OPT:
        HP = {}
        HP.update({'Batch_Size'                : BATCH_SIZE})
        HP.update({'Epoch'                     : EPOCH     })
        HP.update({'H_Resize'                  : H_Resize  })
        HP.update({'W_Resize'                  : W_Resize  })
        HP.update({'LR'                        : 1e-2      })
        HP.update({'LR_Strategy'               : '2times'  })
        HP.update({'LR_Final'                  : 1e-3      })
        HP.update({'LR_Decade'                 : 10        })   
        HP.update({'LR_Decade_1st_Epoch'       : 100       })
        HP.update({'LR_Decade_2nd_Epoch'       : 150       })
        HP.update({'L2_Lambda'                 : 1e-5      })
        HP.update({'Opt_Method'                : 'Momentum'})
        HP.update({'Momentum_Rate'             : 0.9       })
        HP.update({'IS_STUDENT'                : False     })
        HP.update({'Ternary_Epoch'             : 50        })
        HP.update({'Quantized_Activation_Epoch': 100       })
        HP.update({'Dropout_Rate'              : 0.0       })
        
    #-----------------------------------#
    #   Hyperparameter : Optimization   #
    #-----------------------------------#
    else:
        HP_dict, Model_dict = Hyperparameter_Decoder(Hyperparameter, Model_dict)
        
        HP = {}
        HP.update({'Batch_Size'                : int(HP_dict['Batch_Size'])                })
        HP.update({'Epoch'                     : EPOCH                                     })
        HP.update({'H_Resize'                  : H_Resize                                  })
        HP.update({'W_Resize'                  : W_Resize                                  })
        HP.update({'LR'                        : float(HP_dict['LR'])                      })
        HP.update({'LR_Decade'                 : int(HP_dict['LR_Decade'])                 })
        HP.update({'LR_Strategy'               : int(HP_dict['LR_Strategy'])               })
        HP.update({'LR_Final'                  : int(HP_dict['LR_Final'])                  }) 
        HP.update({'LR_Decade_1st_Epoch'       : int(HP_dict['LR_Decade_1st_Epoch'])       })
        HP.update({'LR_Decade_2nd_Epoch'       : int(HP_dict['LR_Decade_2nd_Epoch'])       })
        HP.update({'L2_Lambda'                 : float(HP_dict['L2_Lambda'])               })
        HP.update({'Opt_Method'                : HP_dict['Opt_Method']                     })
        HP.update({'Momentum_Rate'             : float(HP_dict['Momentum_Rate'])           })
        HP.update({'IS_STUDENT'                : HP_dict['IS_STUDENT'] == 'TRUE'           })
        HP.update({'Ternary_Epoch'             : int(HP_dict['Ternary_Epoch'])             })
        HP.update({'Quantized_Activation_Epoch': int(HP_dict['Quantized_Activation_Epoch'])})
        HP.update({'Dropout_Rate'              : float(HP_dict['Dropout_Rate'])            })
    
    print("\033[1;32;40mBATCH SIZE\033[0m : \033[1;37;40m{}\033[0m" .format(HP['Batch_Size']))
    
    #---------------------------#
    #    Hyperparameter Save    #
    #---------------------------#
    if not IS_HYPERPARAMETER_OPT:
        components = np.array(['Batch_Size'                ,
                               'Epoch'                     ,
                               'H_Resize'                  ,
                               'W_Resize'                  ,
                               'LR'                        ,
                               'LR_Strategy'               ,
                               'LR_Final'                  ,
                               'LR_Decade'                 ,
                               'LR_Decade_1st_Epoch'       ,
                               'LR_Decade_2nd_Epoch'       ,
                               'L2_Lambda'                 ,
                               'Opt_Method'                ,
                               'Momentum_Rate'             ,
                               'IS_STUDENT'                ,
                               'Ternary_Epoch'             ,
                               'Quantized_Activation_Epoch',
                               'Dropout_Rate'              ])     
        
        for iter, component in enumerate(components):
            if iter == 0:
                HP_csv = np.array([HP[component]])
            else:
                HP_csv = np.concatenate([HP_csv, np.array([HP[component]])], axis=0)
                
        components = np.expand_dims(components, axis=1)
        HP_csv = np.expand_dims(HP_csv, axis=1)
        HP_csv = np.concatenate([HP_csv, components], axis=1)
    else:
        HP_csv = None
    
    #----------------#
    #    Training    #
    #----------------#
    TRAINED_WEIGHT_FILE = None

    Model_Path, Model = Training( 
        Model_dict            = Model_dict            ,               
        Dataset               = Dataset               ,
        Dataset_Path          = Dataset_Path          ,
        Y_pre_Path            = Y_pre_Path            ,
        class_num             = class_num             ,            
        HP                    = HP                    ,
        weights_bd_ratio      = 50                    ,
        biases_bd_ratio       = 50                    ,         
        HP_csv                = HP_csv                ,
        Model_first_name      = Model_first_name      ,
        Model_second_name     = Model_second_name     ,
        trained_model_path    = trained_model_path    ,
        trained_model         = trained_model         ,
        IS_HYPERPARAMETER_OPT = IS_HYPERPARAMETER_OPT )
    return Model_Path, Model
        
def run_testing(
    Hyperparameter       ,
    # Info               
    FLAGs                ,
    IS_HYPERPARAMETER_OPT,
    # Path               
    Dataset_Path         ,
    testing_model_path   ,
    testing_model        ,
    train_target_path    ,
    valid_target_path    ,
    test_target_path     ,
    train_Y_pre_path     ,
    valid_Y_pre_path     ,
    test_Y_pre_path         
    ):
    
    Dataset           = FLAGs.Dataset
    Model_first_name  = FLAGs.Model_1st
    Model_second_name = FLAGs.Model_2nd
    BATCH_SIZE        = FLAGs.BatchSize
    
    #---------------#
    #   Data Info   #
    #---------------#
    if Dataset=='CamVid':   # original : [360, 480, 12]
        H_Resize = 360
        W_Resize = 480
        class_num = 12
    elif Dataset=='ade20k': # original : All Different
        H_Resize = 224
        W_Resize = 224
        class_num = 151
    elif Dataset=='mnist':  # original : [28, 28, 10]
        H_Resize = 28
        W_Resize = 28
        class_num = 10
    elif Dataset=='cifar10': # original : [28, 28, 10]
        H_Resize = 32
        W_Resize = 32
        class_num = 10
    
    #----------------#
    #   Model_dict   #
    #----------------#
    Model_dict = Model_dict_Generator(testing_model_path + 'model.csv', class_num)
    
    #------------------------------#
    #   Hyperparameter : Testing   #
    #------------------------------#
    """
    HP = {}
    with open(testing_model_path + 'HP.csv') as csvfile:
        HPreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for iter, row in enumerate(HPreader):
            HP.update({row[0]: row[1]})
    """
    
    #---------------#
    #    Testing    #
    #---------------#
    test_accuracy = Testing( 
        Dataset                    = Dataset               ,
        Dataset_Path               = Dataset_Path          ,
        class_num                  = class_num             ,
        H_Resize                   = H_Resize              ,
        W_Resize                   = W_Resize              ,  
        BATCH_SIZE                 = BATCH_SIZE            ,
        Model_dict                 = Model_dict            ,
        IS_SAVING_RESULT_AS_IMAGE  = False                 ,
        IS_SAVING_RESULT_AS_NPZ    = False                 ,
        IS_HYPERPARAMETER_OPT      = IS_HYPERPARAMETER_OPT ,
        IS_TRAIN                   = False                 ,
        IS_VALID                   = False                 ,
        IS_TEST                    = True                  ,
        testing_model_path         = testing_model_path    ,
        testing_model              = testing_model         ,
        train_target_path          = train_target_path     ,
        valid_target_path          = valid_target_path     ,
        test_target_path           = test_target_path      ,
        train_Y_pre_path           = train_Y_pre_path      ,
        valid_Y_pre_path           = valid_Y_pre_path      ,
        test_Y_pre_path            = test_Y_pre_path       )
    
    return test_accuracy
    
def Training( 
    # Model
    Model_dict           ,
    # Dataset            
    Dataset	             ,
    Dataset_Path         ,
    Y_pre_Path           ,
    class_num            ,
    # Parameter	
    HP                   ,
    weights_bd_ratio     ,
    biases_bd_ratio      ,
    # Model Save         
    HP_csv               ,
    Model_first_name     ,
    Model_second_name    ,
    # Pre-trained Weights
    trained_model_path   ,
    trained_model        ,
    # Hyperparameter Opt
    IS_HYPERPARAMETER_OPT
    ):
    
    Model_Name = Model_first_name + '_' + Model_second_name
    
    #---------------#
    #    Dataset    #
    #---------------#
    print("Parsing Data ... ")
    filenames = [os.path.join(Dataset_Path, 'data_batch_%d.bin'%(i+1)) for i in range(5)]
    xs, ys = input_fn(filenames, class_num, True, HP)
    train_data_index   = np.array(open(Dataset_Path + '/train.txt', 'r').read().splitlines())
    train_target_index = np.array(open(Dataset_Path + '/trainannot.txt', 'r').read().splitlines())
    train_data_num = len(train_data_index)               

    print("\033[0;32mTrain Data Number\033[0m : {}" .format(train_data_num))

    #------------------#
    #    Placeholder   #
    #------------------#
    data_shape = [None, HP['H_Resize'], HP['W_Resize'], 3]
    global_step = tf.train.get_or_create_global_step()
    batches_per_epoch = train_data_num / HP['Batch_Size']
    
    ## Is_training
    is_training = tf.placeholder(tf.bool)
    ## Learning Rate
    initial_learning_rate = HP['LR']
    boundaries = [int(batches_per_epoch * epoch) for epoch in [HP['LR_Decade_1st_Epoch'], HP['LR_Decade_2nd_Epoch'], 200]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
    ## is_quantized_activation
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
    ## is_ternary
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})  

    #----------------------#
    #    Building Model    #
    #----------------------# 
    print("Building Model ...")
    device = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    device_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    device_batch = HP['Batch_Size'] / device_num
    prediction_list = []
    # Assign Usage of CPU and GPU
    for d in range(device_num):
        device_use = device[d]
        with tf.device(tf.train.replica_device_setter(worker_device = '/device:GPU:%d' %int(device_use), ps_device = '/device:CPU:0', ps_tasks=1)):
            # -- Build Model --
            net = xs[ d*device_batch : (d+1)*device_batch ]
            Model_dict_ = copy.deepcopy(Model_dict)
            if d == 0:
                model = cifar10_resnet_v2_generator(resnet_size, num_classes, data_format=None)
                prediction = model(net, True)
                """
                prediction, Analysis, max_parameter, inputs_and_kernels = Model_dict_Decoder(
                    net                     = net, 
                    Model_dict              = Model_dict_, 
                    is_training             = is_training,
                    is_ternary              = is_ternary,
                    is_quantized_activation = is_quantized_activation,
                    DROPOUT_RATE            = HP['Dropout_Rate'],
                    reuse                   = None)
                """
            else:
                prediction, Analysis, max_parameter, inputs_and_kernels = Model_dict_Decoder(
                    net                     = net, 
                    Model_dict              = Model_dict_, 
                    is_training             = is_training,
                    is_ternary              = is_ternary,
                    is_quantized_activation = is_quantized_activation,
                    DROPOUT_RATE            = HP['Dropout_Rate'],
                    reuse                   = True)
            
            #prediction = tf.squeeze(prediction, [1, 2])
            prediction_list.append(prediction)
            
            # -- Model Size --
            if d == 0:
                # Your grade will depend on this value.
                all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model")
                # Model Size
                Model_Size = 0
                for iter, variable in enumerate(all_variables):
                    Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
                    # See all your variables in termainl	
                    """
                    print(variable)
                    """
                print("\033[0;36m=======================\033[0m")
                print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size))
                print("\033[0;36m=======================\033[0m")
            
            # -- Collection --
            if d == 0:
                ## Ternary
                float32_weights_collection          = tf.get_collection("float32_weights"        , scope=None)
                float32_biases_collection           = tf.get_collection("float32_biases"         , scope=None)
                ternary_weights_bd_collection       = tf.get_collection("ternary_weights_bd"     , scope=None)
                ternary_biases_bd_collection        = tf.get_collection("ternary_biases_bd"      , scope=None)  
                ## assign ternary or float32 weights/biases to final weights/biases  
                assign_var_list_collection          = tf.get_collection("assign_var_list"        , scope=None)  
                ## Actvation Quantization    
                float32_net_collection              = tf.get_collection("float32_net"            , scope=None)
                is_quantized_activation_collection  = tf.get_collection("is_quantized_activation", scope=None)
                mantissa_collection                 = tf.get_collection("mantissa"               , scope=None)
                fraction_collection                 = tf.get_collection("fraction"               , scope=None)
                ## Gradient Update
                var_list_collection                 = tf.get_collection("var_list"               , scope=None)
                float32_params                      = tf.get_collection("float32_params"         , scope=None) 

            # -- Loss --
            labels = ys[ d*device_batch : (d+1)*device_batch ]
            # KL divergence
            KL = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = prediction_list[d])
            # Corss Entopy
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels, -1), logits=prediction_list[d])
            # L2 Regularization
            l2_norm   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            l2_lambda = tf.constant(HP['L2_Lambda'])
            l2_norm   = tf.multiply(l2_lambda, l2_norm)
            
            # Loss
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels = labels,
                logits        = prediction_list[d])
            loss = tf.add(loss, l2_norm)
            
            # -- Optimizer --
            if HP['Opt_Method']=='Adam':
                opt = tf.train.AdamOptimizer(learning_rate, HP['Momentum_Rate'])
            elif HP['Opt_Method']=='Momentum':
                opt = tf.train.MomentumOptimizer(
                    learning_rate = learning_rate, 
                    momentum      = HP['Momentum_Rate'])
                
            # Batch norm requires update ops to be added as a dependency to the train_op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Compute Gradients
                gradients = opt.compute_gradients(loss, var_list = var_list_collection)
                if d == 0:
                    gra_and_var = [(gradients[i][0], float32_params[i]) for i in range(np.shape(gradients)[0])]
                else:
                    for i in range(np.shape(gradients)[0]):
                        gra_and_var.append((gradients[i][0], float32_params[i]))
                
    # Apply Gradients
    with tf.device(tf.train.replica_device_setter(worker_device = '/device:GPU:%d' %int(device[0]), ps_device = '/device:CPU:0', ps_tasks=1)):
        train_step  = opt.apply_gradients(gra_and_var, global_step)
    
    #-----------------------------#
    #   Some Control Parameters   #
    #-----------------------------#
    IS_TERNARY = False
    IS_QUANTIZED_ACTIVATION = False
    for layer in range(len(Model_dict)):
        if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
            IS_TERNARY = True
        if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
            IS_QUANTIZED_ACTIVATION = True
            
    TERNARY_NOW = False
    QUANTIZED_NOW = False
    
    #-----------#
    #   Saver   #
    #-----------#	
    saver = tf.train.Saver()
    
    #---------------#
    #    Session    #
    #---------------#
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = 256
    with tf.Session(config = config) as sess: 
        #----------------------#
        #    Initialization    #
        #----------------------#
        print("Initializing ...")
        init = tf.global_variables_initializer()
        sess.run(init)
        
        #--------------------------#
        #   Load trained weights   #
        #--------------------------#
        if trained_model_path!=None && trained_model!=None:
            print("Loading the trained weights ... ")
            print("\033[0;35m{}\033[0m" .format(trained_model_path + trained_model))
            save_path = saver.restore(sess, trained_model_path + trained_model)
        
        #-------------------------------#
        #   Loading Pre-trained Model   #
        #-------------------------------#
        if TRAINED_WEIGHT_FILE!=None:
            print("Loading Pre-trained weights ...")
            save_path = saver.load(sess, TRAINED_WEIGHT_FILE + ".ckpt")
            print(save_path)
            
        #-------------#
        #    Epoch    #
        #-------------#
        lr = HP['LR']
        tStart_All = time.time()       
        print("Training ... ")
        for epoch in range(HP['Epoch']):
            total_correct_num = 0
            total_error_num = 0
            Train_loss = 0
            iteration  = 0
            
            ## -- Shuffle Data --
            """
            shuffle_index = np.arange(train_data_num)
            np.random.shuffle(shuffle_index)
            train_data   = train_data  [shuffle_index]
            train_target = train_target[shuffle_index]
            """
            ## -- Quantizad Activation --
            if IS_QUANTIZED_ACTIVATION and (epoch+1)==HP['Quantized_Activation_Epoch']:
                batch_xs = train_data[0:HP['Batch_Size']]
                # Calculate Each Activation's appropriate mantissa and fractional bit
                m, f = quantized_m_and_f(float32_net_collection, is_quantized_activation_collection, xs, Model_dict, batch_xs, sess)	
                # Assign mantissa and fractional bit to the tensor
                assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
                # Start Quantize Activation
                QUANTIZED_NOW = True
            
            ## -- Ternary --
            if IS_TERNARY and (epoch+1)==HP['Ternary_Epoch']:
                # Calculate the ternary boundary of each layer's weights
                weights_bd, biases_bd, weights_table, biases_table = tenarized_bd(
                    float32_weights_collection,  
                    float32_biases_collection, 
                    weights_bd_ratio, 
                    biases_bd_ratio, 
                    sess)
                # assign ternary boundary to tensor
                assign_ternary_boundary(ternary_weights_bd_collection, ternary_biases_bd_collection, weights_bd, biases_bd, sess)
                # Start Quantize Activation
                TERNARY_NOW = True
                
            ## -- Set feed_dict --
            # train_step
            feed_dict_train = {}
            for layer in range(len(Model_dict)):
                feed_dict_train.update({is_ternary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_train.update({is_quantized_activation['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION']=='TRUE' and QUANTIZED_NOW}) 
                
            # Assign float32 or ternary weight and biases to final weights
            feed_dict_assign = {}
            for layer in range(len(Model_dict)):
                feed_dict_assign.update({is_ternary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})

            ## -- Training --
            tStart = time.time()
            total_batch_iter = int(train_data_num / HP['Batch_Size'])
            for batch_iter in range(total_batch_iter):
                iteration = iteration + 1
                # Train data in BATCH SIZE
                """
                batch_xs = train_data   [ batch_iter*HP['Batch_Size'] : (batch_iter+1)*HP['Batch_Size'] ]
                batch_ys = train_target [ batch_iter*HP['Batch_Size'] : (batch_iter+1)*HP['Batch_Size'] ]
                """
                # Run Training Step
                feed_dict_train.update({is_training: True}) # xs: batch_xs, ys: batch_ys, learning_rate: lr, 
                _, Loss, Prediction_list, L2_norm, batch_ys, Learning_Rate = sess.run(
                    [train_step, loss, prediction_list, l2_norm, ys, learning_rate], 
                    feed_dict = feed_dict_train)
                # Combine Different Device Prediction
                for d in range(len(Prediction_list)):
                    if d == 0:
                        Prediction = Prediction_list[d]
                    else:
                        Prediction = np.concatenate([Prediction, Prediction_list[d]], axis = 0)
                # Assign float32 or ternary weight and biases to final weights
                for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                    sess.run(assign_var_list, feed_dict = feed_dict_assign)
                # Result
                y_pre             = np.argmax(Prediction, -1)
                correct_num       = np.sum(np.equal(np.argmax(batch_ys, -1), y_pre) == True , dtype = np.float32)
                error_num         = np.sum(np.equal(np.argmax(batch_ys, -1), y_pre) == False, dtype = np.float32)
                batch_accuracy    = correct_num / (correct_num + error_num)
                total_correct_num = total_correct_num + correct_num
                total_error_num   = total_error_num + error_num
                Train_loss        = Train_loss + np.mean(Loss)  
                # Per Batch Size Info
                """
                print("\033[1;34;40mEpoch\033[0m : {}" .format(epoch))
                print("\033[1;34;40mData Iteration\033[0m : {}" .format(batch_iter*HP['Batch_Size']))
                print("\033[1;32;40m  Batch Accuracy\033[0m : {}".format(batch_accuracy))
                print("\033[1;32;40m  Loss          \033[0m : {}".format(np.mean(Loss)))
                print("\033[1;32;40m  Learning Rate \033[0m : {}".format(lr))
                """
                # Per Class Accuracy
                """
                per_class_accuracy(Prediction, batch_ys)
                """
            
            tEnd = time.time()            
            Train_acc  = total_correct_num  / (total_correct_num + total_error_num)
            Train_loss = Train_loss / iteration 
            print("\r\033[0;33mEpoch{}\033[0m" .format(epoch), end = "")
            print(" (Cost {TIME} sec)" .format(TIME = tEnd - tStart))
            print("\033[0;32mLearning Rate    \033[0m : {}".format(Learning_Rate))
            print("\033[0;32mTraining Accuracy\033[0m : {}".format(Train_acc))
            print("\033[0;32mTraining Loss    \033[0m : {} (l2_norm: {})".format(Train_loss, L2_norm))
            
            # Record Per Epoch Training Result (Finally this will save as the .csv file)
            if epoch==0:
                Train_acc_per_epoch  = np.array([Train_acc ])
                Train_loss_per_epoch = np.array([Train_loss])
            else:
                Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc ])], axis=0)
                Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
            
            #-------------------------#
            #   Learning Rate Decay   #
            #-------------------------#
            """
            if HP['LR_Strategy'] == 'PerEpoch':
                if epoch == 0:
                    alpha = (HP['LR_Final'] / HP['LR'])**(1./HP['Epoch'])
                lr = alpha * lr
            elif HP['LR_Strategy'] == '2times':
                # 1st
                if (epoch+1) == HP['LR_Decade_1st_Epoch']:
                    lr = lr / HP['LR_Decade']
                # 2nd
                if (epoch+1) == HP['LR_Decade_2nd_Epoch']:
                    lr = lr / HP['LR_Decade']
            """
            ##similar_group(inputs_and_kernels, sess)
            
            #---------------------------#
            #   Train Directory Build   #
            #---------------------------#
            if (not IS_HYPERPARAMETER_OPT) and (epoch+1)==HP['Epoch']:
                if (not os.path.exists('Model/'+Model_first_name + '_Model/')) :
                    print("\033[0;35m%s\033[0m is not exist!" %'Model/'+Model_first_name)
                    print("\033[0;35m%s\033[0m is created!" %'Model/'+Model_first_name)
                    os.mkdir(Model_first_name)
                
                Dir = 'Model/' + Model_first_name + '_Model/'
                Dir = Dir + Model_Name + '_'
                Dir = Dir + str(int(Train_acc*100)) + '_'
                try:
                    Dir = Dir + str(int(total_valid_accuracy*100)) + '_'
                except:
                    Dir = Dir + 'cifar10_'
                Dir = Dir + time.strftime("%Y.%m.%d/")
                
                if (not os.path.exists(Dir)):
                    print("\033[0;35m%s\033[0m is not exist!" %Dir)
                    print("\033[0;35m%s\033[0m is created!" %Dir)
                    os.makedirs(Dir)
                
                #--------------------#
                #    Saving Model    #
                #--------------------#
                # Hyperparameter
                np.savetxt(Dir + 'Hyperparameter.csv', HP_csv, delimiter=",", fmt="%s")
                # Model
                Model_csv_Generator(Model_dict, Dir + 'model')
                # Analysis
                Save_Analyzsis_as_csv(Analysis, Dir + 'Analysis')
                
            #----------------------------#
            #   Saving trained weights   #
            #----------------------------#
            if (not IS_HYPERPARAMETER_OPT) and (epoch+1)==HP['Epoch']:
                print("Saving Trained Weights ...")
                save_path = saver.save(sess, Dir + str(epoch+1) + ".ckpt")
                print("\033[0;35m{}\033[0m" .format(save_path))
        
        tEnd_All = time.time()
        print("Total costs {TIME} sec\n" .format(TIME = tEnd_All - tStart_All))

        #-----------------------------------#
        #   Saving Train info as csv file   #
        #-----------------------------------#
        if not IS_HYPERPARAMETER_OPT:
            Train_acc_per_epoch  = np.expand_dims(Train_acc_per_epoch , axis=1)
            Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
            Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
            Save_file_as_csv(Dir+'train_info' , Train_info)
        print("see the batch norm info!")
    tf.reset_default_graph()
        
    Model_Path = Dir
    Model = str(HP['Epoch']) + '.ckpt'
    
    return Model_Path, Model

def Testing(
    # File Read
    Dataset                    ,
    Dataset_Path               ,
    class_num                  ,
    H_Resize                   ,
    W_Resize                   ,
    BATCH_SIZE                 ,
    # Parameter	    
    Model_dict                 ,
    IS_SAVING_RESULT_AS_IMAGE  ,
    IS_SAVING_RESULT_AS_NPZ    ,
    IS_HYPERPARAMETER_OPT      ,
    IS_TRAIN                   ,
    IS_VALID                   ,
    IS_TEST                    ,
    # Loading Trained Weight
    testing_model_path         ,
    testing_model              ,
    # For Saving Result
    train_target_path          ,
    valid_target_path          ,
    test_target_path           ,
    train_Y_pre_path           ,
    valid_Y_pre_path           ,
    test_Y_pre_path            
    ):
    
    print("Testing ... ")
    dStart = time.time()
    #------------------#
    #    Placeholder   #
    #------------------#
    data_shape = [None, H_Resize, W_Resize, 3]
    # Is_training
    is_training = tf.placeholder(tf.bool)
    
    # is_quantized_activation
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
        
    # is_ternary
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})  
        
    #----------------------#
    #    (Beta) Dataset    #
    #----------------------#
    filenames = [os.path.join(Dataset_Path, 'test_batch.bin')]
    xs, ys = input_fn(
        filenames,
        class_num,
        False,
        HP = {
            'H_Resize'  : H_Resize,
            'W_Resize'  : W_Resize,
            'Batch_Size': BATCH_SIZE,
            'Epoch'     : None})
    
    #---------------------#
    #    Loading Model    #
    #---------------------#
    print("Building Model ...")
    device = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    device_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    device_batch = int(BATCH_SIZE / device_num)
    prediction_list = []
    # Assign Usage of CPU and GPU
    for d in range(device_num):
        device_use = device[d]
        with tf.device(tf.train.replica_device_setter(worker_device = '/device:GPU:%d' %int(device_use), ps_device = '/device:CPU:0', ps_tasks=1)):
            # -- Build Model --
            net = xs[ d*device_batch : (d+1)*device_batch ]
            Model_dict_ = copy.deepcopy(Model_dict)
            if d == 0:
                prediction, Analysis, max_parameter, inputs_and_kernels = Model_dict_Decoder(
                    net                     = net, 
                    Model_dict              = Model_dict_, 
                    is_training             = is_training,
                    is_ternary              = is_ternary,
                    is_quantized_activation = is_quantized_activation,
                    DROPOUT_RATE            = None,
                    reuse                   = None)
            else:
                prediction, Analysis, max_parameter, inputs_and_kernels = Model_dict_Decoder(
                    net                     = net, 
                    Model_dict              = Model_dict_, 
                    is_training             = is_training,
                    is_ternary              = is_ternary,
                    is_quantized_activation = is_quantized_activation,
                    DROPOUT_RATE            = None,
                    reuse                   = True)
                
            prediction = tf.squeeze(prediction, [1, 2])
            prediction_list.append(prediction)
            
            # -- Model Size --
            if d == 0:
                #--------------------------#
                #    Compute Model Size    #
                #--------------------------#
                # Your grade will depend on this value.
                all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model")
                # Model Size
                Model_Size = 0
                for iter, variable in enumerate(all_variables):
                    Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
                    # See all your variables in termainl	
                    """
                    print(variable)
                    """
                print("\033[0;36m=======================\033[0m")
                print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size))
                print("\033[0;36m=======================\033[0m")
                 
    #-----------------------------#
    #   Some Control Parameters   #
    #-----------------------------#
    IS_TERNARY = False
    IS_QUANTIZED_ACTIVATION = False
    for layer in range(len(Model_dict)):
        if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
            IS_TERNARY = True
        if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
            IS_QUANTIZED_ACTIVATION = True
    
    #-----------#
    #   Saver   #
    #-----------#	
    saver = tf.train.Saver()
    
    #---------------#
    #    Session    #
    #---------------#
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = 256
    with tf.Session(config = config) as sess:
        #--------------------------#
        #   Load trained weights   #
        #--------------------------#
        print("Loading the trained weights ... ")
        print("\033[0;35m{}\033[0m" .format(testing_model_path + testing_model))
        save_path = saver.restore(sess, testing_model_path + testing_model)

        #similar_group(inputs_and_kernels, sess)

        print("Testing Data Result ... ")
        test_result, test_accuracy, test_accuracy_top2, test_accuracy_top3, test_Y_pre = compute_accuracy(
            xs                      = xs, 
            ys                      = ys, 
            is_training             = is_training,
            is_quantized_activation = is_quantized_activation,
            Model_dict              = Model_dict,                
            QUANTIZED_NOW           = True, 
            prediction_list         = prediction_list, 
            BATCH_SIZE              = BATCH_SIZE, 
            sess                    = sess)
        print("\033[0;32mtesting Data Accuracy\033[0m = {test_Accuracy}, top2 = {top2}, top3 = {top3}\n"
        .format(test_Accuracy = test_accuracy, top2 = test_accuracy_top2, top3 = test_accuracy_top3))

        #--------------------------#
        #   Save Result as Image   #
        #--------------------------#
        if IS_SAVING_RESULT_AS_IMAGE:
            print("Coloring test result ... ")
            test_result = color_result(test_result)
            print("Saving the test data result as image ... ")
            Save_result_as_image(test_target_path, test_result, test_data_index)
        #-----------------------------#
        #   Save Result as NPZ File   #
        #-----------------------------#
        if IS_SAVING_RESULT_AS_NPZ:              
            print("Saving the test Y_pre result as npz ... ")
            Save_result_as_npz(	
                Path       = test_Y_pre_path, 
                result     = test_Y_pre, 
                file_index = test_data_index,
                sess       = sess)
                    
    #-----------------------------#
    #    Save Accuracy as .csv    #
    #-----------------------------#
    #accuracy_train = np.concatenate([[['Train Data  ']], [[train_accuracy]], [[train_accuracy_top2]], [[train_accuracy_top3]]], axis=1)
    #accuracy_valid = np.concatenate([[['Valid Data  ']],[[valid_accuracy]], [[valid_accuracy_top2]], [[valid_accuracy_top3]]], axis=1)
    accuracy_test  = np.concatenate([[['Test Data   ']],[[ test_accuracy]], [[ test_accuracy_top2]], [[ test_accuracy_top3]]], axis=1)
    # Title
    title_col = np.expand_dims(np.array(['            ', 'TOP1	', 'TOP2	', 'TOP3	']),axis=0)
    # Data
    accuracy = np.concatenate([title_col, accuracy_test], axis=0)
    # Save
    np.savetxt(testing_model_path + 'Accuracy.csv', accuracy, delimiter=",	", fmt="%5s")	

    return test_accuracy
 
def Model_dict_Generator(
    csv_file = None,
    class_num = 1
    ):
    
    #----------------------#
    #    Hyperparameter    #
    #----------------------#
    """
    Read Hyperparameter in .csv file 
    Or, you can directly define it here.
    
    i.e. 
        HP_tmp.update({'type'       :['CONV', 'CONV', ...]})
        HP_tmp.update({'kernel_size':['3', '3', ...]})
        ... (Until all hyperparameter is defined)
    """
    
    HP_tmp = {}
    keys = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for iter, row in enumerate(reader):
            keys.append(row[0])
            HP_tmp.update({row[0]:row[1:len(row)]})
    
    #------------#
    #    Info    #
    #------------#
    depth = len(HP_tmp['layer'])
    
    #------------------------#
    #    Model Definition    #
    #------------------------#
    Model_dict = {}
    for layer in range(depth):
        HP = {}		
        for iter, key in enumerate(keys):
            if key == 'output_channel' and layer == (depth-1):
                HP.update({'output_channel': class_num})
            else:
                HP.update({key: HP_tmp[key][layer]})
        
        #HP.update({'is_ternary'             : is_ternary             ['layer' + str(layer)]}) # Placeholder
        #HP.update({'is_quantized_activation': is_quantized_activation['layer' + str(layer)]}) # Placeholder
        Model_dict.update({'layer'+str(layer):HP})
    
    return Model_dict	

def Hyperparameter_Decoder(
    Hyperparameter, 
    Model_dict
    ):
    
    """
    Training Hyperparameter (Reference)
    |=======================================================================================================|
    | Num|              Type                     |              0               |              1            |
    |=======================================================================================================|
    | 00 | Optimization Method                   | MOMENTUM                     | ADAM                      |
    | 01 | Momentum Rate                         | 0.9                          | 0.99                      |
    | 02 | Initial Learning Rate                 | < 0.01                       | >= 0.01                   |
    | 03 | Initial Learning Rate                 | < 0.001; <0.1;               | >= 0.001; >= 0.1          |
    | 04 | Initial Learning Rate                 | 0.0001; 0.001; 0.01; 0.1;    | 0.0003; 0.003; 0.03; 0.3  |
    | 05 | Learning Rate Drop                    | 5                            | 10                        |
    | 06 | Learning Rate First Drop Time         | Drop by 1/10 at Epoch 40     | Drop by 1/10 at Epoch 60  |
    | 07 | Learning Rate Second Drop Time        | Drop by 1/10 at Epoch 80     | Drop by 1/10 at Epoch 100 |
    | 08 | Weight Decay                          | No                           | Yes                       | 
    | 09 | Weight Decay Lambda                   | 1e-4                         | 1e-3                      |
    | 10 | Batch Size                            | Small                        | Big                       |
    | 11 | Batch Size                            | 32; 128                      | 64; 256                   |
    | 12 | Teacher-Student Strategy              | No                           | Yes                       |
    | 13 | Use Dropout                           | No                           | Yes                       |
    | 14 | Dropout Rate                          | Low                          | High                      | 
    | 15 | Dropout Rate                          | 0.05; 0.2                    | 0.1; 0.3                  |
    | 16 | Weight Ternary Epoch                  | 40                           | 60                        |
    | 17 | Activation Quantized Epoch            | 80                           | 100                       |
    |=======================================================================================================|
    
    Model Hyperparameter (Per Layer)
    |=======================================================================================================|
    | Num|              Type                     |              0               |              1            |
    |=======================================================================================================|
    | 00 | Weight Ternary                        | No                           | Yes                       |
    | 01 | Activation Quantized                  | No                           | Yes                       |
    | 02 | Batch Normalization                   | No                           | Yes                       |
    | 03 | Shortcut                              | No                           | Yes                       |
    | 04 | Shortcut Distance                     | 1                            | 2                         |
    | 05 | Bottlneck                             | No                           | Yes                       |
    | 06 | Inception                             | No                           | Yes                       |
    | 07 | Dilated (Rate=2)                      | No                           | Yes                       |
    | 08 | Depthwise                             | No                           | Yes                       |
    | 09 | Activation                            | Sigmoid                      | ReLU                      |
    | 10 | Group                                 | <3                           | >3                        |
    | 11 | Group                                 | 1, 4                         | 2, 8                      |
    |=======================================================================================================|
    """
    
    HP_dict = {}
    Bit_Now = 0
    Bits = 0
    
    #-------------------------------#
    #    Training Hyperparameter    #
    #-------------------------------#
    # Optimization Method
    Bit_Now = Bit_Now + Bits
    Bits = 1
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'Opt_Method': 'Momentum'})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'Opt_Method': 'Adam'})
    # Momentum Rate 
    Bit_Now = Bit_Now + Bits
    Bits = 1
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'Momentum_Rate': 0.9})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'Momentum_Rate': 0.99})
    # Initial Learning Rate
    Bit_Now = Bit_Now + Bits
    Bits = 3
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1, -1]): HP_dict.update({'LR': 0.0001})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1,  1]): HP_dict.update({'LR': 0.0003})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1, -1]): HP_dict.update({'LR': 0.001 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1,  1]): HP_dict.update({'LR': 0.003 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1, -1]): HP_dict.update({'LR': 0.01  })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1,  1]): HP_dict.update({'LR': 0.03  })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1, -1]): HP_dict.update({'LR': 0.1   })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1,  1]): HP_dict.update({'LR': 0.3   })
    # Learning Rate Drop
    Bit_Now = Bit_Now + Bits
    Bits = 1
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'LR_Decade':  5})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'LR_Decade': 10})
    # Learning Rate First Drop Time
    Bit_Now = Bit_Now + Bits
    Bits = 1
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'LR_Decade_1st_Epoch': 40})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'LR_Decade_1st_Epoch': 60})
    # Learning Rate Second Drop Time
    Bit_Now = Bit_Now + Bits
    Bits = 1
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'LR_Decade_2nd_Epoch': 80 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'LR_Decade_2nd_Epoch': 100})	
    # Weight Decay Lambda
    Bit_Now = Bit_Now + Bits
    Bits = 2
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1]): HP_dict.update({'L2_Lambda': 0.0 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1]): HP_dict.update({'L2_Lambda': 0.0 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1]): HP_dict.update({'L2_Lambda': 1e-4})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1]): HP_dict.update({'L2_Lambda': 1e-3})
    # Batch Size
    Bit_Now = Bit_Now + Bits
    Bits = 2
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1]): HP_dict.update({'Batch_Size': 128 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1]): HP_dict.update({'Batch_Size': 256 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1]): HP_dict.update({'Batch_Size': 512 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1]): HP_dict.update({'Batch_Size': 1024})
    # Teacher-Student Strategy
    Bit_Now = Bit_Now + Bits
    Bits = 1
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'IS_STUDENT': 'FALSE'})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'IS_STUDENT': 'TRUE' })
    # Dropout Rate
    Bit_Now = Bit_Now + Bits
    Bits = 3
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1, -1]): HP_dict.update({'Dropout_Rate': 0.0 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1,  1]): HP_dict.update({'Dropout_Rate': 0.0 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1, -1]): HP_dict.update({'Dropout_Rate': 0.0 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1,  1]): HP_dict.update({'Dropout_Rate': 0.0 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1, -1]): HP_dict.update({'Dropout_Rate': 0.05})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1,  1]): HP_dict.update({'Dropout_Rate': 0.1 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1, -1]): HP_dict.update({'Dropout_Rate': 0.2 })
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1,  1]): HP_dict.update({'Dropout_Rate': 0.3 })
    # Weight Ternary Epoch
    Bit_Now = Bit_Now + Bits
    Bits = 1
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'Ternary_Epoch': 2})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'Ternary_Epoch': 2})
    # Activation Quantized Epoch
    Bit_Now = Bit_Now + Bits
    Bits = 1
    if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'Quantized_Activation_Epoch': 3})
    elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'Quantized_Activation_Epoch': 3})
    
    #-----------------------------------#
    #    Model Hyperparameter Update    #
    #-----------------------------------#
    for layer in range(len(Model_dict)):
        # IS_TERNARY
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'IS_TERNARY': 'FALSE'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'IS_TERNARY': 'TRUE' })
        # IS_QUANTIZED_ACTIVATION
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'IS_QUANTIZED_ACTIVATION': 'FALSE'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'IS_QUANTIZED_ACTIVATION': 'TRUE' })
        # is_batch_norm
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_batch_norm': 'FALSE'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_batch_norm': 'TRUE' })
        # is_shortcut
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_shortcut': 'FALSE'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_shortcut': 'TRUE' })
        # shortcut_destination
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'shortcut_destination': 'layer'+str(layer+1)})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'shortcut_destination': 'layer'+str(layer+2)})
        # is_bottleneck
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_bottleneck': 'FALSE'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_bottleneck': 'TRUE' })
        # is_inception
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_inception': 'FALSE'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_inception': 'TRUE' })
        # is_dilated
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_dilated': 'FALSE'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_dilated': 'TRUE' })
        # is_depthwise
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_depthwise': 'FALSE'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_depthwise': 'TRUE' })
        # Activation
        Bit_Now = Bit_Now + Bits
        Bits = 1
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'Activation': 'Sigmoid'})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'Activation': 'CReLU'  })
        # Group
        Bit_Now = Bit_Now + Bits
        Bits = 2
        if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'Group': 1})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'Group': 2})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'Group': 4})
        elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'Group': 8})
    return HP_dict, Model_dict

def Model_dict_Decoder(
    net, 
    Model_dict, 
    is_training,
    is_ternary,
    is_quantized_activation,
    DROPOUT_RATE,
    reuse = None
    ):
    
    Analysis = {}
    inputs_and_kernels = {} # For finding the similar weights
    
    Analyzer(Analysis, net, type='DATA', name='Input')
    past_layer = 0
    max_parameter = 0
    with tf.variable_scope("Model", reuse = reuse):
        for layer in range(len(Model_dict)):
            layer_now = Model_dict['layer'+str(layer)]
            
            # -- Shortcut --
            if bool(layer_now.get('shortcut_num')):
                for shortcut_index in range(layer_now['shortcut_num']):
                    with tf.variable_scope(layer_now['shortcut_input_layer'][str(shortcut_index)]):
                        shortcut_input_layer = Model_dict[layer_now['shortcut_input_layer'][str(shortcut_index)]]
                        shortcut = layer_now['shortcut_input'][str(shortcut_index)]
                        ## Show the model
                        shortcut_layer = layer_now['shortcut_input_layer'][str(shortcut_index)]
                        print("\033[0;35mShortcut from {}\033[0m" .format(shortcut_layer))
                        # Add shortcut 
                        shortcut = Model.shortcut_Module(
                            net                     = shortcut, 
                            destination             = net,
                            initializer             = tf.contrib.layers.variance_scaling_initializer(),
                            is_training             = is_training,
                            is_add_biases           = shortcut_input_layer['is_add_biases']           == 'TRUE',
                            is_projection_shortcut  = shortcut_input_layer['is_projection_shortcut']  == 'TRUE',
                            is_batch_norm           = shortcut_input_layer['is_batch_norm']           == 'TRUE',
                            is_ternary              = is_ternary[shortcut_layer]                               ,
                            is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                            IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE', 	
                            IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                            padding                 = "SAME",			     
                            Analysis                = Analysis)
                        
                    with tf.variable_scope('layer%d' %(layer)):
                        if shortcut_input_layer['is_shuffle'] == "TRUE" and shortcut_input_layer['stride'] == '2':
                            net = tf.concat([net, shortcut], axis = 3)
                        else:
                            net = tf.add(net, shortcut)
                        
                        # Activation Quantization
                        if layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
                            quantize_Module(net, is_quantized_activation['layer'+str(layer)])
 
                        # -- Analyzer --
                        Analyzer( Analysis, 
                                  net, 
                                  type                    = 'ADD', 
                                  IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE', 
                                  name                    = 'shortcut_ADD')  
                
            # Show the model
            print("\033[0;33mlayer{}\033[0m -> {}" .format(layer, net.shape))
                
            # -- Destination --
            if layer_now['is_shortcut'] == 'TRUE':
                if bool(Model_dict[layer_now['shortcut_destination']].get('shortcut_num')):
                    shortcut_num = Model_dict[layer_now['shortcut_destination']]['shortcut_num'] + 1
                    Model_dict[layer_now['shortcut_destination']].update({'shortcut_num':shortcut_num})
                    Model_dict[layer_now['shortcut_destination']]['shortcut_input'].update({str(shortcut_num-1):net})
                    Model_dict[layer_now['shortcut_destination']]['shortcut_input_layer'].update({str(shortcut_num-1):'layer%d' %(layer)})
                else:
                    shortcut_num = 1
                    Model_dict[layer_now['shortcut_destination']].update({'shortcut_num':shortcut_num})
                    Model_dict[layer_now['shortcut_destination']].update({'shortcut_input':{str(shortcut_num-1):net}})
                    Model_dict[layer_now['shortcut_destination']].update({'shortcut_input_layer':{str(shortcut_num-1):'layer%d' %(layer)}})
            
            # -- Pooling, Unpooling, Convolution --
            with tf.variable_scope('layer%d' %(layer)):                
                # -- Max Pooling --
                if layer_now['type'] == 'MAX_POOL':
                    ## Show the model
                    print("-> Max Pooling, Ksize={}, Stride={}".format(int(layer_now['kernel_size']), int(layer_now['stride'])))
                    net, indices, output_shape = Model.indice_pool( 
                        net, 
                        kernel_size             = int(layer_now['kernel_size']), 
                        stride                  = int(layer_now['stride']), 
                        IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                        Analysis                = Analysis,
                        scope                   = layer_now['scope'])
                        
                    if layer_now['indice'] != 'None' and layer_now['indice'] != 'FALSE':
                        Model_dict[layer_now['indice']].update({'indice':indices, 'output_shape':output_shape})
                
                # -- Max Unpooling --
                elif layer_now['type'] == 'MAX_UNPOOL':
                    ## Show the model
                    print("-> Max Unpooling")
                    net = Model.indice_unpool( 
                        net,  
                        output_shape = layer_now['output_shape'], 
                        indices      = layer_now['indice'], 
                        scope        = layer_now['scope'])
                        
                # -- Avg Pooling --
                elif layer_now['type'] == 'AVG_POOL':
                    ## Show the model
                    print("-> Avg Pooling")
                    net = tf.nn.avg_pool(
                        value   = net,
                        ksize   = [1, int(layer_now['kernel_size']), int(layer_now['kernel_size']), 1],
                        strides = [1, int(layer_now['stride']), int(layer_now['stride']), 1],
                        padding = 'SAME')

                # -- Batch Normalization --
                elif layer_now['type'] == 'BN':
                    ## Show the model
                    print("-> Batch_Norm")
                    net = Model.batch_norm(net, is_training)
                    # Activation
                    if layer_now['Activation'] == 'ReLU':
                        # Show the model
                        print("-> ReLU")
                        net = tf.nn.relu(net)
                    elif layer_now['Activation'] == 'Sigmoid':
                        net = tf.nn.sigmoid(net)
                        # Show the model
                        print("-> Sigmoid")

                # -- Convolution --
                elif layer_now['type'] == 'CONV':
                    # Dropout
                    #if layer==(len(Model_dict)-1):
                    #    net = tf.cond(is_training, lambda: tf.layers.dropout(net, DROPOUT_RATE), lambda: net)

                    # For finding the similar weights
                    inputs_and_kernels.update({'layer%d' %(layer): {'inputs' : net}})
                    
                    net  = Model.conv2D( net, 
                                        kernel_size             = int(layer_now['kernel_size']     ), 
                                        stride                  = int(layer_now['stride']          ),
                                        internal_channel        = int(layer_now['internal_channel']),
                                        output_channel          = int(layer_now['output_channel']  ),
                                        rate                    = int(layer_now['rate']            ),
                                        group                   = int(layer_now['group']           ),
                                        initializer             = tf.variance_scaling_initializer(), 
                                        is_training             = is_training,
                                        is_add_biases           = layer_now['is_add_biases']           == 'TRUE',
                                        is_bottleneck           = layer_now['is_bottleneck']           == 'TRUE',                  
                                        is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',      
                                        is_dilated              = layer_now['is_dilated']              == 'TRUE',      
                                        is_depthwise            = layer_now['is_depthwise']            == 'TRUE',  
                                        is_inception            = layer_now['is_inception']            == 'TRUE',
                                        is_shuffle              = layer_now['is_shuffle']              == 'TRUE',
                                        is_ternary              = is_ternary['layer'+str(layer)]                ,
                                        is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                                        IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',     
                                        IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',						              
                                        Activation              = layer_now['Activation'],
                                        padding                 = "SAME",
                                        Analysis                = Analysis,
                                        scope                   = layer_now['scope'])

        # For finding the similar weights
        for layer in range(len(Model_dict)):
            try:
                inputs_and_kernels['layer%d' %(layer)].update({
                    #'kernels': tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model/layer%d" %(layer))})
                    'kernels': tf.get_collection("final_weights", scope="Model/layer%d" %(layer))})
            except:
                None
                
        # -- Calcute the max parameter size in all layers --
        current_layer = len(Analysis)-1
        layer_num = current_layer - past_layer
        
        # parameter_in
        if past_layer>=0 and past_layer < 10:
            parameter_in  = Analysis['layer00'+str(past_layer)]['Activation'] * Analysis['layer00'+str(past_layer)]['Activation Bits']
        elif past_layer>=10 and past_layer < 100:
            parameter_in  = Analysis['layer0'+str(past_layer)]['Activation'] * Analysis['layer0'+str(past_layer)]['Activation Bits']
        elif past_layer > 100:
            parameter_in  = Analysis['layer'+str(past_layer)]['Activation'] * Analysis['layer'+str(past_layer)]['Activation Bits']
            
        # parameter_out
        if current_layer>=0 and current_layer < 10:
            parameter_out  = Analysis['layer00'+str(current_layer)]['Activation'] * Analysis['layer00'+str(current_layer)]['Activation Bits']
        elif current_layer>=10 and current_layer < 100:
            parameter_out  = Analysis['layer0'+str(current_layer)]['Activation'] * Analysis['layer0'+str(current_layer)]['Activation Bits']
        elif current_layer > 100:
            parameter_out  = Analysis['layer'+str(current_layer)]['Activation'] * Analysis['layer'+str(current_layer)]['Activation Bits']
        
        # parameter_kernel
        parameter_kernel = 0
        for i in range(layer_num):
            layer_now = past_layer + i + 1
            if layer_now>=0 and layer_now < 10:
                parameter_kernel = Analysis['layer00'+str(layer_now)]['Param'] * Analysis['layer00'+str(layer_now)]['Kernel Bits']
            elif layer_now>=10 and layer_now < 100:
                parameter_kernel = Analysis['layer0'+str(layer_now)]['Param'] * Analysis['layer0'+str(layer_now)]['Kernel Bits']
            elif layer_now > 100:
                parameter_kernel = Analysis['layer'+str(layer_now)]['Param'] * Analysis['layer'+str(layer_now)]['Kernel Bits']
                
            parameter = parameter_in + parameter_out + parameter_kernel
    
            if parameter > max_parameter:
                max_parameter = parameter
    
        past_layer = current_layer
        
    return net, Analysis, max_parameter, inputs_and_kernels

def Model_csv_Generator(
    Model_dict,
    csv_file
    ):
    
    Model = np.array(['layer'                  ,
                      'type'                   ,
                      'kernel_size'            ,
                      'stride'                 ,
                      'internal_channel'       ,
                      'output_channel'         ,
                      'rate'                   ,
                      'group'                  ,
                      'is_add_biases'          ,
                      'is_shortcut'            ,
                      'shortcut_destination'   ,
                      'is_projection_shortcut' ,
                      'is_bottleneck'          ,
                      'is_batch_norm'          ,
                      'is_dilated'             ,
                      'is_depthwise'           ,
                      'is_inception'           ,
                      'is_shuffle'             ,
                      'IS_TERNARY'             ,
                      'IS_QUANTIZED_ACTIVATION',
                      'Activation'             ,
                      'indice'                 ,
                      'scope'                  ])     
                    
    Model = np.expand_dims(Model, axis=1)
    
    for layer in range(len(Model_dict)):
        for iter, key in enumerate(Model):
            if iter==0:
                Model_per_layer = np.array(['layer'+str(layer)])
            else:
                if key[0]=='indice':
                    if Model_dict['layer'+str(layer)]['type']=='POOL':
                        Model_per_layer = np.concatenate([Model_per_layer, np.array([Model_dict['layer'+str(layer)][key[0]]])], axis=0)
                    else:
                        Model_per_layer = np.concatenate([Model_per_layer,np.array(['None'])], axis=0)
                else:
                    Model_per_layer = np.concatenate([Model_per_layer, np.array([Model_dict['layer'+str(layer)][key[0]]])], axis=0)
                    
        Model_per_layer = np.expand_dims(Model_per_layer, axis=1)
        Model = np.concatenate([Model, Model_per_layer], axis=1)
        
    np.savetxt(csv_file + '.csv', Model, delimiter=",", fmt="%s")

def input_fn(
    filenames,
    class_num,
    is_training,
    HP
    ):
    
    record_bytes = HP['H_Resize'] * HP['W_Resize'] * 3 + 1
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    dataset = dataset.map(lambda raw_record: parse_record(raw_record, HP['H_Resize'], HP['W_Resize'], class_num))
    dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))
    dataset = dataset.prefetch(2 * HP['Batch_Size'])
    dataset = dataset.repeat(HP['Epoch'])
    dataset = dataset.batch(HP['Batch_Size'])
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    
    return images, labels

def model_fn( # No use
    features,
    labels,
    mode,
    params
    ):

    print("Building Model ...")
    device = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    device_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    device_batch = params['Batch_Size'] / device_num
    batches_per_epoch = 50000 / params['Batch_Size']
    prediction_list = []
    # Assign Usage of CPU and GPU
    for d in range(device_num):
        device_use = device[d]
        with tf.device(tf.train.replica_device_setter(worker_device = '/device:GPU:%d' %int(device_use), ps_device = '/device:CPU:0', ps_tasks=1)):
            # -- Build Model --
            is_ternary = {}
            is_quantized_activation = {}
            for layer in range(len(params['Model_dict'])):
                # ternary
                global_step = tf.train.get_or_create_global_step()
                boundaries = [int(batches_per_epoch * epoch) for epoch in [params['Ternary_Epoch']]]
                values = [False, params['Model_dict']['layer%d'%layer]['IS_TERNARY']=='TRUE']
                is_ternary_ = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
                is_ternary.update({'layer%d'%layer : is_ternary_}) 
                # quantization
                global_step = tf.train.get_or_create_global_step()
                boundaries = [int(batches_per_epoch * epoch) for epoch in [params['Quantized_Activation_Epoch']]]
                values = [False, params['Model_dict']['layer%d'%layer]['IS_QUANTIZED_ACTIVATION']=='TRUE']
                is_quantized_activation_ = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
                is_quantized_activation.update({'layer%d'%layer : is_quantized_activation_}) 
            
            if d == 0:
                Model_dict = copy.deepcopy(params['Model_dict'])
                prediction, Analysis, max_parameter, inputs_and_kernels = Model_dict_Decoder(
                    features, 
                    params['Model_dict'], 
                    params['Dropout_Rate'],
                    mode == tf.estimator.ModeKeys.TRAIN,
                    is_ternary,
                    is_quantized_activation
                    )
            else:
                prediction, Analysis, max_parameter, inputs_and_kernels = Model_dict_Decoder(
                    features, 
                    Model_dict, 
                    params['Dropout_Rate'], 
                    mode == tf.estimator.ModeKeys.TRAIN,
                    is_ternary,
                    is_quantized_activation,
                    reuse = True)

            logits = tf.squeeze(prediction, [1, 2])
            
            predictions = {
                'classes': tf.argmax(logits, axis = -1),
                'probabilities': tf.nn.softmax(logits, name = 'softmax_tensor')
                }
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
                
            prediction_list.append(prediction)
            
            # -- Model Size --
            if d == 0:
                # Your grade will depend on this value.
                all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model")
                # Model Size
                Model_Size = 0
                for iter, variable in enumerate(all_variables):
                    Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
                    # See all your variables in termainl	
                    """
                    print(variable)
                    """
                print("\033[0;36m=======================\033[0m")
                print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size))
                print("\033[0;36m=======================\033[0m")
            
            # -- Collection --
            if d == 0:
                ## Gradient Update
                var_list_collection                 = tf.get_collection("var_list"               , scope=None)
                float32_params                      = tf.get_collection("float32_params"         , scope=None) 

            # -- Loss --
            # L2 Regularization
            l2_norm   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            l2_lambda = tf.constant(params['L2_Lambda'])
            l2_norm   = tf.multiply(l2_lambda, l2_norm)
            # Loss
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels = labels,
                logits        = logits)  
            loss = tf.add(loss, l2_norm)

            # Create a tensor named cross_entropy for logging purposes.
            tf.identity(loss, name='loss')
            tf.summary.scalar('loss', loss)
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                # -- Learning Rate --
                initial_learning_rate = params['LR']  
                global_step = tf.train.get_or_create_global_step()
                boundaries = [int(batches_per_epoch * epoch) for epoch in [params['LR_Decade_1st_Epoch'], params['LR_Decade_2nd_Epoch']]]
                values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01]]
                learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
                
                # Create a tensor named learning_rate for logging purposes
                tf.identity(learning_rate, name='learning_rate')
                tf.summary.scalar('learning_rate', learning_rate)

                #--------------------#
                #    Optimization    #
                #--------------------#
                # Optimizer
                if params['Opt_Method']=='Adam':
                    opt = tf.train.AdamOptimizer(learning_rate, params['Momentum_Rate'])
                elif params['Opt_Method']=='Momentum':
                    opt = tf.train.MomentumOptimizer(
                        learning_rate = learning_rate, 
                        momentum      = params['Momentum_Rate'])
                    
                # Batch norm requires update ops to be added as a dependency to the train_op
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                #for i, variable in enumerate(update_ops):
                #    print(variable)
                with tf.control_dependencies(update_ops):
                    # Compute Gradients
                    gradients = opt.compute_gradients(loss, var_list = var_list_collection)
                    gra_and_var = [(gradients[i][0], float32_params[i]) for i in range(np.shape(gradients)[0])]
                    train_op  = opt.apply_gradients(gra_and_var)
            else:
                train_op = None
                
            accuracy = tf.metrics.accuracy(tf.argmax(labels, axis = 1), predictions['classes'])
            metrics = {'accuracy': accuracy}
            
            # Create a tensor named train_accuracy for logging purposes
            tf.identity(accuracy[1], name='train_accuracy')
            tf.summary.scalar('train_accuracy', accuracy[1])
            print("...")
            return tf.estimator.EstimatorSpec(
                mode            = mode,
                predictions     = predictions,
                loss            = loss,
                train_op        = train_op,
                eval_metric_ops = metrics)
	
#============#
#   Parser   #
#============#
def file_name_parser(
    filepath
    ):
    
    filename = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    
    return filename

def dataset_parser(
    Dataset,
    # Path
    Path,             # e.g. '/Path_to_Dataset/CamVid'
    Y_pre_Path,       # e.g. '/Path_to_Y_pre/CamVid'
    data_index,
    target_index,
    # Variable
    class_num,
    H_resize,
    W_resize,
    # Parameter
    IS_STUDENT,
    IS_TRAINING
    ):
    
    # Data
    data = read_dataset_file(data_index, Path, H_resize, W_resize)
    data = data / 255.
    # Target
    if IS_TRAINING and IS_STUDENT:	 
        if Dataset == 'mnist' or Dataset == 'cifar10':
            target = read_Y_pre_file(data_index, Y_pre_Path, 1, 1, layer)
        else:
            target = read_Y_pre_file(data_index, Y_pre_Path, H_resize, W_resize, layer)
    else:
        if Dataset == 'mnist' or Dataset == 'cifar10':
            target = read_dataset_file(target_index, Path, 1, 1, True)
        else:
            target = read_dataset_file(target_index, Path, H_resize, W_resize, True)   
        target = one_of_k(target, class_num)
    
    return data, target

def read_dataset_file(
    data_index, 
    Path, 
    H_resize, 
    W_resize, 
    IS_TARGET = False
    ):
    
    num = len(data_index)
    batch_size = 250.
    batch_num = int(math.ceil(num / batch_size))
    iter = 0
    for batch in range(batch_num):
        batch_data_index = data_index[ batch*int(batch_size) : (batch+1)*int(batch_size)]
        for i, file_name in enumerate(batch_data_index):
            print("\r{} / {}" .format(iter, num), end = "")
            iter = iter + 1
            # Read Data
            data_tmp  = misc.imread(Path + file_name)
            # Data Preprocessing
            ## Image Resizing
            data_tmp = scipy.misc.imresize(data_tmp, (H_resize, W_resize))
            if len(np.shape(data_tmp))==2 and (not IS_TARGET):
                data_tmp = np.repeat(data_tmp[:, :, np.newaxis], 3, axis=2)
            # Concatenate the Data
            if i == 0:
                batch_data  = np.expand_dims(data_tmp , axis=0)
            else:
                batch_data  = np.concatenate([batch_data , np.expand_dims(data_tmp , axis = 0)], axis = 0)
        if batch == 0:
            data = batch_data
        else:
            data = np.concatenate([data, batch_data], axis = 0)
    return data

def read_Y_pre_file(
    data_index, 
    Path, 
    H_resize, 
    W_resize
    ):
    
    for i, file_name in enumerate(data_index):
        # Get the file name
        file_name = file_name.split('.')[0]
        
        # Load the npz file
        y_pre = np.load(Path + file_name + '.npz')
        
        Y_pre = y_pre[y_pre.keys()[0]]
    
        if i==0:
            data = np.expand_dims(Y_pre, axis=0)
        else:
            data = np.concatenate([data, np.expand_dims(Y_pre, axis=0)], axis=0)
            
    return data

def read_csv_file( # Recall how to read csv file. Not to use.
    file
    ): 
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for iter, row in enumerate(reader):
            print(row)

def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, 32 + 8, 32 + 8)
        # Randomly crop a [_HEIGHT, _WIDTH] section of the image. 
        image = tf.random_crop(image, [32, 32, 3])
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image
            
def parse_record(
    raw_record,
    height,
    width,
    class_num
    ):
    
    """Parse CIFAR-10 image and label from a raw record."""
    # Every record consists of a label followed by the image, with a fixed number
    # of bytes for each.

    label_bytes = 1
    image_bytes = height * width * 3
    record_bytes = label_bytes + image_bytes
    # Convert bytes to a vector of uint8 that is record_bytes long.
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, class_num)
    
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        record_vector[label_bytes:record_bytes], [3, height, width])
    
    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    
    return image, label

"""	
def preprocess_image(
    images,
    is_training
    ):
    
    [Num, Height, Width, Depth] = images.get_shape().as_list()
    
    for iter in range(Num):
        image = images[iter]
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(image, Height + 8, Width + 8)
        # Randomly crop a [Height, Width] section of the image.
        image = tf.random_crop(image, [Height, Width, Depth])
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        ##
        image = tf.cond(is_training, lambda: image, lambda: images[iter])
        # Substract off the mean and divide by the cariance of the pixels.
        #image_out = tf.image.per_image_standardization(image)
            
        if iter == 0:
            images_out = tf.expand_dims(image, axis = 0)
        else:
            images_out = tf.concat([images_out, tf.expand_dims(image, axis = 0)], axis = 0)

    return images_out
"""
#===============#
#   File Save   #
#===============#
def Save_file_as_csv(
    Path, 
    file
    ):
    
    np.savetxt(Path + '.csv', file, delimiter=",")

#==============#
#   Analyzer   #
#==============#
def Analyzer(
    Analysis, 
    net, 
    type, 
    kernel_shape            = None, 
    stride                  = 0, 
    group                   = 1,
    is_depthwise            = False,
    IS_TERNARY              = False,
    IS_QUANTIZED_ACTIVATION = False,
    padding                 = 'SAME',
    name                    = None
    ):
    if net !=None:
        [B, H, W, D] = net.get_shape().as_list() # B:Batch // H:Height // W:Width // D:Depth
        h = 0 
        w = 0 
        i = 0 
        o = 0
    
    if kernel_shape!=None:
        [h, w, i, o] = kernel_shape # h:Kernel Height // w:Kernel Width // i:input channel // o:output channel
    
    if is_depthwise:
        name = name + '_Depthwise'
        o = 1
    else: 
        group = 1
    
    if IS_QUANTIZED_ACTIVATION:
        activation_bits = 16
    else:
        activation_bits = 32
        
    weight_bits = 0
    
    if type=='CONV':
        if IS_TERNARY:
            weight_bits = 2
        else:
            weight_bits = 32
            
        if padding=='SAME':
            if is_depthwise:
                macc = H*W*h*w*D / (stride*stride)
                comp = 0
                add  = 0
                div  = 0
                exp  = 0
                activation = H*W*D / (stride*stride)
                param = h*w*i
            else:
                macc = ((h*w*i) / group) * ((H*W*o) / (stride*stride))
                comp = 0
                add  = 0
                div  = 0
                exp  = 0
                activation = H*W*o / (stride*stride)
                param = h*w*i*o
        elif padding=='VALID':
            if is_depthwise:
                macc = (H-h+1)*(W-w+1)*h*w*D / (stride*stride)
                comp = 0
                add  = 0
                div  = 0
                exp  = 0
                activation = (H-h+1)*(W-w+1)*D / (stride*stride)
                param = h*w*i
            else:
                macc = ((h*w*i) / group) * (((H-h+1)*(W-w+1)*o) / (stride*stride))
                comp = 0
                add  = 0
                div  = 0
                exp  = 0
                activation = (H-h+1)*(W-w+1)*o / (stride*stride) 
                param = h*w*i*o
    elif type=='POOL':
        macc = 0
        comp = h*w*H*W*D / (stride*stride)
        add  = 0
        div  = 0
        exp  = 0
        activation = H*W*D / (stride*stride)
        param = 0
    elif type=='UNPOOL':
        macc = 0
        comp = 0
        add  = 0
        div  = 0
        activation = H*W*D*stride*stride
        param = 0
    elif type=='RELU':
        macc = 0
        comp = H*W*D
        add  = 0
        div  = 0
        exp  = 0
        activation = H*W*D
        param = 0
    elif type=='DATA':
        macc = 0
        comp = 0
        add  = 0
        div  = 0
        exp  = 0
        activation = H*W*D
        param = 0
    elif type=='ADD':
        macc = 0
        comp = 0
        add  = H*W*D
        div  = 0
        exp  = 0
        activation = H*W*D
        param = 0
    elif type=='TOTAL':
        name   		= 'Total' 
        H      		= 'None'
        W      		= 'None'
        D      		= 'None'
        h      		= 'None'
        w      		= 'None'
        i      		= 'None' 
        o      		= 'None'
        stride 		= 'None'
        activation  = 'None' 
        param	    = 'None'
    
        for i, key in enumerate(Analysis.keys()):
            if i==0:
                macc   	     = Analysis[key]['Macc'] 
                comp   	     = Analysis[key]['Comp'] 
                add    	     = Analysis[key]['Add'] 
                div    	     = Analysis[key]['Div'] 
                exp    	     = Analysis[key]['Exp'] 
                PE_row       = Analysis[key]['PE Height']
                PE_col       = Analysis[key]['PE Width']
                Macc_Cycle   = Analysis[key]['Macc Cycle'] 
                Input_Cycle  = Analysis[key]['Input Cycle']
                Kernel_Cycle = Analysis[key]['Kernel Cycle']
                Output_Cycle = Analysis[key]['Output Cycle']
                Bottleneck   = Analysis[key]['Bottleneck']
            else:
                macc   	     = macc         + Analysis[key]['Macc'] 
                comp   	     = comp         + Analysis[key]['Comp'] 
                add    	     = add          + Analysis[key]['Add'] 
                div    	     = div          + Analysis[key]['Div'] 
                exp    	     = exp          + Analysis[key]['Exp'] 
                Macc_Cycle   = Macc_Cycle   + Analysis[key]['Macc Cycle'] 
                Input_Cycle  = Input_Cycle  + Analysis[key]['Input Cycle']
                Kernel_Cycle = Kernel_Cycle + Analysis[key]['Kernel Cycle']
                Output_Cycle = Output_Cycle + Analysis[key]['Output Cycle']
                Bottleneck   = Bottleneck   + Analysis[key]['Bottleneck']
    
    # Estimate the Cycle Times which will be taken in hardware
    ## Hardware Environment
    if type!='TOTAL':
        PE_row = 15
        PE_col = 16
        
        Tile_row = 8
        Tile_col = 8
    
        Data_Bits = 32
    
        Memory_Bandwidth = 128
    
        ## Some Meaningful Variable
        # initialization
        Input_Retake_Times  = 1
        Kernel_Retake_Times = 1
        Output_Retake_Times = 1
        # conv
        if type=='CONV':
            if is_depthwise:
                Input_Retake_Times  = 1
                Kernel_Retake_Times = (H / Tile_row) * (W / Tile_col)
            else:
                Input_Retake_Times  = o / ((PE_row / h) * (PE_col / Tile_row))
                Kernel_Retake_Times = (H / Tile_row) * (W / Tile_col)
        
            if Input_Retake_Times<1:
                Input_Retake_Times = 1
            if Kernel_Retake_Times<1: 
                Kernel_Retake_Times = 1
    
        # Data Access In Cycle
        Data_Access_In_One_Cycle = Memory_Bandwidth / Data_Bits
    
        ## Cycle times
        if type=='CONV':
            Macc_Cycle   = macc / (PE_row * PE_col)
            Input_Cycle  = (H * W * D) * (Input_Retake_Times) / (Data_Access_In_One_Cycle)		 
            Kernel_Cycle = (h * w * i * o * group) * (Kernel_Retake_Times) / (Data_Access_In_One_Cycle)		
            Output_Cycle = (activation) * (Output_Retake_Times) / (Data_Access_In_One_Cycle)		 
        else:
            Macc_Cycle   = comp + add +div + exp 
            Input_Cycle  = (H * W * D) * (Input_Retake_Times) / (Data_Access_In_One_Cycle)		 
            Kernel_Cycle = 0 
            Output_Cycle = (activation) * (Output_Retake_Times) / (Data_Access_In_One_Cycle)		 
        
        # Bottleneck
        Bottleneck = max(Macc_Cycle, Input_Cycle + Kernel_Cycle + Output_Cycle)
        
    
    components = {'name'            : name, 
                  'Input Height'    : H,
                  'Input Width'     : W,
                  'Input Depth'     : D,
                  'Type'            : type,
                  'Kernel Height'   : h,
                  'Kernel Width'    : w,
                  'Kernel Depth'    : i, 
                  'Kernel Number'   : o,
                  'Stride'          : stride,
                  'Kernel Bits'     : weight_bits,
                  'Macc'            : macc, 
                  'Comp'            : comp, 
                  'Add'             : add, 
                  'Div'             : div, 
                  'Exp'             : exp, 
                  'Activation'      : activation, 
                  'Activation Bits' : activation_bits,
                  'Param'           : param,
                  'PE Height'       : PE_row,
                  'PE Width'        : PE_col,
                  'Macc Cycle'      : Macc_Cycle,
                  'Input Cycle'     : Input_Cycle,
                  'Kernel Cycle'    : Kernel_Cycle,
                  'Output Cycle'    : Output_Cycle,
                  'Bottleneck'      : Bottleneck}
    
    if len(Analysis.keys())<10:
        layer_now = '00' + str(len(Analysis.keys()))
    elif len(Analysis.keys())<100:
        layer_now = '0' + str(len(Analysis.keys()))
    else:
        layer_now = str(len(Analysis.keys()))
    
    Analysis.update({'layer' + layer_now : components})
    
def Save_Analyzsis_as_csv(
    Analysis, 
    FILE
    ):
    
    Analyzer(Analysis, net=None, type='TOTAL')
    
    keys = sorted(Analysis.keys())
    
    components = np.array(['name'            ,
                           'Input Height'    ,
                           'Input Width'     ,
                           'Input Depth'     ,
                           'Type'            ,
                           'Kernel Height'   ,
                           'Kernel Width'    ,
                           'Kernel Depth'    ,
                           'Kernel Number'   ,
                           'Stride'          ,
                           'Kernel Bits'     ,
                           'Macc'            ,
                           'Comp'            ,
                           'Add'             ,
                           'Div'             ,
                           'Exp'             ,
                           'Activation'      ,
                           'Activation Bits' ,
                           'Param'           ,
                           'PE Height'       ,
                           'PE Width'        ,
                           'Macc Cycle'      ,
                           'Input Cycle'     ,
                           'Kernel Cycle'    ,
                           'Output Cycle'    ,
                           'Bottleneck'      ])
    
    Ana = np.expand_dims(components, axis=1)
    for i, key in enumerate(keys):
        Ana = np.concatenate([Ana, np.expand_dims(np.array([Analysis[key][x] for x in components]), axis=1)], axis=1)
    
    np.savetxt(FILE + '.csv', Ana, delimiter=",", fmt="%s") 

#=========================#
#   Training Components   #
#=========================#	
def one_of_k(
    target, 
    class_num
    ):
    
    target.astype('int64')
    
    one_of_k_target = np.zeros([np.shape(target)[0], np.shape(target)[1], np.shape(target)[2], class_num])
    
    meshgrid_target = np.meshgrid(np.arange(np.shape(target)[1]), np.arange(np.shape(target)[0]), np.arange(np.shape(target)[2]))
    
    one_of_k_target[meshgrid_target[1], meshgrid_target[0], meshgrid_target[2], target] = 1
    
    return one_of_k_target

def shuffle_image(
    x, 
    y
    ):
    
    [NUM, HEIGHT, WIDTH, DIM] = np.shape(x)
    
    shuffle_index = np.arange(NUM)
    np.random.shuffle(shuffle_index)
    
    x_shuffle = x[shuffle_index, :, :, :]
    y_shuffle = y[shuffle_index, :, :, :]
    
    return x_shuffle, y_shuffle

def per_class_accuracy(
    prediction, 
    batch_ys
    ):
    print("  Per Class Accuracy	: ")
    [BATCH, HEIGHT, WIDTH, CLASS_NUM] = np.shape(batch_ys)
    correct_num = np.zeros([CLASS_NUM, 1])
    total_num = np.zeros([CLASS_NUM, 1])
    
    print_per_row = 10
    cn = np.zeros([print_per_row], np.int32)
    tn = np.zeros([print_per_row], np.int32)
    
    for i in range(CLASS_NUM):
        y_tmp = np.equal(np.argmax(batch_ys, -1), i)
        p_tmp = np.equal(np.argmax(prediction, -1), i)
        total_num = np.count_nonzero(y_tmp)
        zeros_num = np.count_nonzero( (p_tmp+y_tmp) == 0)
        correct_num = np.count_nonzero(np.equal(y_tmp, p_tmp)) - zeros_num
        if total_num == 0:
            accuracy = -1
        else:
            accuracy = float(correct_num) / float(total_num)
        
    
        if CLASS_NUM <= 15:
            print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=correct_num, target=total_num))
        else:
            iter = i % print_per_row
            cn[iter] = correct_num
            tn[iter] = total_num
            if i%print_per_row==0:
                print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=np.sum(cn), target=np.sum(tn)))

def tenarized_bd(
    weights_collection,
    biases_collection,
    weights_bd_ratio,   # percentile. Ex 50=50%
    biases_bd_ratio,    # percentile. Ex 50=50%
    sess
    ):
    
    NUM = len(weights_collection)
    
    for i in range(NUM):
        w     = np.absolute(sess.run(weights_collection[i]))
        b     = np.absolute(sess.run(biases_collection [i]))
        
        w_bd  = np.percentile(w, weights_bd_ratio) 
        b_bd  = np.percentile(b, biases_bd_ratio )
    
    
        if i==0:
            weights_bd = np.array([[-w_bd, w_bd]])
            biases_bd  = np.array([[-b_bd, b_bd]])
        else:
            weights_bd = np.concatenate([weights_bd, np.array([[-w_bd, w_bd]])], axis=0)
            biases_bd  = np.concatenate([biases_bd , np.array([[-b_bd, b_bd]])], axis=0)
    
    weights_table = [-1, 0, 1]
    biases_table  = [-1, 0, 1]
    
    return weights_bd, biases_bd, weights_table, biases_table

def assign_ternary_boundary(
    ternary_weights_bd_collection, 
    ternary_biases_bd_collection, 
    ternary_weights_bd,
    ternary_biases_bd,
    sess
    ):
    
    NUM = len(ternary_weights_bd_collection)
    
    for i in range(NUM):
        sess.run(ternary_weights_bd_collection[i].assign(ternary_weights_bd[i]))
        sess.run( ternary_biases_bd_collection[i].assign(ternary_biases_bd [i]))
                
def quantized_m_and_f(
    activation_collection, 
    is_quantized_activation_collection, 
    xs, 
    Model_dict, 
    batch_xs, 
    sess
    ):
     
    NUM = len(activation_collection)
    for i in range(NUM):
        activation = sess.run(activation_collection[i], feed_dict={xs: batch_xs, is_quantized_activation_collection[i]: False})
    
        mantissa = 16
        var = np.var( ((activation+1e-1)*9)/(pow(2, mantissa)*pow(2, mantissa)) ) + 0.001
        fraction = -int(np.log2(var))-1
        #fraction = 3
    
        if i==0:
            m = np.array([[mantissa]])
            f = np.array([[fraction]])
        else:
            m = np.concatenate([m, np.array([[mantissa]])], axis=0)
            f = np.concatenate([f, np.array([[fraction]])], axis=0)
            
    return m, f

def assign_quantized_m_and_f(
    mantissa_collection, 
    fraction_collection, 
    m, 
    f, 
    sess
    ):
    
    NUM = len(mantissa_collection)
    for i in range(NUM):
        sess.run(mantissa_collection[i].assign(m[i]))
        sess.run(fraction_collection[i].assign(f[i]))

def similar_group(
    inputs_and_kernels,
    sess
    ):
    
    kernel_values_per_layer = {}
    
    for layer in range(len(inputs_and_kernels)):
        try:
            inputs_tensor = inputs_and_kernels['layer%d'%layer]['inputs']
            kernels_tensor = inputs_and_kernels['layer%d'%layer]['kernels']
            for kernel_iter, kernel in enumerate(kernels_tensor):
                if kernel.name.split('_')[-1] == 'weights:0':# or kernel.name.split('_')[-1] == 'biases:0':
                    kernel_value = sess.run(kernel)
                    kernel_value = np.reshape(kernel_value, [np.size(kernel_value)/np.shape(kernel_value)[-1], np.shape(kernel_value)[-1]])
                    try:
                        kernel_values = np.concatenate([kernel_values, kernel_value], axis = 0)
                    except:
                        kernel_values = kernel_value
        except:
            inputs_tensor = None
            kernels_tensor = None
            kernel_values = None
                    
        kernel_values_per_layer.update({"layer%d"%layer: kernel_values})
        try:
            np.savetxt('layer%d.csv' %layer, kernel_values, delimiter=",")
        except:
            None
        
        ##print(np.shape(kernel_values))
        
    return kernel_values_per_layer
        
#========================#
#   Testing Components   #
#========================#
def compute_accuracy(
    xs, 
    ys, 
    is_training,
    is_quantized_activation,
    Model_dict, 
    QUANTIZED_NOW, 
    prediction_list, 
    #v_xs, 
    #v_ys, 
    BATCH_SIZE, 
    sess
    ):
    
    test_batch_size = BATCH_SIZE
    #batch_num = int(len(v_xs) / test_batch_size)
    batch_num = 10000/128
    total_correct_num_top1 = 0
    total_correct_num_top2 = 0
    total_correct_num_top3 = 0
    total_error_num_top1 = 0
    total_error_num_top2 = 0
    total_error_num_top3 = 0
    
    dict_test = {}
    for layer in range(len(Model_dict)):
        dict_index = is_quantized_activation['layer'+str(layer)]
        dict_data = Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION']=='TRUE' and QUANTIZED_NOW
        dict_test.update({dict_index: dict_data})
    
    for i in range(batch_num): 
        #v_xs_part = v_xs[i*test_batch_size : (i+1)*test_batch_size]
        #v_ys_part = v_ys[i*test_batch_size : (i+1)*test_batch_size]
        dict = dict_test
        dict_test.update({ is_training: False}) #xs: v_xs_part,

        Y_pre_list, v_ys_part = sess.run([prediction_list, ys], feed_dict = dict_test)				
        
        # Combine Different Device Prediction
        for d in range(len(Y_pre_list)):
            if d == 0:
                Y_pre = Y_pre_list[d]
            else:
                Y_pre = np.concatenate([Y_pre, Y_pre_list[d]], axis = 0)
                
        if Y_pre.ndim == 3:
            Y_pre = np.expand_dims(Y_pre, axis = 0)
        
        if i%1000 == 0:
            Y_pre_part = Y_pre
        else:
            Y_pre_part = np.concatenate([Y_pre_part, Y_pre], axis=0)
        
        if i == 0:
            Y_pre_total = Y_pre_part
        else:
            Y_pre_total = np.concatenate([Y_pre_total, Y_pre_part], axis=0)
        
        top1 = np.argsort(-Y_pre, axis=-1)[:, 0] 
        top2 = np.argsort(-Y_pre, axis=-1)[:, 1] 
        top3 = np.argsort(-Y_pre, axis=-1)[:, 2] 

        correct_prediction_top1 = np.equal(top1, np.argmax(v_ys_part, -1))
        correct_prediction_top2 = np.equal(top2, np.argmax(v_ys_part, -1)) | correct_prediction_top1
        correct_prediction_top3 = np.equal(top3, np.argmax(v_ys_part, -1)) | correct_prediction_top2
        
        correct_num_top1 = np.sum(correct_prediction_top1 == True, dtype = np.float32)
        correct_num_top2 = np.sum(correct_prediction_top2 == True, dtype = np.float32)
        correct_num_top3 = np.sum(correct_prediction_top3 == True, dtype = np.float32)
        
        error_num_top1 = np.sum(correct_prediction_top1 == False, dtype = np.float32)
        error_num_top2 = np.sum(correct_prediction_top2 == False, dtype = np.float32)
        error_num_top3 = np.sum(correct_prediction_top3 == False, dtype = np.float32)
        
        total_correct_num_top1 = total_correct_num_top1 + correct_num_top1
        total_correct_num_top2 = total_correct_num_top2 + correct_num_top2
        total_correct_num_top3 = total_correct_num_top3 + correct_num_top3
        
        total_error_num_top1 = total_error_num_top1 + error_num_top1
        total_error_num_top2 = total_error_num_top2 + error_num_top2
        total_error_num_top3 = total_error_num_top3 + error_num_top3
        
        accuracy_top1 = total_correct_num_top1 / (total_correct_num_top1 + total_error_num_top1)
        accuracy_top2 = total_correct_num_top2 / (total_correct_num_top2 + total_error_num_top2)  
        accuracy_top3 = total_correct_num_top3 / (total_correct_num_top3 + total_error_num_top3)  
        
        ## if you just want to see the result of top1 accuarcy, then using follwing codes
        if i==0:
        	result = np.argmax(Y_pre, -1)
        else:
        	result = np.concatenate([result, np.argmax(Y_pre, -1)], axis=0)
        
        ## if you want to see the top2 result, then using the following codes
        #if i==0:
        #	result = np.multiply(~correct_prediction_top2, top1) + np.multiply(correct_prediction_top2, np.argmax(v_ys_part, -1))
        #else:
        #	result = np.concatenate([result, np.multiply(~correct_prediction_top2, top1) + np.multiply(correct_prediction_top2, np.argmax(v_ys_part, -1))], axis=0)
        
        ## if you want to see the top3 result, then using the following codes
        #if i==0:
        #    result = np.multiply(~correct_prediction_top3, top1) + np.multiply(correct_prediction_top3, np.argmax(v_ys_part, -1))
        #else:
        #    result = np.concatenate([result, np.multiply(~correct_prediction_top3, top1) + np.multiply(correct_prediction_top3, np.argmax(v_ys_part, -1))], axis=0)
        
        print("\r{} / {} -> Accuracy:{}" .format(i*test_batch_size, 10000, accuracy_top1), end = "")
        
    print("")    
    return result, accuracy_top1, accuracy_top2, accuracy_top3, Y_pre_total

def color_result(
    result
    ):
    
    #***************************************#
    #	class0 : (	128 	128 	128	)	#
    #	class1 : (	128 	0 		0	)	#
    #	class2 : (	192 	192 	128	)	#
    #	class3 : (	128 	64 		128	)	#
    #	class4 : (	0 		0 		192	)	#
    #	class5 : (	128 	128 	0	)	#
    #	class6 : (	192 	128 	128	)	#
    #	class7 : (	64 		64 		128	)	#
    #	class8 : (	64 		0 		128	)	#
    #	class9 : (	64 		64 		0	)	#
    #	class10 : (	0		128 	192	)	#
    #	class11 : (	0		0		0	)	#
    #***************************************#
    shape = np.shape(result)
    
    RGB = np.zeros([shape[0], shape[1], shape[2], 3], np.uint8)
    
    for i in range(shape[0]):
        for x in range(shape[1]):
            for y in range(shape[2]):
                if result[i][x][y] == 0:
                    RGB[i][x][y][0] = np.uint8(128)
                    RGB[i][x][y][1] = np.uint8(128)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 1:
                    RGB[i][x][y][0] = np.uint8(128) 
                    RGB[i][x][y][1] = np.uint8(0)
                    RGB[i][x][y][2] = np.uint8(0) 
                elif result[i][x][y] == 2:
                    RGB[i][x][y][0] = np.uint8(192)
                    RGB[i][x][y][1] = np.uint8(192)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 3:
                    RGB[i][x][y][0] = np.uint8(128)
                    RGB[i][x][y][1] = np.uint8(64)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 4:
                    RGB[i][x][y][0] = np.uint8(0)
                    RGB[i][x][y][1] = np.uint8(0)
                    RGB[i][x][y][2] = np.uint8(192)
                elif result[i][x][y] == 5:
                    RGB[i][x][y][0] = np.uint8(128)
                    RGB[i][x][y][1] = np.uint8(128)
                    RGB[i][x][y][2] = np.uint8(0)
                elif result[i][x][y] == 6:
                    RGB[i][x][y][0] = np.uint8(192)
                    RGB[i][x][y][1] = np.uint8(128)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 7:
                    RGB[i][x][y][0] = np.uint8(64)
                    RGB[i][x][y][1] = np.uint8(64)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 8:
                    RGB[i][x][y][0] = np.uint8(64)
                    RGB[i][x][y][1] = np.uint8(0)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 9:
                    RGB[i][x][y][0] = np.uint8(64)
                    RGB[i][x][y][1] = np.uint8(64)
                    RGB[i][x][y][2] = np.uint8(0)
                elif result[i][x][y] == 10:
                    RGB[i][x][y][0] = np.uint8(0)
                    RGB[i][x][y][1] = np.uint8(128)
                    RGB[i][x][y][2] = np.uint8(192)
                elif result[i][x][y] == 11:
                    RGB[i][x][y][0] = np.uint8(0)
                    RGB[i][x][y][1] = np.uint8(0)
                    RGB[i][x][y][2] = np.uint8(0)
    return RGB

def Save_result_as_image(
    Path, 
    result, 
    file_index
    ):
    
    for i, target in enumerate(result):
        scipy.misc.imsave(Path + file_index[i], target)
	
def Save_result_as_npz(	
    Path, 
    result, 
    file_index,
    sess
    ):
    
    for i, target in enumerate(result):
        file_index[i] = file_index[i].split('.')[0]
        np.savez(Path + file_index[i], target)
	

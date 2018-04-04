from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
from prettytable import PrettyTable
import csv
import copy
import pickle
import random
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
from sys import platform
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from functools import reduce

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import Model
import vgg_preprocessing
import imagenet_preprocessing
import resnet_model

SHOW_MODEL = False
IS_IN_IPYTHON = False

#=========#
#   Run   #
#=========#
def run_training(
    Hyperparameter       , 
    # Info               
    FLAGs                ,
    Epoch                ,
    Global_Epoch         ,
    IS_HYPERPARAMETER_OPT,
    # Path               
    Dataset_Path         ,
    Y_pre_Path           ,
    trained_model_path   ,
    trained_model        
    ):
    
    
    if IS_IN_IPYTHON:
        Dataset           = FLAGs['Dataset']
        Model_first_name  = FLAGs['Model_1st']
        Model_second_name = FLAGs['Model_2nd']
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs['BatchSize']
    else:
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
        train_num = 367
        valid_num = 101
        test_num = 356
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
        train_num = 50000
        valid_num = 0
        test_num = 10000
    elif Dataset=='ILSVRC2012': 
        H_Resize = 224
        W_Resize = 224
        class_num = 1001
        train_num = 1281167
        valid_num = 0
        test_num = 50000
        
    #----------------#
    #   Model_dict   #
    #----------------#
    if platform == 'win32':
        Model_dict = Model_dict_Generator('Model\\' + Model_first_name + '_Model\\' + Model_Name + '.csv', class_num)
    else:
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
        HP.update({'train_num'                 : train_num })
        HP.update({'valid_num'                 : valid_num })
        HP.update({'test_num'                  : test_num  })
        if Model_first_name == 'ResNet':
            HP.update({'LR'                        : 0.0000125 }) 
            HP.update({'LR_Strategy'               : '3times'  })
            HP.update({'LR_Final'                  : 1e-3      })
            HP.update({'LR_Decade'                 : 10        })   
            HP.update({'LR_Decade_1st_Epoch'       : 30        })
            HP.update({'LR_Decade_2nd_Epoch'       : 60        })
            HP.update({'LR_Decade_3rd_Epoch'       : 80        }) 
            HP.update({'LR_Decade_4th_Epoch'       : 90        }) 
            HP.update({'L2_Lambda'                 : 2e-4      })
            HP.update({'Opt_Method'                : 'Momentum'})
            HP.update({'Momentum_Rate'             : 0.9       })
            HP.update({'IS_STUDENT'                : False     })
            HP.update({'Ternary_Epoch'             : 50        })
            HP.update({'Quantized_Activation_Epoch': 100       })
            HP.update({'Dropout_Rate'              : 0.0       })
        elif Model_first_name == 'DenseNet':
            HP.update({'LR'                        : 1e-1      }) 
            HP.update({'LR_Strategy'               : '3times'  })
            HP.update({'LR_Final'                  : 1e-3      })
            HP.update({'LR_Decade'                 : 10        })   
            HP.update({'LR_Decade_1st_Epoch'       : 150       })
            HP.update({'LR_Decade_2nd_Epoch'       : 225       })
            HP.update({'LR_Decade_3rd_Epoch'       : 300       }) 
            HP.update({'L2_Lambda'                 : 2e-4      })
            HP.update({'Opt_Method'                : 'Momentum'})
            HP.update({'Momentum_Rate'             : 0.9       })
            HP.update({'IS_STUDENT'                : False     })
            HP.update({'Ternary_Epoch'             : 50        })
            HP.update({'Quantized_Activation_Epoch': 100       })
            HP.update({'Dropout_Rate'              : 0.0       })
        elif Model_first_name == 'MobileNet':
            HP.update({'LR'                        : 1e-1      }) 
            HP.update({'LR_Strategy'               : '3times'  })
            HP.update({'LR_Final'                  : 1e-3      })
            HP.update({'LR_Decade'                 : 10        })   
            HP.update({'LR_Decade_1st_Epoch'       : 150       })
            HP.update({'LR_Decade_2nd_Epoch'       : 225       })
            HP.update({'LR_Decade_3rd_Epoch'       : 300       }) 
            HP.update({'L2_Lambda'                 : 2e-4      })
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
        HP.update({'LR_Decade_3rd_Epoch'       : int(HP_dict['LR_Decade_3rd_Epoch'])       })
        HP.update({'L2_Lambda'                 : float(HP_dict['L2_Lambda'])               })
        HP.update({'Opt_Method'                : HP_dict['Opt_Method']                     })
        HP.update({'Momentum_Rate'             : float(HP_dict['Momentum_Rate'])           })
        HP.update({'IS_STUDENT'                : HP_dict['IS_STUDENT'] == 'TRUE'           })
        HP.update({'Ternary_Epoch'             : int(HP_dict['Ternary_Epoch'])             })
        HP.update({'Quantized_Activation_Epoch': int(HP_dict['Quantized_Activation_Epoch'])})
        HP.update({'Dropout_Rate'              : float(HP_dict['Dropout_Rate'])            })
    
    print("\033[1;32mBATCH SIZE\033[0m : {}" .format(HP['Batch_Size']))
    
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
                               'LR_Decade_3rd_Epoch'       ,
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
        Global_Epoch          = Global_Epoch          ,
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
    
    
    if IS_IN_IPYTHON:
        Dataset           = FLAGs['Dataset']
        Model_first_name  = FLAGs['Model_1st']
        Model_second_name = FLAGs['Model_2nd']
        BATCH_SIZE        = FLAGs['BatchSize']
    else:
        Dataset           = FLAGs.Dataset
        Model_first_name  = FLAGs.Model_1st
        Model_second_name = FLAGs.Model_2nd
        BATCH_SIZE        = FLAGs.BatchSize
    
    if Dataset=='CamVid':   # original : [360, 480, 12]
        H_Resize = 360
        W_Resize = 480
        class_num = 12
        train_num = 367
        valid_num = 101
        test_num = 356
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
        train_num = 50000
        valid_num = 0
        test_num = 10000
    elif Dataset=='ILSVRC2012': 
        H_Resize = 224
        W_Resize = 224
        class_num = 1001
        train_num = 1281167
        valid_num = 0
        test_num = 50000
        
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
        test_num                   = test_num              ,
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
    
def run_pruning(
    # Info               
    FLAGs                ,
    Epoch                ,
    Global_Epoch         ,
    # Path               
    Dataset_Path         ,
    Y_pre_Path           ,
    pruning_model_path   ,
    pruning_model        
    ):
    
    if IS_IN_IPYTHON:
        Dataset           = FLAGs['Dataset']
        Model_first_name  = FLAGs['Model_1st']
        Model_second_name = FLAGs['Model_2nd']
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs['BatchSize']
        Pruning_Strategy  = FLAGs['Pruning_Strategy']
    else:
        Dataset           = FLAGs.Dataset
        Model_first_name  = FLAGs.Model_1st
        Model_second_name = FLAGs.Model_2nd
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs.BatchSize
        Pruning_Strategy  = FLAGs.Pruning_Strategy
    
    Model_Name = Model_first_name + '_' + Model_second_name
    
    if Dataset=='CamVid':   # original : [360, 480, 12]
        H_Resize = 360
        W_Resize = 480
        class_num = 12
        train_num = 367
        valid_num = 101
        test_num = 356
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
        train_num = 50000
        valid_num = 0
        test_num = 10000
    elif Dataset=='ILSVRC2012': 
        H_Resize = 224
        W_Resize = 224
        class_num = 1001
        train_num = 1281167
        valid_num = 0
        test_num = 50000
        
    #----------------#
    #   Model_dict   #
    #----------------#
    Model_dict = Model_dict_Generator(pruning_model_path + 'model.csv', class_num)
    
    #-----------------------------------#
    #   Hyperparameter : User Defined   #
    #-----------------------------------#
    HP = {}
    HP.update({'Batch_Size'                : BATCH_SIZE      })
    HP.update({'Epoch'                     : EPOCH           })
    HP.update({'H_Resize'                  : H_Resize        })
    HP.update({'W_Resize'                  : W_Resize        })
    HP.update({'train_num'                 : train_num       })
    HP.update({'valid_num'                 : valid_num       })
    HP.update({'test_num'                  : test_num        })
    HP.update({'Pruning_Strategy'          : Pruning_Strategy})
    if Model_first_name == 'ResNet':
        HP.update({'LR'                        : 0.0000125        })  
        HP.update({'LR_Strategy'               : '3times'        })
        HP.update({'LR_Final'                  : 1e-3            })
        HP.update({'LR_Decade'                 : 10              }) 
        HP.update({'LR_Decade_1st_Epoch'       : 30              })
        HP.update({'LR_Decade_2nd_Epoch'       : 150             })
        HP.update({'LR_Decade_3rd_Epoch'       : 150             })
        HP.update({'L2_Lambda'                 : 2e-4            })
        HP.update({'Opt_Method'                : 'Momentum'      })
        HP.update({'Momentum_Rate'             : 0.9             })
        HP.update({'IS_STUDENT'                : False           })
        HP.update({'Ternary_Epoch'             : 50              })
        HP.update({'Quantized_Activation_Epoch': 100             })
        HP.update({'Dropout_Rate'              : 0.0             })
        HP.update({'Pruning_Propotion_Per'     : 0.1             })
        HP.update({'Pruning_Retrain_Epoch'     : 50              })
    elif Model_first_name == 'DenseNet':
        HP.update({'LR'                        : 1e-3            })
        HP.update({'LR_Strategy'               : '3times'        })
        HP.update({'LR_Final'                  : 1e-3            })
        HP.update({'LR_Decade'                 : 10              })
        HP.update({'LR_Decade_1st_Epoch'       : 100             })
        HP.update({'LR_Decade_2nd_Epoch'       : 300             })
        HP.update({'LR_Decade_3rd_Epoch'       : 300             })
        HP.update({'L2_Lambda'                 : 2e-4            })
        HP.update({'Opt_Method'                : 'Momentum'      })
        HP.update({'Momentum_Rate'             : 0.9             })
        HP.update({'IS_STUDENT'                : False           })
        HP.update({'Ternary_Epoch'             : 50              })
        HP.update({'Quantized_Activation_Epoch': 100             })
        HP.update({'Dropout_Rate'              : 0.0             })
        HP.update({'Pruning_Propotion_Per'     : 0.1             })
        HP.update({'Pruning_Retrain_Epoch'     : 200             })
    elif Model_first_name == 'MobileNet':
        HP.update({'LR'                        : 1e-3            })
        HP.update({'LR_Strategy'               : '3times'        })
        HP.update({'LR_Final'                  : 1e-3            })
        HP.update({'LR_Decade'                 : 10              })
        HP.update({'LR_Decade_1st_Epoch'       : 100             })
        HP.update({'LR_Decade_2nd_Epoch'       : 300             })
        HP.update({'LR_Decade_3rd_Epoch'       : 300             })
        HP.update({'L2_Lambda'                 : 2e-4            })
        HP.update({'Opt_Method'                : 'Momentum'      })
        HP.update({'Momentum_Rate'             : 0.9             })
        HP.update({'IS_STUDENT'                : False           })
        HP.update({'Ternary_Epoch'             : 50              })
        HP.update({'Quantized_Activation_Epoch': 100             })
        HP.update({'Dropout_Rate'              : 0.0             })
        HP.update({'Pruning_Propotion_Per'     : 0.1             })
        HP.update({'Pruning_Retrain_Epoch'     : 200             }) 
        
    print("\033[1;32mBATCH SIZE\033[0m : {}" .format(HP['Batch_Size']))
    
    #---------------------------#
    #    Hyperparameter Save    #
    #---------------------------#
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
                           'LR_Decade_3rd_Epoch'       ,
                           'L2_Lambda'                 ,
                           'Opt_Method'                ,
                           'Momentum_Rate'             ,
                           'IS_STUDENT'                ,
                           'Ternary_Epoch'             ,
                           'Quantized_Activation_Epoch',
                           'Dropout_Rate'              ,
                           'Pruning_Propotion_Per'     ,
                           'Pruning_Retrain_Epoch'     ,
                           'Pruning_Strategy'          ])
    
    for iter, component in enumerate(components):
        if iter == 0:
            HP_csv = np.array([HP[component]])
        else:
            HP_csv = np.concatenate([HP_csv, np.array([HP[component]])], axis=0)
            
    components = np.expand_dims(components, axis=1)
    HP_csv = np.expand_dims(HP_csv, axis=1)
    HP_csv = np.concatenate([HP_csv, components], axis=1)
    
    Global_Epoch = Global_Epoch % HP['Pruning_Retrain_Epoch']
    Model_Path, Model = Pruning(
        Model_dict         = Model_dict           ,        
        Dataset	           = Dataset              ,
        Dataset_Path       = Dataset_Path         ,
        Y_pre_Path         = Y_pre_Path           ,
        class_num          = class_num            ,
        HP                 = HP                   ,
        Global_Epoch       = Global_Epoch         ,
        weights_bd_ratio   = 50                   ,
        biases_bd_ratio    = 50                   ,    
        HP_csv             = HP_csv               ,
        Model_first_name   = Model_first_name     ,
        Model_second_name  = Model_second_name    ,
        pruning_model_path = pruning_model_path   ,
        pruning_model      = pruning_model        )
    
    return Model_Path, Model
    
def run_rebuilding(
    # Info               
    FLAGs                     ,
    Epoch                     ,
    Global_Epoch              ,
    # Path                    
    Dataset_Path              ,
    Y_pre_Path                ,
    rebuilding_model_path_base,
    rebuilding_model_base     ,
    rebuilding_model_path     ,
    rebuilding_model          ,
    index_now
    ):
    
    if IS_IN_IPYTHON:
        Dataset           = FLAGs['Dataset']
        Model_first_name  = FLAGs['Model_1st']
        Model_second_name = FLAGs['Model_2nd']
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs['BatchSize']
    else:
        Dataset           = FLAGs.Dataset
        Model_first_name  = FLAGs.Model_1st
        Model_second_name = FLAGs.Model_2nd
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs.BatchSize
    
    Model_Name = Model_first_name + '_' + Model_second_name
    
    if Dataset=='CamVid':   # original : [360, 480, 12]
        H_Resize = 360
        W_Resize = 480
        class_num = 12
        train_num = 367
        valid_num = 101
        test_num = 356
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
        train_num = 50000
        valid_num = 0
        test_num = 10000
    elif Dataset=='ILSVRC2012': 
        H_Resize = 224
        W_Resize = 224
        class_num = 1000
        train_num = 1300
        valid_num = 0
        test_num = 10
        
    #----------------#
    #   Model_dict   #
    #----------------#
    Model_dict = Model_dict_Generator(rebuilding_model_path + 'model.csv', class_num)
    
    #-----------------------------------#
    #   Hyperparameter : User Defined   #
    #-----------------------------------#
    HP = {}
    HP.update({'Batch_Size'                : BATCH_SIZE      })
    HP.update({'Epoch'                     : EPOCH           })
    HP.update({'H_Resize'                  : H_Resize        })
    HP.update({'W_Resize'                  : W_Resize        })
    HP.update({'train_num'                 : train_num       })
    HP.update({'valid_num'                 : valid_num       })
    HP.update({'test_num'                  : test_num        })
    HP.update({'LR'                        : 1e-3            }) 
    HP.update({'LR_Strategy'               : '3times'        })
    HP.update({'LR_Final'                  : 1e-3            })
    HP.update({'LR_Decade'                 : 10              })   
    HP.update({'LR_Decade_1st_Epoch'       : 80              })
    HP.update({'LR_Decade_2nd_Epoch'       : 120             })
    HP.update({'LR_Decade_3rd_Epoch'       : 200             })
    HP.update({'L2_Lambda'                 : 2e-4            })
    HP.update({'Opt_Method'                : 'Momentum'      })
    HP.update({'Momentum_Rate'             : 0.9             })
    HP.update({'IS_STUDENT'                : False           })
    HP.update({'Ternary_Epoch'             : 50              })
    HP.update({'Quantized_Activation_Epoch': 100             })
    HP.update({'Dropout_Rate'              : 0.0             })
    
    print("\033[1;32mBATCH SIZE\033[0m : {}" .format(HP['Batch_Size']))
    
    #---------------------------#
    #    Hyperparameter Save    #
    #---------------------------#
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
                           'LR_Decade_3rd_Epoch'       ,
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
    
    base_mask = get_mask(
        model_path = rebuilding_model_path_base,
        model      = rebuilding_model_base,
        H_Resize   = H_Resize,
        W_Resize   = W_Resize,
        class_num  = class_num)
    
    base_weights = get_weights(
        model_path = rebuilding_model_path_base,
        model      = rebuilding_model_base,
        H_Resize   = H_Resize,
        W_Resize   = W_Resize,
        class_num  = class_num)
    
    Model_Path, Model = Rebuilding(
        Model_dict                 = Model_dict                ,            
        Dataset	                   = Dataset                   ,
        Dataset_Path               = Dataset_Path              ,
        Y_pre_Path                 = Y_pre_Path                ,
        class_num                  = class_num                 ,
        HP                         = HP                        ,
        Global_Epoch               = Global_Epoch              ,
        weights_bd_ratio           = 50                        ,
        biases_bd_ratio            = 50                        ,    
        HP_csv                     = HP_csv                    ,
        Model_first_name           = Model_first_name          ,
        Model_second_name          = Model_second_name         ,
        rebuilding_model_path_base = rebuilding_model_path_base,
        rebuilding_model_path      = rebuilding_model_path     ,
        rebuilding_model           = rebuilding_model          ,
        base_mask                  = base_mask                 ,
        base_weights               = base_weights              ,
        index_now                  = index_now                 )
    
    return Model_Path, Model

    
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
    Global_Epoch         ,
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
    ## Cifar10
    if Dataset == 'cifar10':
        filenames = [os.path.join(Dataset_Path, 'data_batch_%d.bin'%(i+1)) for i in range(5)]
    ## Imagenet ILSVRC_2012
    elif Dataset == 'ILSVRC2012':
        _NUM_TRAIN_FILES = 1024
        filenames = [os.path.join(Dataset_Path, 'train-%05d-of-01024' % i) for i in range(_NUM_TRAIN_FILES)]
        #_NUM_TRAIN_FILES = 128
        #filenames = [os.path.join(Dataset_Path, 'validation-%05d-of-00128' % i) for i in range(_NUM_TRAIN_FILES)]
        
    xs, ys = input_fn(filenames, class_num, True, HP, Dataset)
    """
    if Dataset == 'ILSVRC2012':
        with tf.Session() as sess:
            image_float = sess.run(xs)
            label_float = sess.run(ys)
            image = np.uint8(image_float)
            label = np.uint8(label_float)
            pdb.set_trace()
            for i in range(64):
                imgplot = plt.imshow(image[i])
                plt.show()
            exit()
    """
    train_data_num = HP['train_num'] 

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
    learning_rate = tf.placeholder(tf.float32)

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
    data_format = "NCHW"
    ## -- Build Model --
    net = xs[0 : HP['Batch_Size']]
    Model_dict_ = copy.deepcopy(Model_dict)
    
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict_, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = HP['Dropout_Rate'],
        data_format             = data_format,
        reuse                   = None)
    """
    Analysis = {}
    max_parameter = 83525632
    inputs_and_kernels = {}
    prune_info_dict = {}
    network = resnet_model.imagenet_resnet_v2(50, 1001, data_format = 'channels_first')
    prediction = network(inputs = net, is_training = True)
    """
    ## -- Model Size --
    # Your grade will depend on this value.
    # Model Size
    Model_Size = 0
    trainable_variables = []
    for iter, variable in enumerate(tf.trainable_variables()):
        Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
        # See all your variables in termainl	
        #print("{}, {}" .format(iter, variable))
        trainable_variables.append(variable.name)
    
    # For Load trained Model
    all_variables = []
    i = 0
    for iter, variable in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=None)):
        # See all your variables in termainl	
        if not any("mask" in s for s in variable.name.split('_:/')) and not any("constant" in s for s in variable.name.split('_:/')):
            print("{}, {}" .format(i, variable))
            all_variables.append(variable.name)
            i = i + 1
            
    np.savetxt('all_variables.csv', all_variables, delimiter=",", fmt="%s")
    exit()
    print("\033[0;36m=======================\033[0m")
    print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size))
    print("\033[0;36m=======================\033[0m")
    
    ## -- Collection --
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
        
    ## -- Loss --
    labels = ys[0 : HP['Batch_Size']]
    
    # L2 Regularization
    l2_norm   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if 'batch_normalization' not in v.name])
    l2_lambda = tf.constant(HP['L2_Lambda'])
    l2_norm   = tf.multiply(l2_lambda, l2_norm)
    
    # Cross Entropy
    cross_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels = labels,
        logits        = prediction)
    
    # Loss
    loss = cross_entropy + l2_norm #HP['L2_Lambda'] * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    
    ## -- Optimizer --
    if HP['Opt_Method']=='Adam':
        opt = tf.train.AdamOptimizer(learning_rate, HP['Momentum_Rate'])
    elif HP['Opt_Method']=='Momentum':
        opt = tf.train.MomentumOptimizer(
            learning_rate = learning_rate, 
            momentum      = HP['Momentum_Rate'])
        
    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        ## Compute Gradients
        var_list = tf.trainable_variables()
        gra_and_var = opt.compute_gradients(loss, var_list = var_list)
            
    ## Apply Gradients
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
    #config.gpu_options.allow_growth = True
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
        #kernel_values_per_layer = similar_group(inputs_and_kernels, sess)
        #--------------------------#
        #   Load trained weights   #
        #--------------------------#
        if trained_model_path!=None and trained_model!=None:
            print("Loading the trained weights ... ")
            print("\033[0;35m{}\033[0m" .format(trained_model_path + trained_model))
            save_path = saver.restore(sess, trained_model_path + trained_model)
            
        """
        path = '/home/2016/b22072117/tmp/resnet_model/'
        name = 'resnet_50'
        obj = load_obj(path, name)
        for _, variable in enumerate(tf.trainable_variables()):
            print(variable.name)
            value = obj[variable.name]
            if np.size(np.shape(value)) == 2:
                value = np.expand_dims(value, axis = 0)
                value = np.expand_dims(value, axis = 0)
            sess.run(tf.assign(variable, value))
        """
        #-------------------#
        #    Computation    #
        #-------------------#
        computation = compute_computation(data_format, sess)
        print("\033[0;36m=======================\033[0m")
        print("\033[0;36m Computation\033[0m = {}" .format(computation))
        print("\033[0;36m=======================\033[0m")
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
            
            ## -- Learning Rate --
            if HP['LR_Strategy'] == '3times':
                if   (Global_Epoch+epoch+1) <= HP['LR_Decade_1st_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 0)
                elif (Global_Epoch+epoch+1) <= HP['LR_Decade_2nd_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 1)
                elif (Global_Epoch+epoch+1) <= HP['LR_Decade_3rd_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 2)
                elif (Global_Epoch+epoch+1) <= HP['LR_Decade_4th_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 3)
                else:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 4)
            
            ## -- Shuffle Data --
            """
            shuffle_index = np.arange(train_data_num)
            np.random.shuffle(shuffle_index)
            train_data   = train_data  [shuffle_index]
            train_target = train_target[shuffle_index]
            """
            ## -- Quantizad Activation --
            if IS_QUANTIZED_ACTIVATION and (epoch+1+Global_Epoch)==HP['Quantized_Activation_Epoch']:
                batch_xs = train_data[0:HP['Batch_Size']]
                # Calculate Each Activation's appropriate mantissa and fractional bit
                m, f = quantized_m_and_f(float32_net_collection, is_quantized_activation_collection, xs, Model_dict, batch_xs, sess)	
                # Assign mantissa and fractional bit to the tensor
                assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
                # Start Quantize Activation
                QUANTIZED_NOW = True
            
            ## -- Ternary --
            if IS_TERNARY and (epoch+1+Global_Epoch)==HP['Ternary_Epoch']:
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
            tStart_Batch = time.time()
            total_batch_iter = int(train_data_num / HP['Batch_Size'])
            for batch_iter in range(total_batch_iter):
                iteration = iteration + 1
                # Train data in BATCH SIZE
                """
                batch_xs = train_data   [ batch_iter*HP['Batch_Size'] : (batch_iter+1)*HP['Batch_Size'] ]
                batch_ys = train_target [ batch_iter*HP['Batch_Size'] : (batch_iter+1)*HP['Batch_Size'] ]
                """
                
                # Run Training Step
                
                feed_dict_train.update({is_training: True, learning_rate: lr}) # xs: batch_xs, ys: batch_ys, 
                _, Loss, Prediction, L2_norm, batch_ys = sess.run(
                    [train_step, loss, prediction, l2_norm, ys], 
                    feed_dict = feed_dict_train)
                """
                feed_dict_train.update({is_training: True, learning_rate: lr}) # xs: batch_xs, ys: batch_ys, 
                Loss, Prediction, L2_norm, batch_ys = sess.run(
                    [loss, prediction, l2_norm, ys], 
                    feed_dict = feed_dict_train)
                """
                # Assign float32 or ternary weight and biases to final weights
                # This may can be placed into GraphKeys.UPDATE_OPS, but i am not sure.
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
                #print(y_pre)
                #print(np.argmax(batch_ys, -1))
                
                if batch_iter % 1 == 0:
                    tEND_Batch = time.time()
                    print("\033[1;34;40mEpoch\033[0m : %3d" %(epoch+Global_Epoch), end=" ")
                    print("\033[1;34;40mData Iteration\033[0m : %7d" %(batch_iter*HP['Batch_Size']), end=" ")
                    print("\033[1;32;40mBatch Accuracy\033[0m : %5f" %(batch_accuracy), end=" ")
                    print("\033[1;32;40mLoss\033[0m : %5f" %(np.mean(Loss)), end=" ")
                    print("\033[1;32;40mL2-norm\033[0m : %5f" %(L2_norm), end=" ")
                    print("\033[1;32;40mLearning Rate\033[0m : %4f" %(lr), end=" ")
                    print("(%2f sec)" %(tEND_Batch-tStart_Batch))
                    tStart_Batch = time.time()
                    
                # Per Class Accuracy
                """
                per_class_accuracy(Prediction, batch_ys)
                """
            
            tEnd = time.time()            
            Train_acc  = total_correct_num  / (total_correct_num + total_error_num)
            Train_loss = Train_loss / iteration 
            print("\r\033[0;33mEpoch{}\033[0m" .format(epoch+Global_Epoch), end = "")
            print(" (Cost {TIME} sec)" .format(TIME = tEnd - tStart))
            print("\033[0;32mLearning Rate    \033[0m : {}".format(lr))
            print("\033[0;32mTraining Accuracy\033[0m : {}".format(Train_acc))
            print("\033[0;32mTraining Loss    \033[0m : {} (l2_norm: {})".format(Train_loss, L2_norm))
            
            # Record Per Epoch Training Result (Finally this will save as the .csv file)
            if epoch==0:
                Train_acc_per_epoch  = np.array([Train_acc ])
                Train_loss_per_epoch = np.array([Train_loss])
            else:
                Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc ])], axis=0)
                Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
            
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
                #Save_Analyzsis_as_csv(Analysis, Dir + 'Analysis')
                
            #----------------------------#
            #   Saving trained weights   #
            #----------------------------#
            if (not IS_HYPERPARAMETER_OPT) and (epoch+1)==HP['Epoch']:
                print("Saving Trained Weights ...")
                save_path = saver.save(sess, Dir + str(Global_Epoch+epoch+1) + ".ckpt")
                print("\033[0;35m{}\033[0m" .format(save_path))
                
            #------------------------#
            #   Saving Computation   #
            #------------------------#
            if (epoch+1)==HP['Epoch']:
                np.savetxt(Dir + 'computation.csv', np.array([computation]), delimiter=",", fmt="%d")
                
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
    Model = str(HP['Epoch']+Global_Epoch) + '.ckpt'
    
    return Model_Path, Model

def Testing(
    # File Read
    Dataset                    ,
    Dataset_Path               ,
    class_num                  ,
    H_Resize                   ,
    W_Resize                   ,
    test_num                   ,
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
    ## cifar10
    if Dataset == 'cifar10':
        filenames = [os.path.join(Dataset_Path, 'test_batch.bin')]
    ## Imagenet ILSVRC_2012
    elif Dataset == 'ILSVRC2012':
        _NUM_TEST_FILES = 128
        filenames = [os.path.join(Dataset_Path, 'validation-%05d-of-00128' % i) for i in range(_NUM_TEST_FILES)]

    xs, ys = input_fn(
        filenames   = filenames,
        class_num   = class_num,
        is_training = False,
        HP          = {'H_Resize'  : H_Resize,
                       'W_Resize'  : W_Resize,
                       'Batch_Size': BATCH_SIZE,
                       'Epoch'     : None,
                       'test_num'  : test_num},
        Dataset     = Dataset
        )
    
    #test_data_index  = np.array(open(Dataset_Path + '/test.txt', 'r').read().splitlines())
    test_data_num = test_num
    
    #----------------------#
    #    Building Model    #
    #----------------------# 
    print("Building Model ...")
    data_format = "NHWC"
    ## -- Build Model --
    net = xs[0 : BATCH_SIZE]
    Model_dict_ = copy.deepcopy(Model_dict)     
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict_, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = None,
        data_format             = data_format,
        reuse                   = None)
    
    ## -- Model Size --
    # Your grade will depend on this value.
    all_variables = tf.trainable_variables() #tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model")
    # Model Size
    Model_Size = 0
    for iter, variable in enumerate(all_variables):
        Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
        # See all your variables in termainl	
        #print("{}, {}" .format(iter, variable))
                     
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
        """
        path = '/home/2016/b22072117/tmp/resnet_model/'
        name = 'resnet_50'
        obj = load_obj(path, name)
        for _, variable in enumerate(tf.trainable_variables()):
            print(variable.name)
            value = obj[variable.name]
            if np.size(np.shape(value)) == 2:
                value = np.expand_dims(value, axis = 0)
                value = np.expand_dims(value, axis = 0)
            if np.sum(sess.run(variable) != value)!=0:
                pdb.set_trace()
            sess.run(tf.assign(variable, value))
        """
        #------------------#
        #    Model Size    #
        #------------------#
        Pruned_Size = 0
        for iter, mask in enumerate(tf.get_collection('float32_weights_mask')):
            Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)
                
        print("\033[0;36m=======================\033[0m")
        print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size-Pruned_Size))
        print("\033[0;36m=======================\033[0m")
        
        #-------------------#
        #    Computation    #
        #-------------------#
        computation = compute_computation(data_format, sess)
        print("\033[0;36m=======================\033[0m")
        print("\033[0;36m Computation\033[0m = {}" .format(computation))
        print("\033[0;36m=======================\033[0m")
        
        #similar_group(inputs_and_kernels, sess)
        print("Testing Data Result ... ")
        test_result, test_accuracy, test_accuracy_top2, test_accuracy_top3, test_Y_pre = compute_accuracy(
            xs                      = xs, 
            ys                      = ys, 
            is_training             = is_training,
            is_quantized_activation = is_quantized_activation,
            Model_dict              = Model_dict,                
            QUANTIZED_NOW           = True, 
            prediction_list         = [prediction], 
            data_num                = test_data_num,
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
    tf.reset_default_graph()
    return test_accuracy
 
def Pruning(
    # Model
    Model_dict           ,
    # Dataset            
    Dataset	             ,
    Dataset_Path         ,
    Y_pre_Path           ,
    class_num            ,
    # Parameter	
    HP                   ,
    Global_Epoch         ,
    weights_bd_ratio     ,
    biases_bd_ratio      ,
    # Model Save         
    HP_csv               ,
    Model_first_name     ,
    Model_second_name    ,
    # Pre-trained Weights
    pruning_model_path   ,
    pruning_model        
    ):
    
    
    Model_Name = Model_first_name + '_' + Model_second_name
    
    #---------------#
    #    Dataset    #
    #---------------#
    print("Parsing Data ... ")
    ## Cifar10
    if Dataset == 'cifar10':
        filenames = [os.path.join(Dataset_Path, 'data_batch_%d.bin'%(i+1)) for i in range(5)]
    ## Imagenet ILSVRC_2012
    elif Dataset == 'ILSVRC2012':
        _NUM_TRAIN_FILES = 1024
        filenames = [os.path.join(Dataset_Path, 'train-%05d-of-01024' % i) for i in range(_NUM_TRAIN_FILES)]
    xs, ys = input_fn(filenames, class_num, True, HP, Dataset)
    train_data_num = HP['train_num'] 

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
    if HP['LR_Strategy'] == '3times':
        if   (Global_Epoch+1) <= HP['LR_Decade_1st_Epoch']:
            learning_rate = HP['LR'] / pow(HP['LR_Decade'], 0)
        elif (Global_Epoch+1) <= HP['LR_Decade_2nd_Epoch']:
            learning_rate = HP['LR'] / pow(HP['LR_Decade'], 1)
        elif (Global_Epoch+1) <= HP['LR_Decade_3rd_Epoch']:
            learning_rate = HP['LR'] / pow(HP['LR_Decade'], 2)
        else:
            learning_rate = HP['LR'] / pow(HP['LR_Decade'], 3)
    
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
    data_format = "NCHW"
    print("Building Model ...")
    ## -- Build Model --
    net = xs[0 : HP['Batch_Size']]
    Model_dict_ = copy.deepcopy(Model_dict)     
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict_, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = HP['Dropout_Rate'],
        data_format             = data_format,
        reuse                   = None)
    
    ## -- Model Size --
    # Your grade will depend on this value.
    #all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model")
    # Model Size
    Model_Size = 0
    for iter, variable in enumerate(tf.trainable_variables()):
        Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
        # See all your variables in termainl	
        #print("{}, {}" .format(iter, variable))

    ## -- Collection --
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
        
    ## -- Loss --
    labels = ys[0 : HP['Batch_Size']]
    
    # L2 Regularization
    l2_norm   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if 'batch_normalization' not in v.name])
    l2_lambda = tf.constant(HP['L2_Lambda'])
    l2_norm   = tf.multiply(l2_lambda, l2_norm)
    #l2_norm   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    #l2_lambda = tf.constant(HP['L2_Lambda'])
    #l2_norm   = tf.multiply(l2_lambda, l2_norm)
    
    # Cross Entropy
    cross_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels = labels,
        logits        = prediction)
    
    # Loss
    loss = cross_entropy + l2_norm
    
    ## -- Optimizer --
    if HP['Opt_Method']=='Adam':
        opt = tf.train.AdamOptimizer(learning_rate, HP['Momentum_Rate'])
    elif HP['Opt_Method']=='Momentum':
        opt = tf.train.MomentumOptimizer(
            learning_rate = learning_rate, 
            momentum      = HP['Momentum_Rate'])
        
    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        ## Compute Gradients
        var_list = tf.trainable_variables()
        gra_and_var = opt.compute_gradients(loss, var_list = var_list)
            
    ## Apply Gradients
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
    #config.gpu_options.allow_growth = True
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
        print("Loading the trained weights ... ")
        print("\033[0;35m{}\033[0m" .format(pruning_model_path + pruning_model))
        save_path = saver.restore(sess, pruning_model_path + pruning_model)
        
        #-------------#
        #   Pruning   #
        #-------------#
        if Global_Epoch%HP['Pruning_Retrain_Epoch'] == 0:
            print("pruning ...")
            pruned_weights_info = load_obj(pruning_model_path, "pruned_info")
            pruning_propotion = HP['Pruning_Propotion_Per']
            kernel_values_per_layer = similar_group(inputs_and_kernels, sess)
            if HP['Pruning_Strategy'] == 'Filter_Angle':
                if Model_first_name == 'DenseNet':
                    pruned_weights_info = denseNet_filter_prune_by_angle(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
                else:
                    pruned_weights_info = filter_prune_by_angle(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
            elif HP['Pruning_Strategy'] == 'Filter_AngleII':
                if Model_first_name == 'DenseNet':
                    pruned_weights_info = denseNet_filter_prune_by_angleII(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
                else:
                    filter_prune_by_angleII(prune_info_dict, pruning_propotion, sess)
            elif HP['Pruning_Strategy'] == 'Filter_Magnitude':
                if Model_second_name.split('_')[0] == '110':
                    pruning_propotion = [0.6, 0.3, 0.1]
                    pruning_layer = []
                    skip_layer = [] # conv 36, 38, 74
                if Model_second_name.split('_')[0] == '56':
                    pruning_propotion = [0.5, 0.4, 0.3]
                    pruning_layer = []
                    skip_layer = [] # conv 16, 18, 20, 34, 38, 54
                filter_prune_by_magnitude(prune_info_dict, pruning_propotion, sess)
            elif HP['Pruning_Strategy'] == 'Sparse_Magnitude':
                sparse_prune_by_magnitude(0.05, sess)
            elif HP['Pruning_Strategy'] == 'Plane_Angle':
                pruned_weights_info = plane_prune_by_angle(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
            elif HP['Pruning_Strategy'] == 'Filter_Angle_with_Skip':
                if Model_first_name == 'DenseNet':
                    if Model_second_name == '40_12':
                        skip_layer = [22, 28, 55, 79] # conv 16, 20, 38, 54
                    densenet_filter_prune_by_angle_with_skip(prune_info_dict, pruning_propotion, skip_layer, sess)
                else:
                    if Model_second_name.split('_')[0] == '110':
                        skip_layer = [52] # conv 36
                    if Model_second_name.split('_')[0] == '56':
                        skip_layer = [22, 28, 55, 79] # conv 16, 20, 38, 54
                    filter_prune_by_angle_with_skip(prune_info_dict, pruning_propotion, skip_layer, sess)
            elif HP['Pruning_Strategy'] == 'Filter_AngleII_with_Skip':
                if Model_first_name == 'DenseNet':
                    if Model_second_name == '40_12':
                        skip_layer = ['layer0', 'layer26', 'layer53', 'layer78']
                    denseNet_filter_prune_by_angleII_with_skip(prune_info_dict, pruning_propotion, pruned_weights_info, skip_layer, sess)
            elif HP['Pruning_Strategy'] == 'Filter_Angle_with_Penalty':
                max_penalty = 2
                pruned_weights_info = filter_prune_by_angle_with_penalty(
                    prune_info_dict,
                    pruning_propotion,
                    pruned_weights_info,
                    max_penalty,
                    sess)
            elif HP['Pruning_Strategy'] == 'Filter_AngleII_with_Skip_with_Penalty':
                if Model_first_name == 'DenseNet':
                    if Model_second_name == '40_12':
                        skip_layer = ['layer0', 'layer26', 'layer53', 'layer78']
                    denseNet_filter_prune_by_angleII_with_skip_with_penalty(
                        prune_info_dict, 
                        pruning_propotion, 
                        pruned_weights_info, 
                        skip_layer, 
                        sess)
            else:
                print("\033[1;31mError\033[0m : No such strategy!")
                exit()
        #path = 'Model/ResNet_Model/ResNet_56_cifar10_0_99_cifar10_2018.02.09_Filter_Angle10_88/'
        #save_dict(pruned_weights_info, path, "pruned_info")
        #aaa = load_obj(path, "pruned_info")
        
        #------------------#
        #    Model Size    #
        #------------------#
        Pruned_Size = 0
        for iter, mask in enumerate(tf.get_collection('float32_weights_mask')):
            Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)

        print("\033[0;36m=======================\033[0m")
        print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size-Pruned_Size))
        print("\033[0;36m=======================\033[0m")
        
        #-------------------#
        #    Computation    #
        #-------------------#
        computation = compute_computation(data_format, sess)
        print("\033[0;36m=======================\033[0m")
        print("\033[0;36m Computation\033[0m = {}" .format(computation))
        print("\033[0;36m=======================\033[0m")
        #np.savetxt(path + 'computation.csv', np.array([computation]), delimiter=",", fmt="%d")
        #with open(path + 'computation.csv') as csvfile:
        #    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #    for _, row in enumerate(reader):
        #        computation = int(row[0])
        #        print(int(row[0]))
        #exit()
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
            if IS_QUANTIZED_ACTIVATION and (epoch + 1 + Global_Epoch) == HP['Quantized_Activation_Epoch']:
                batch_xs = train_data[0:HP['Batch_Size']]
                # Calculate Each Activation's appropriate mantissa and fractional bit
                m, f = quantized_m_and_f(float32_net_collection, is_quantized_activation_collection, xs, Model_dict, batch_xs, sess)	
                # Assign mantissa and fractional bit to the tensor
                assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
                # Start Quantize Activation
                QUANTIZED_NOW = True
            ## -- Ternary --
            if IS_TERNARY and (epoch+1+Global_Epoch)==HP['Ternary_Epoch']:
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
            tStart_Batch = time.time()
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
                _, Loss, Prediction, L2_norm, batch_ys = sess.run(
                    [train_step, loss, prediction, l2_norm, ys], 
                    feed_dict = feed_dict_train)
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
                tEND_Batch = time.time()
                print("\033[1;34;40mEpoch\033[0m : %3d" %(epoch+Global_Epoch), end=" ")
                print("\033[1;34;40mData Iteration\033[0m : %7d" %(batch_iter*HP['Batch_Size']), end=" ")
                print("\033[1;32;40mBatch Accuracy\033[0m : %5f" %(batch_accuracy), end=" ")
                print("\033[1;32;40mLoss\033[0m : %5f" %(np.mean(Loss)), end=" ")
                print("\033[1;32;40mL2-norm\033[0m : %5f" %(L2_norm), end=" ")
                print("\033[1;32;40mLearning Rate\033[0m : %4f" %(lr), end=" ")
                print("(%2f sec)" %(tEND_Batch-tStart_Batch))
                tStart_Batch = time.time()
                # Per Class Accuracy
                """
                per_class_accuracy(Prediction, batch_ys)
                """
            
            tEnd = time.time()            
            Train_acc  = total_correct_num  / (total_correct_num + total_error_num)
            Train_loss = Train_loss / iteration 
            
            """
            print("\r\033[0;33mEpoch{}\033[0m" .format(epoch+Global_Epoch), end = "")
            print(" (Cost {TIME} sec)" .format(TIME = tEnd - tStart))
            print("\033[0;32mLearning Rate    \033[0m : {}".format(learning_rate))
            print("\033[0;32mTraining Accuracy\033[0m : {}".format(Train_acc))
            print("\033[0;32mTraining Loss    \033[0m : {} (l2_norm: {})".format(Train_loss, L2_norm))
            """
            
            # Record Per Epoch Training Result (Finally this will save as the .csv file)
            if epoch==0:
                Train_acc_per_epoch  = np.array([Train_acc ])
                Train_loss_per_epoch = np.array([Train_loss])
            else:
                Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc ])], axis=0)
                Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
            
            #---------------------------#
            #   Train Directory Build   #
            #---------------------------#
            if (epoch+1)==HP['Epoch']:
                Pruned_time = Global_Epoch / HP['Pruning_Retrain_Epoch']
                Dir = pruning_model_path[0:len(pruning_model_path)-1]
                Pruned_Size = 0.
                for iter, mask in enumerate(tf.get_collection('float32_weights_mask')):
                    Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)
                Pruning_Propotion_Now = int(Pruned_Size / Model_Size * 100) #int(HP['Pruning_Propotion_Per']*100*(Pruned_time+1))
                Dir = Dir.split('_' + HP['Pruning_Strategy'])[0] + '_' + HP['Pruning_Strategy'] + str(int(HP['Pruning_Propotion_Per']*100)) + '_' + str(Pruning_Propotion_Now) + '/'
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
            if (epoch+1)==HP['Epoch']:
                print("Saving Trained Weights ...")
                save_path = saver.save(sess, Dir + str(Global_Epoch+epoch+1) + ".ckpt")
                print("\033[0;35m{}\033[0m" .format(save_path))
            
            #------------------------#
            #   Saving Computation   #
            #------------------------#
            if (epoch+1)==HP['Epoch']:
                np.savetxt(Dir + 'computation.csv', np.array([computation]), delimiter=",", fmt="%d")
            
            #------------------------#
            #   Saving Pruned info   #
            #------------------------#
            if (epoch+1)==HP['Epoch']:
                save_dict(pruned_weights_info, Dir, "pruned_info")
                
        tEnd_All = time.time()
        print("Total costs {TIME} sec\n" .format(TIME = tEnd_All - tStart_All))

        #-----------------------------------#
        #   Saving Train info as csv file   #
        #-----------------------------------#
        Train_acc_per_epoch  = np.expand_dims(Train_acc_per_epoch , axis=1)
        Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
        Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
        Save_file_as_csv(Dir+'train_info' , Train_info)
        print("see the batch norm info!")
    tf.reset_default_graph()
        
    Model_Path = Dir
    Model = str(HP['Epoch']+Global_Epoch) + '.ckpt'
    
    return Model_Path, Model

def Rebuilding(
    # Model
    Model_dict                 ,
    # Dataset                  
    Dataset	                   ,
    Dataset_Path               ,
    Y_pre_Path                 ,
    class_num                  ,
    # Parameter	               
    HP                         ,
    Global_Epoch               ,
    weights_bd_ratio           ,
    biases_bd_ratio            ,
    # Model Save               
    HP_csv                     ,
    Model_first_name           ,
    Model_second_name          ,
    # Pre-trained Weights
    rebuilding_model_path_base ,
    rebuilding_model_path      ,
    rebuilding_model           , 
    base_mask                  ,
    base_weights               ,
    index_now
    ):
    
    
    Model_Name = Model_first_name + '_' + Model_second_name
    
    #---------------#
    #    Dataset    #
    #---------------#
    print("Parsing Data ... ")
    filenames = [os.path.join(Dataset_Path, 'data_batch_%d.bin'%(i+1)) for i in range(5)]
    xs, ys = input_fn(filenames, class_num, True, HP)
    train_data_index   = np.array(open(Dataset_Path + '/train.txt', 'r').read().splitlines())
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
    if HP['LR_Strategy'] == '3times':
        if   (Global_Epoch+1) <= HP['LR_Decade_1st_Epoch']:
            learning_rate = HP['LR'] / pow(HP['LR_Decade'], 0)
        elif (Global_Epoch+1) <= HP['LR_Decade_2nd_Epoch']:
            learning_rate = HP['LR'] / pow(HP['LR_Decade'], 1)
        elif (Global_Epoch+1) <= HP['LR_Decade_3rd_Epoch']:
            learning_rate = HP['LR'] / pow(HP['LR_Decade'], 2)
        else:
            learning_rate = HP['LR'] / pow(HP['LR_Decade'], 3)
    
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
    data_format = "NCHW"
    ## -- Build Model --
    net = xs[0 : HP['Batch_Size']]
    Model_dict_ = copy.deepcopy(Model_dict)     
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict_, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = HP['Dropout_Rate'],
        data_format             = data_format,
        reuse                   = None)
    
    ## -- Model Size --
    # Your grade will depend on this value.
    all_variables = tf.trainable_variables() #tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model")
    # Model Size
    Model_Size = 0
    for iter, variable in enumerate(all_variables):
        Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
        # See all your variables in termainl	
        #print("{}, {}" .format(iter, variable))

    ## -- Collection --
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
        
    ## -- Loss --
    labels = ys[0 : HP['Batch_Size']]
    
    # L2 Regularization
    l2_norm   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    l2_lambda = tf.constant(HP['L2_Lambda'])
    l2_norm   = tf.multiply(l2_lambda, l2_norm)
    
    # Cross Entropy
    cross_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels = labels,
        logits        = prediction)
    
    # Loss
    loss = cross_entropy + HP['L2_Lambda'] * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    
    ## -- Optimizer --
    if HP['Opt_Method']=='Adam':
        opt = tf.train.AdamOptimizer(learning_rate, HP['Momentum_Rate'])
    elif HP['Opt_Method']=='Momentum':
        opt = tf.train.MomentumOptimizer(
            learning_rate = learning_rate, 
            momentum      = HP['Momentum_Rate'])
        
    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        ## Compute Gradients
        var_list = tf.trainable_variables()
        gra_and_var = opt.compute_gradients(loss, var_list = var_list)
            
    ## Apply Gradients
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
    #config.gpu_options.allow_growth = True
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
        print("Loading the trained weights ... ")
        print("\033[0;35m{}\033[0m" .format(rebuilding_model_path + rebuilding_model))
        save_path = saver.restore(sess, rebuilding_model_path + rebuilding_model)
        
        #--------------------------------------------------#
        #    Assign untrainable weights and masks value    #
        #--------------------------------------------------#
        is_train_mask_collection = tf.get_collection('is_train_float32_weights_mask', scope = None)
        constant_weights_collection = tf.get_collection('constant_float32_weights', scope = None)
        if Global_Epoch == 0:
            # mask
            for iter in range(len(is_train_mask_collection)):
                is_train_mask = is_train_mask_collection[iter]
                mask_value = 1. - base_mask[iter]
                sess.run(tf.assign(is_train_mask, mask_value))
                
            # weights
            for iter in range(len(constant_weights_collection)):
                constant_weights = constant_weights_collection[iter]
                constant_weights_value = base_mask[iter] * base_weights[iter]
                sess.run(tf.assign(constant_weights, constant_weights_value))
        
        #------------------------------#
        #    Set the trainable mask    #
        #------------------------------#
        pruned_weights_info = load_obj(rebuilding_model_path_base, "pruned_info")
        if Global_Epoch == 0:
            for index in range(index_now[1], index_now[0] + 1)[::-1]:
                #print(index)
                keys = pruned_weights_info[index].keys()
                for _, key in enumerate(keys):
                    if key != 'computation':
                        mask_tensor = tf.get_collection("float32_weights_mask", scope="Model/" + key + "/")[0]
                        mask = sess.run(mask_tensor)
                        if pruned_weights_info[index][key].keys()[0] == 'depth':
                            assert np.sum(mask[:, :, pruned_weights_info[index][key]['depth'], :]) == 0, "Something Error"
                            for channel in range(np.shape(mask)[3]):
                                if np.sum(mask[:, :, :, channel]) != 0:
                                    mask[:, :, pruned_weights_info[index][key]['depth'], channel] = 1
                            sess.run(tf.assign(mask_tensor, mask))
                        elif pruned_weights_info[index][key].keys()[0] == 'channel':
                            assert np.sum(mask[:, :, :, pruned_weights_info[index][key]['channel']]) == 0, "Something Error"
                            for depth in range(np.shape(mask)[2]):
                                if np.sum(mask[:, :, depth, :]) != 0:
                                    mask[:, :, depth, pruned_weights_info[index][key]['channel']] = 1
                            sess.run(tf.assign(mask_tensor, mask))

        #------------------#
        #    Model Size    #
        #------------------#
        Pruned_Size = 0
        for iter, mask in enumerate(tf.get_collection('float32_weights_mask')):
            Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)

        print("\033[0;36m=======================\033[0m")
        print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size-Pruned_Size))
        print("\033[0;36m=======================\033[0m")
        
        #-------------------#
        #    Computation    #
        #-------------------#
        computation = compute_computation(data_format, sess)
        print("\033[0;36m=======================\033[0m")
        print("\033[0;36m Computation\033[0m = {}" .format(computation))
        print("\033[0;36m=======================\033[0m")
        
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
            ## -- Quantizad Activation --
            if IS_QUANTIZED_ACTIVATION and (epoch + 1 + Global_Epoch) == HP['Quantized_Activation_Epoch']:
                batch_xs = train_data[0:HP['Batch_Size']]
                # Calculate Each Activation's appropriate mantissa and fractional bit
                m, f = quantized_m_and_f(float32_net_collection, is_quantized_activation_collection, xs, Model_dict, batch_xs, sess)	
                # Assign mantissa and fractional bit to the tensor
                assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
                # Start Quantize Activation
                QUANTIZED_NOW = True
            ## -- Ternary --
            if IS_TERNARY and (epoch+1+Global_Epoch)==HP['Ternary_Epoch']:
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
                ## Run Training Step ##
                feed_dict_train.update({is_training: True}) # xs: batch_xs, ys: batch_ys, learning_rate: lr, 
                _, Loss, Prediction, L2_norm, batch_ys, gradient = sess.run(
                    [train_step, loss, prediction, l2_norm, ys, gra_and_var[0][0]], 
                    feed_dict = feed_dict_train)
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
                print("\033[1;34;40mEpoch\033[0m : {}" .format(epoch+Global_Epoch))
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
            print("\r\033[0;33mEpoch{}\033[0m" .format(epoch+Global_Epoch), end = "")
            print(" (Cost {TIME} sec)" .format(TIME = tEnd - tStart))
            print("\033[0;32mLearning Rate    \033[0m : {}".format(learning_rate))
            print("\033[0;32mTraining Accuracy\033[0m : {}".format(Train_acc))
            print("\033[0;32mTraining Loss    \033[0m : {} (l2_norm: {})".format(Train_loss, L2_norm))

            # Record Per Epoch Training Result (Finally this will save as the .csv file)
            if epoch==0:
                Train_acc_per_epoch  = np.array([Train_acc ])
                Train_loss_per_epoch = np.array([Train_loss])
            else:
                Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc ])], axis=0)
                Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
            
            #---------------------------#
            #   Train Directory Build   #
            #---------------------------#
            if (epoch+1)==HP['Epoch']:
                Dir = rebuilding_model_path_base[0:len(rebuilding_model_path_base)-1]
                Pruned_Size = 0.
                for iter, mask in enumerate(tf.get_collection('float32_weights_mask')):
                    Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)
                Pruning_Propotion_Now = int(Pruned_Size / Model_Size * 100)
                
                if len(Dir.split('Rebuild')) == 1:
                    Dir = Dir + '_' + 'Rebuild' + str(Pruning_Propotion_Now) + '/'
                else:
                    Dir = Dir + '_' + str(Pruning_Propotion_Now) + '/'
                
                if (not os.path.exists(Dir)):
                    print("\033[0;35m%s\033[0m is not exist!" %Dir)
                    print("\033[0;35m%s\033[0m is created!" %Dir)
                    os.makedirs(Dir)
                #------------------------------------------------------#
                #   Update the constant weights to trainable weights   #
                #------------------------------------------------------#
                for iter in range(len(is_train_mask_collection)):
                    constant_weights_value = base_mask[iter] * base_weights[iter]
                    float32_weights = float32_weights_collection[iter]
                    float32_weights_value = (1. - base_mask[iter]) * sess.run(float32_weights) + constant_weights_value
                    sess.run(tf.assign(float32_weights, float32_weights_value))
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
            if (epoch+1)==HP['Epoch']:
                print("Saving Trained Weights ...")
                save_path = saver.save(sess, Dir + str(Global_Epoch+epoch+1) + ".ckpt")
                print("\033[0;35m{}\033[0m" .format(save_path))
            #------------------------#
            #   Saving Computation   #
            #------------------------#
            if (epoch+1)==HP['Epoch']:
                np.savetxt(Dir + 'computation.csv', np.array([computation]), delimiter=",", fmt="%d")
            #--------------------------#
            #    Saving pruned_info    #
            #--------------------------#
            if (epoch+1)==HP['Epoch']:
                print("Saving pruned_info ...")
                save_dict(pruned_weights_info[0:index_now[-1]], Dir, "pruned_info")
        tEnd_All = time.time()
        print("Total costs {TIME} sec\n" .format(TIME = tEnd_All - tStart_All))

        #-----------------------------------#
        #   Saving Train info as csv file   #
        #-----------------------------------#
        Train_acc_per_epoch  = np.expand_dims(Train_acc_per_epoch , axis=1)
        Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
        Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
        Save_file_as_csv(Dir+'train_info' , Train_info)
        
        
        # Debug
        """
        constant = tf.get_collection('constant_float32_weights', scope = None)
        is_train_mask = tf.get_collection('is_train_float32_weights_mask', scope = None)
        weights = tf.get_collection('float32_weights', scope = None)
        aa = sess.run(constant[0])
        bb = sess.run(weights[0])
        cc = sess.run(is_train_mask[0])
        pdb.set_trace()
        print("...")
        """
        # End Debug
        
    tf.reset_default_graph()
        
    Model_Path = Dir
    Model = str(HP['Epoch']+Global_Epoch) + '.ckpt'
    
    return Model_Path, Model

#=============#
#    Model    #
#=============#
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
    data_format,
    reuse = None
    ):
    
    if data_format == "NCHW":
        net = tf.transpose(net, [0, 3, 1, 2])
    
    Analysis = {}
    inputs_and_kernels = {} # For finding the similar weights
    prune_info_dict = {}
    # For Pruning
    is_shortcut_past_layer = False
    children_tmp = {}
    past_conv_layer = None
    # --
    Analyzer(Analysis, net, type='DATA', name='Input')
    past_layer = 0
    max_parameter = 0
    with tf.variable_scope("Model", reuse = reuse):
        for layer in range(len(Model_dict)):
            layer_now = Model_dict['layer'+str(layer)]
            #-------------------#
            #    Shortcut Add   #
            #-------------------#
            if bool(layer_now.get('shortcut_num')):
                is_shortcut_past_layer = True # For Pruning
                for shortcut_index in range(layer_now['shortcut_num']):
                    with tf.variable_scope(layer_now['shortcut_input_layer'][str(shortcut_index)] + "_to_" + "layer%d"%layer):
                        shortcut_input_layer = Model_dict[layer_now['shortcut_input_layer'][str(shortcut_index)]]
                        shortcut = layer_now['shortcut_input'][str(shortcut_index)]
                        shortcut_layer = layer_now['shortcut_input_layer'][str(shortcut_index)]
                        
                        if shortcut_input_layer['shortcut_connection'] == "ADD":
                            ## Show the model
                            if SHOW_MODEL:
                                print("\033[0;35mShortcut from {}\033[0m" .format(shortcut_layer))
    
                            shortcut = Model.shortcut_Module(
                                net                     = shortcut,
                                group                   = int(shortcut_input_layer['group']),
                                destination             = net,
                                initializer             = tf.contrib.layers.variance_scaling_initializer(),
                                is_training             = is_training,
                                is_add_biases           = shortcut_input_layer['is_add_biases']           == 'TRUE',
                                is_projection_shortcut  = shortcut_input_layer['is_projection_shortcut']  == 'TRUE',
                                shortcut_type           = shortcut_input_layer['shortcut_type']                    ,
                                shortcut_connection     = shortcut_input_layer['shortcut_connection']              ,
                                is_batch_norm           = shortcut_input_layer['is_batch_norm']           == 'TRUE',
                                is_ternary              = is_ternary[shortcut_layer]                               ,
                                is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                                IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE', 	
                                IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                padding                 = "SAME",			 
                                data_format             = data_format,
                                Analysis                = Analysis)
                    
                    # Add the shortcut and net
                    with tf.variable_scope('layer%d' %(layer)):
                        if shortcut_input_layer['shortcut_connection'] == "ADD":
                            ## Show the model
                            if SHOW_MODEL:
                                print("-> ADD")
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
                
            
            #-------------------#
            #    Destination    # 
            #-------------------#
            if layer_now['is_shortcut'] == 'TRUE':
                #pdb.set_trace()
                for shortcut_destination in layer_now['shortcut_destination'].split('/'):
                    if bool(Model_dict[shortcut_destination].get('shortcut_num')):
                        shortcut_num = Model_dict[shortcut_destination]['shortcut_num'] + 1
                        Model_dict[shortcut_destination].update({'shortcut_num':shortcut_num})
                        Model_dict[shortcut_destination]['shortcut_input'].update({str(shortcut_num-1):net})
                        Model_dict[shortcut_destination]['shortcut_input_layer'].update({str(shortcut_num-1):'layer%d' %(layer)})
                    else:
                        shortcut_num = 1
                        Model_dict[shortcut_destination].update({'shortcut_num':shortcut_num})
                        Model_dict[shortcut_destination].update({'shortcut_input':{str(shortcut_num-1):net}})
                        Model_dict[shortcut_destination].update({'shortcut_input_layer':{str(shortcut_num-1):'layer%d' %(layer)}})
            
            ## Show the model
            if SHOW_MODEL:
                print("\033[0;33mlayer{}\033[0m -> {}, scope = {}" .format(layer, net.shape, layer_now['scope']))
            
            #-----------------------#
            #    Shortcut Concat    #
            #-----------------------#
            if bool(layer_now.get('shortcut_num')):
                is_shortcut_past_layer = True # For Pruning
                for shortcut_index in range(layer_now['shortcut_num']):
                    with tf.variable_scope(layer_now['shortcut_input_layer'][str(shortcut_index)] + "_to_" + "layer%d"%layer):
                        shortcut_input_layer = Model_dict[layer_now['shortcut_input_layer'][str(shortcut_index)]]
                        shortcut = layer_now['shortcut_input'][str(shortcut_index)]
                        shortcut_layer = layer_now['shortcut_input_layer'][str(shortcut_index)]
                        if shortcut_input_layer['shortcut_connection'] == "CONCAT":
                            ## Show the model
                            if SHOW_MODEL:
                                print("\033[0;35mShortcut from {}\033[0m" .format(shortcut_layer))

                            shortcut = Model.shortcut_Module(
                                net                     = shortcut,
                                group                   = int(shortcut_input_layer['group']),
                                destination             = net,
                                initializer             = tf.contrib.layers.variance_scaling_initializer(),
                                is_training             = is_training,
                                is_add_biases           = shortcut_input_layer['is_add_biases']           == 'TRUE',
                                is_projection_shortcut  = shortcut_input_layer['is_projection_shortcut']  == 'TRUE',
                                shortcut_type           = shortcut_input_layer['shortcut_type']                    ,
                                shortcut_connection     = shortcut_input_layer['shortcut_connection']              ,
                                is_batch_norm           = shortcut_input_layer['is_batch_norm']           == 'TRUE',
                                is_ternary              = is_ternary[shortcut_layer]                               ,
                                is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                                IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE', 	
                                IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                padding                 = "SAME",			 
                                data_format             = data_format,
                                Analysis                = Analysis)            
                    
                    # Concatenate the shortcut and net
                    with tf.variable_scope('layer%d' %(layer)):
                        if shortcut_input_layer['shortcut_connection'] == "CONCAT":
                            ## Show the model
                            if SHOW_MODEL:
                                print("-> CONCAT")
                            # Get Destination
                            destination = 'layer' + str(layer)
                            while(1):
                                if Model_dict[destination]['type'] != 'CONV':
                                    destination = 'layer' + str(int(destination.split('layer')[-1]) + 1)
                                    if destination == 'layer' + str(len(Model_dict)-1):
                                        break
                                else:
                                    break
                            # Update Parent & children
                            shortcut_conv_layer = shortcut_layer
                            while(1):
                                if Model_dict[shortcut_conv_layer]['type'] != 'CONV':
                                    shortcut_conv_layer = 'layer' + str(int(shortcut_conv_layer.split('layer')[-1]) - 1)
                                else:
                                    # Children
                                    if destination != 'layer' + str(len(Model_dict)-1): # Not the last layer
                                        if data_format == "NHWC":
                                            prune_info_dict[shortcut_conv_layer]['children'].update({destination: net.get_shape().as_list()[3]})
                                        elif data_format == "NCHW":
                                            prune_info_dict[shortcut_conv_layer]['children'].update({destination: net.get_shape().as_list()[1]})
                                    # Parent
                                        if not bool(prune_info_dict.get(destination)):
                                            if data_format == "NHWC":
                                                prune_info_dict.update({destination: {'parents': {shortcut_conv_layer: net.get_shape().as_list()[3]}}})
                                            elif data_format == "NCHW":
                                                prune_info_dict.update({destination: {'parents': {shortcut_conv_layer: net.get_shape().as_list()[1]}}})
                                        else:
                                            if data_format == "NHWC":
                                                prune_info_dict[destination]['parents'].update({shortcut_conv_layer: net.get_shape().as_list()[3]})
                                            elif data_format == "NCHW":
                                                prune_info_dict[destination]['parents'].update({shortcut_conv_layer: net.get_shape().as_list()[1]})
                                    break
                            
                            # Concatenate
                            if data_format == "NHWC":
                                net = tf.concat([net, shortcut], axis = 3)
                            elif data_format == "NCHW":
                                net = tf.concat([net, shortcut], axis = 1)
                            #print(".............{}".format(prune_info_dict[shortcut_conv_layer]['children']))
                            
            # Other Operation
            with tf.variable_scope('layer%d' %(layer)):                
                #-------------------#
                #    Max Pooling    #
                #-------------------#
                if layer_now['type'] == 'MAX_POOL':         
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Max Pooling, Ksize={}, Stride={}".format(int(layer_now['kernel_size']), int(layer_now['stride'])))
                    
                    if data_format == "NCHW":
                        net = tf.transpose(net, [0, 2, 3, 1])
                    net, indices, output_shape = Model.indice_pool( 
                        net, 
                        kernel_size             = int(layer_now['kernel_size']), 
                        stride                  = int(layer_now['stride']), 
                        IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                        Analysis                = Analysis,
                        scope                   = layer_now['scope'])
                    if data_format == "NCHW":
                        net = tf.transpose(net, [0, 3, 1, 2])    
                    if layer_now['indice'] != 'None' and layer_now['indice'] != 'FALSE':
                        Model_dict[layer_now['indice']].update({'indice':indices, 'output_shape':output_shape})
                    """
                    if data_format == "NCHW":
                        data_format_ = 'channels_first'
                    elif data_format == "NHWC":
                        data_format_ = 'channels_last'
                        
                    net = tf.layers.max_pooling2d(
                            inputs      = net, 
                            pool_size   = int(layer_now['kernel_size']), 
                            strides     = int(layer_now['stride']), 
                            padding     = 'SAME',
                            data_format = data_format_)
                    """
                #---------------------#
                #    Max Unpooling    #
                #---------------------#
                if layer_now['type'] == 'MAX_UNPOOL':
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Max Unpooling")
                    if data_format == "NCHW":
                        net = tf.transpose(net, [0, 2, 3, 1])
                    net = Model.indice_unpool( 
                        net,  
                        output_shape = layer_now['output_shape'], 
                        indices      = layer_now['indice'], 
                        scope        = layer_now['scope'])
                    if data_format == "NCHW":
                        net = tf.transpose(net, [0, 3, 1, 2])
                #-------------------#
                #    Avg Pooling    #
                #-------------------#
                if layer_now['type'] == 'AVG_POOL':
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Avg Pooling")

                    if data_format == "NCHW":
                        data_format_ = "channels_first"
                    elif data_format == "NHWC":
                        data_format_ = "channels_last"
                        
                    net = tf.layers.average_pooling2d(
                        inputs      = net, 
                        pool_size   = int(layer_now['kernel_size']), 
                        strides     = int(layer_now['stride']), 
                        padding     = 'VALID',
                        data_format = data_format_)
                #---------------#
                #    Shuffle    #
                #---------------#
                if layer_now['type'] == "SHUFFLE":
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Shuffle")
                    net = Model.shuffle_net(
                        net         = net, 
                        group       = int(layer_now['group']),
                        data_format = data_format) 
                #-------------------#
                #    Combine Add    #
                #-------------------#
                if layer_now['type'] == "COMBINE_ADD":
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Combine Add")
                    net = Model.combine_add(
                        net            = net,
                        is_training    = is_training,
                        group          = int(layer_now['group']),
                        combine_number = int(layer_now['kernel_size']),
                        is_batch_norm  = layer_now['is_batch_norm'] == 'TRUE',
                        Activation     = layer_now['Activation'],
                        initializer    = tf.variance_scaling_initializer(), 
                        data_format    = data_format,
                        scope          = layer_now['scope'])
                #--------------------#
                #    Combine Conv    #
                #--------------------#
                if layer_now['type'] == "COMBINE_CONCAT":
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Combine Conv")
                    net = Model.combine_conv(
                        net            = net,
                        is_training    = is_training,
                        group          = int(layer_now['group']),
                        combine_number = int(layer_now['kernel_size']),
                        is_batch_norm  = layer_now['is_batch_norm'] == 'TRUE',
                        Activation     = layer_now['Activation'],
                        initializer    = tf.variance_scaling_initializer(), 
                        data_format    = data_format,
                        scope          = layer_now['scope'])
                #---------------------------#
                #    Batch Normalization    #
                #---------------------------#
                if layer_now['type'] == 'BN':
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Batch_Norm")
                    net = Model.batch_norm(
                        net         = net, 
                        is_training = is_training, 
                        data_format = data_format)
                    # Activation
                    if layer_now['Activation'] == 'ReLU':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> ReLU")
                        net = tf.nn.relu(net)
                    elif layer_now['Activation'] == 'Sigmoid':
                        net = tf.nn.sigmoid(net) 
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Sigmoid")
                #-----------------------#
                #    Fully Connected    #
                #-----------------------#
                if layer_now['type'] == 'FC':
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Fully-Connected")
                    # Dropout
                    if layer==(len(Model_dict)-1):
                        net = tf.cond(is_training, lambda: tf.layers.dropout(net, DROPOUT_RATE), lambda: net)
                    # FC
                    
                    net = Model.FC(
                        net, 
                        output_channel          = int(layer_now['output_channel']),
                        initializer             = tf.variance_scaling_initializer(),
                        is_training             = is_training,
                        is_add_biases           = layer_now['is_add_biases']           == 'TRUE',
                        is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',
                        is_dilated              = layer_now['is_dilated']              == 'TRUE',
                        is_ternary              = is_ternary['layer'+str(layer)]                ,
                        is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                        IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                        IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                        Activation              = layer_now['Activation'],
                        data_format             = data_format,
                        Analysis                = Analysis,
                        scope                   = layer_now['scope'])
                    
                    if layer==(len(Model_dict)-1):
                        net = tf.reshape(net, [-1, reduce(lambda x, y: x*y, net.get_shape().as_list()[1:4])])
                    """
                    net = tf.reshape(net, [-1, 2048])
                    net = tf.layers.dense(
                        inputs = net, 
                        units  = int(layer_now['output_channel']))
                    """
                #-------------------#
                #    Convolution    #
                #-------------------#
                if layer_now['type'] == 'CONV':
                    # Dropout
                    if layer==(len(Model_dict)-1):
                        net = tf.cond(is_training, lambda: tf.layers.dropout(net, DROPOUT_RATE), lambda: net)

                    # For finding the similar weights
                    inputs_and_kernels.update({'layer%d' %(layer): {'inputs' : net}})
                    
                    # Convolution
                    net = Model.conv2D( 
                        net, 
                        kernel_size             = int(layer_now['kernel_size']     ), 
                        stride                  = int(layer_now['stride']          ),
                        internal_channel        = int(layer_now['internal_channel']),
                        output_channel          = int(layer_now['output_channel']  ),
                        rate                    = int(layer_now['rate']            ),
                        group                   = int(layer_now['group']           ),
                        initializer             = tf.variance_scaling_initializer(), 
                        is_training             = is_training,
                        is_add_biases           = layer_now['is_add_biases']           == 'TRUE', 
                        is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',      
                        is_dilated              = layer_now['is_dilated']              == 'TRUE',      
                        is_depthwise            = layer_now['is_depthwise']            == 'TRUE',  
                        is_ternary              = is_ternary['layer'+str(layer)]                ,
                        is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                        IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',     
                        IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',						              
                        Activation              = layer_now['Activation'],
                        padding                 = "SAME",
                        data_format             = data_format,
                        Analysis                = Analysis,
                        scope                   = layer_now['scope'])
                    
                    if layer==(len(Model_dict)-1) and data_format == "NCHW":
                        net = tf.transpose(net, [0, 2, 3, 1])
                    
                    # For Pruning
                    if not bool(prune_info_dict.get('layer%d'%layer)):
                        prune_info_dict.update(
                            {'layer%d'%layer: {'weights': tf.get_collection("float32_weights", scope="Model/layer%d/"%layer)[0]}}
                        )
                    else:
                        prune_info_dict['layer%d'%layer].update(
                            {'weights': tf.get_collection("float32_weights", scope="Model/layer%d/"%layer)[0],
                             
                            }
                        )

                    prune_info_dict['layer%d'%layer].update({
                        'mask'       : tf.get_collection("float32_weights_mask", scope="Model/layer%d/"%layer)[0],
                        'outputs'    : tf.get_collection("conv_outputs"  , scope="Model/layer%d/"%layer)[0],
                        'stride'     : int(layer_now['stride']),
                        'is_shortcut': is_shortcut_past_layer
                    })
                    if past_conv_layer != None:
                        # Children
                        if not bool(prune_info_dict[past_conv_layer].get('children')):
                            prune_info_dict[past_conv_layer].update({'children': {'layer%d'%layer: 0}})
                        else:
                            prune_info_dict[past_conv_layer]['children'].update({'layer%d'%layer: 0})
                        
                        # Parents
                        if not bool(prune_info_dict['layer%d'%layer].get('parents')):
                            prune_info_dict['layer%d'%layer].update({'parents': {past_conv_layer: 0}})
                        else:
                            prune_info_dict['layer%d'%layer]['parents'].update({past_conv_layer: 0})
                            
                    past_conv_layer = 'layer%d'%layer
                    is_shortcut_past_layer = False
                """
                #----------------------------#
                #    children For Pruning    #
                #----------------------------#
                if layer_now['shortcut_destination'] != 'FALSE' and layer_now['shortcut_connection'] == 'True':
                    for destination in layer_now['shortcut_destination'].split('/'):
                        while(1):
                            if Model_dict[destination]['type'] != 'CONV':
                                destination = 'layer' + str(int(destination.split('layer')[-1]) + 1)
                                if destination == 'layer' + str(len(Model_dict)-1):
                                    break
                            else:
                                children_tmp.update({destination: 0})
                                break
                    if not bool(prune_info_dict[past_conv_layer].get('children')):
                        prune_info_dict[past_conv_layer].update({'children': children_tmp})
                    else:
                        prune_info_dict[past_conv_layer]['children'].update(children_tmp)
                    children_tmp = {}
                """
        # For finding the similar weights
        for layer in range(len(Model_dict)):
            #print(layer)
            try:
                inputs_and_kernels['layer%d' %(layer)].update({
                    #'kernels': tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model/layer%d" %(layer))})
                    'kernels': tf.get_collection("float32_weights", scope="Model/layer%d/" %(layer))})
                #print(tf.get_collection("float32_weights", scope="Model/layer%d/" %(layer)))
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
        
        if SHOW_MODEL:
            exit()
    return net, Analysis, max_parameter, inputs_and_kernels, prune_info_dict

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
                      'shortcut_type'          ,
                      'shortcut_connection'    ,
                      'is_batch_norm'          ,
                      'is_dilated'             ,
                      'is_depthwise'           ,
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
    HP,
    Dataset
    ):
    # record dataset
    if Dataset == "ILSVRC2012":
        batch_size = HP['Batch_Size']
        num_epochs = HP['Epoch']
        num_parallel_calls = 1
        multi_gpu = False
        
        _NUM_TRAIN_FILES = 1024
        _SHUFFLE_BUFFER = 1500
        
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if is_training:
            # Shuffle the input files
            dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
        
        num_images = is_training and HP['train_num'] or HP['test_num']
        
        # Convert to individual records
        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        
        # Parse the raw records into images and labels
        dataset = dataset.map(lambda value: parse_record_ILSVRC2012(value, is_training),
                                num_parallel_calls=num_parallel_calls)
                                
        ## process_record_dataset
        # We prefetch a batch at a time, This can help smooth out the time taken to
        # load input files as we go through shuffling and processing.
        dataset = dataset.prefetch(buffer_size=batch_size)
        if is_training:
            # Shuffle the records. Note that we shuffle before repeating to ensure
            # that the shuffling respects epoch boundaries.
            dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)
        
        # If we are training over multiple epochs before evaluating, repeat the
        # dataset for the appropriate number of epochs.
        dataset = dataset.repeat(num_epochs)
        
        # Currently, if we are using multiple GPUs, we can't pass in uneven batches.
        # (For example, if we have 4 GPUs, the number of examples in each batch
        # must be divisible by 4.) We already ensured this for the batch_size, but
        # we have to additionally ensure that any "leftover" examples-- the remainder
        # examples (total examples % batch_size) that get called a batch for the very
        # last batch of an epoch-- do not raise an error when we try to split them
        # over the GPUs. This will likely be handled by Estimator during replication
        # in the future, but for now, we just drop the leftovers here.
        if multi_gpu:
            total_examples = num_epochs * examples_per_epoch
            dataset = dataset.take(batch_size * (total_examples // batch_size))

        dataset = dataset.batch(batch_size)
        
        # Operations between the final prefetch and the get_next call to the iterator
        # will happen synchronously during run time. We prefetch here again to
        # background all of the above processing work and keep it out of the
        # critical training path.
        #dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        
        return images, labels

    else:
        record_bytes = HP['H_Resize'] * HP['W_Resize'] * 3 + 1
        dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance. Because CIFAR-10
        # is a relatively small dataset, we choose to shuffle the full epoch.
        if is_training:
            dataset = dataset.shuffle(buffer_size = HP['train_num'])
        
        if Dataset == "cifar10":
            dataset = dataset.map(lambda raw_record: parse_record_cifar10(raw_record, HP['H_Resize'], HP['W_Resize'], class_num))
            dataset = dataset.map(lambda image, label: (preprocess_image_cifar10(image, is_training, HP['H_Resize'], HP['W_Resize']), label))
            
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

def preprocess_image_cifar10(
    image, 
    is_training,
    H,
    W
    ):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, H + 8, W + 8)
        # Randomly crop a [_HEIGHT, _WIDTH] section of the image. 
        image = tf.random_crop(image, [H, W, 3])
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image
 
def preprocess_image_ILSVRC2012(
    image, 
    is_training,
    H,
    W
    ):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, H + 8, W + 8)
        # Randomly crop a [_HEIGHT, _WIDTH] section of the image. 
        image = tf.random_crop(image, [H, W, 3])
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image
  
def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox
  
def parse_record_cifar10(
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

def parse_record_ILSVRC2012(
    value, 
    is_training
    ):
    """Parses a record containing a training example of an image.
    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).
    Args:
        raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
        is_training: A boolean denoting whether the input is for training.
    Returns:
        Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    
    _DEFAULT_IMAGE_SIZE = 224
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 1001
    """
    image_buffer, label, bbox = _parse_example_proto(raw_record)
    
    image = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        num_channels=_NUM_CHANNELS,
        is_training=is_training)
    """
    
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }
    
    parsed = tf.parse_single_example(value, keys_to_features)
    
    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]),
        _NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        is_training=is_training)
    
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]),
        dtype=tf.int32)
    
    label = tf.one_hot(tf.reshape(label, shape=[]), _NUM_CLASSES)
    
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

def save_dict(
    obj, 
    path, 
    name
    ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(
    path, 
    name
    ):
    if os.path.isfile(path + name + '.pkl'):
        with open(path + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return []
        
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

def compute_computation(
    data_format, 
    sess
    ):
    
    mask_collection = tf.get_collection("float32_weights_mask", scope = None)
    outputs_collection = tf.get_collection("conv_outputs", scope = None)
    assert len(mask_collection) == len(outputs_collection), 'Number of mask is not equal to the Number of outputs.'
    total_computation = 0
    for iter in range(len(mask_collection)):
        mask = sess.run(mask_collection[iter])
        if data_format == "NCHW" or data_format == "channels_first":
            H = outputs_collection[iter].get_shape().as_list()[2]
            W = outputs_collection[iter].get_shape().as_list()[3]
        elif data_format == "NHWC" or data_format == "channels_last":
            H = outputs_collection[iter].get_shape().as_list()[1]
            W = outputs_collection[iter].get_shape().as_list()[2]
        # remove the current pruned mask (for computing computation)
        unpruned_mask = mask
        channel_num = np.shape(unpruned_mask)[3]
        depth = np.shape(unpruned_mask)[2]
        pruned_channels = []
        pruned_depth = []
        # channel
        for i in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, i]) == 0:
                pruned_channels = np.append(pruned_channels, [i])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # depth
        for i in range(depth):
            if np.sum(unpruned_mask[:, :, i, :]) == 0:
                pruned_depth = np.append(pruned_depth, [i])
        unpruned_mask = np.delete(unpruned_mask, pruned_depth, axis = 2)
        
        mask_shape = np.shape(unpruned_mask)
        computation = mask_shape[0] * mask_shape[1] * mask_shape[2] * mask_shape[3] * H * W
        total_computation = total_computation + computation
    
    return total_computation

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
    
    for _, layer in enumerate(inputs_and_kernels.keys()):
        inputs_tensor = inputs_and_kernels[layer]['inputs']
        kernels_tensor = inputs_and_kernels[layer]['kernels']
        #print(kernels_tensor)
        for kernel_iter, kernel in enumerate(kernels_tensor):
            if kernel.name.split('_')[-1] == 'weights:0':# or kernel.name.split('_')[-1] == 'biases:0':
                kernel_value = sess.run(kernel)
                kernel_value = np.transpose(kernel_value, (0,1,3,2))
                #kernel_value = np.reshape(kernel_value, [np.size(kernel_value)/np.shape(kernel_value)[-1], np.shape(kernel_value)[-1]])
                if kernel_iter == 0:
                    kernel_values = kernel_value
                else:
                    kernel_values = np.concatenate([kernel_values, kernel_value], axis = 3)
        #np.savetxt(layer + '.csv', kernel_values, delimiter=",")
        kernel_values_per_layer.update({layer: kernel_values})
        #print(np.shape(kernel_values))

    return kernel_values_per_layer

#==========================#
#    Pruning Components    #
#==========================#
def compute_angle(
    x, 
    y
    ):
    cos_theta = sum(x * y) / (np.sqrt(sum(x*x)) * np.sqrt(sum(y*y)));
    if cos_theta > 1.0:
        cos_theta = float(int(cos_theta))
    theta = (np.arccos(cos_theta) / np.pi) * 180;
    
    return theta
    
def filter_prune_by_magnitude( # not finished
    prune_info_dict,
    pruning_propotion,
    pruning_layer,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]
    
    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
    
    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})

    # Build the dictionary for magnitude
    dict = {}
    iter = 0
    tmp = 0
    for layer_iter in range(0, len(sorted_layer)):
        # Current Layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer] * all_mask[layer]
        # Next Layer
        if layer_iter != len(sorted_layer)-1:
            next_layer = sorted_layer[layer_iter+1]
            next_weights = all_weights[next_layer] * all_mask[next_layer]
        
        # Calculate magnitude
        channel_num = np.shape(weights)[-1]
        for i in range(channel_num):
            if not pruned_dict[layer][str(i)]:
                x = weights[:, :, :, i]
                x = np.reshape(x, [np.size(x)])
                y = next_weights[:, :, i, :]
                y = np.reshape(y, [np.size(y)])
                magnitude = np.mean(np.abs(np.concatenate([x,y])))
                
                # Build Dictionary
                if bool(dict.get(str(magnitude))):
                    number = len(dict[str(magnitude)])
                    dict[str(magnitude)].update({str(number): {'next_layer': next_layer, 'layer': layer, 'i': i, 'pruned': False}})
                else:
                    dict.update({str(magnitude): {'0': {'next_layer': next_layer, 'layer': layer, 'i': i, 'pruned': False}}})
                # Record Magnitude
                if iter == 0:
                    total_magnitude = np.array([magnitude])
                else:
                    total_magnitude = np.concatenate([total_magnitude, np.array([magnitude])])
                iter = iter + 1    
                    
    # Sort the angle
    sorted_magnitude = np.sort(total_magnitude)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
        
    # Prune the weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel'])
    t.align = 'l'
    pruned_num = 0
    for _, magnitude in enumerate(sorted_magnitude):
        for _, index in enumerate(dict[str(magnitude)].keys()):
            if dict[str(magnitude)][index]['pruned']:
                continue
            layer      = dict[str(magnitude)][index]['layer']
            next_layer = dict[str(magnitude)][index]['next_layer']
            channel_i  = dict[str(magnitude)][index]['i']
            is_channel_i_pruned = pruned_dict[layer][str(channel_i)]
            current_weights_mask = all_mask[layer]         
            current_weights      = all_weights[layer]   
            next_weights_mask    = all_mask[next_layer]    
            next_weights         = all_weights[next_layer]
            # Assign Zero to mask
            if not is_channel_i_pruned:
                pruned_dict[layer].update({str(channel_i): True})
                current_weights_mask[:, :, :, channel_i] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, channel_i]))
                next_weights_mask[:, :, channel_i, :] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(next_weights_mask[:, :, channel_i, :]))
                t.add_row([str(pruned_num)+' / '+str(total_num), magnitude, layer, 'channel'+str(channel_i)])
            break          
        if pruned_num >= total_num * pruning_propotion:
            print(t)
            break  
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        #sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    print("Prune Over!")
            
def sparse_prune_by_magnitude(
    threshold,
    sess
    ):
    # Get the tensor
    weights_collection = tf.get_collection("float32_weights")
    mask_collection = tf.get_collection('float32_weights_mask')
    sorted_layer = []
    for i in range(len(weights_collection)):
        layer = weights_collection[i].name.split('/')[1]
        if len(layer.split('_')) == 1:
            sorted_layer.append(layer)
    sorted_layer = np.sort(np.array([int(sorted_layer[i].split('layer')[-1]) for i in range(len(sorted_layer))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(sorted_layer))]
    
    # Total Parameter Number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = tf.get_collection("float32_weights_mask", scope = 'Model/' + layer + '/')[0]
        total_num += reduce(lambda x, y: x*y, mask_tensor.get_shape().as_list())
        
    # Pruning
    pruned_num = 0
    for iter in range(len(sorted_layer)):
        layer = sorted_layer[iter]
        current_weights_tensor = tf.get_collection("float32_weights", scope = 'Model/' + layer + '/')[0]
        current_mask_tensor = tf.get_collection("float32_weights_mask", scope = 'Model/' + layer + '/')[0]
        current_weights = sess.run(current_weights_tensor)
        current_weights = np.abs(current_weights)
        mask_value = (current_weights >= threshold).astype('float32')
        sess.run(tf.assign(current_mask_tensor, mask_value))
        pruned_num = pruned_num + np.sum(mask_value==0)
        print("\r{} / {} " .format(pruned_num, total_num), end = "")
        print(np.sum(mask_value==0))
    print('Prune over!')

def filter_prune_by_angle(
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]

    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    all_outputs_shape = {}
    all_stride = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})

    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})
    
    # Build the dictionary for angle
    print('Calculating Each Layer Cosine Similarity ...')
    dict = {}
    iter = 0
    for layer_iter in range(1, len(sorted_layer)):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        weights = np.transpose(weights, (0,1,3,2))
        print(layer, end=" ")
        tStart = time.time()
        # past layer
        if layer_iter != 0:
            past_layer = sorted_layer[layer_iter-1]
        
        # remove the pruned channel
        channel_num = np.shape(weights)[2]
        pruned_channels = []
        for i in range(channel_num):
            if pruned_dict[layer][str(i)]:
                pruned_channels = np.append(pruned_channels, [i])
        weights = np.delete(weights, pruned_channels, axis = 2)

        # calculate angle
        if weights.size != 0:
            depth = np.shape(weights)[-1]
            angles_iter = 0
            for i in range(depth):
                angles_per_i_iter = 0
                for j in range(i+1, depth):
                    # If the depth i or j has been pruned, we don't calculate its angle to others
                    # By doing so, we will not prune the same kernel which has been pruned before.
                    if not pruned_dict[past_layer][str(i)] and not pruned_dict[past_layer][str(j)]:
                        x = weights[:, :, :, i]
                        y = weights[:, :, :, j]
                        x = np.reshape(x, [np.size(x)])
                        y = np.reshape(y, [np.size(y)])
                        
                        if sum(x*x) == 0 or sum(y*y) == 0:
                            angle = 90.0
                            print('{}, {}, {}' .format(layer, i, j))
                        else:
                            angle = compute_angle(x, y)
                            angle = abs(angle - 90)
                        
                        if bool(dict.get(str(angle))):
                            number = len(dict[str(angle)])
                            if layer_iter != 0:
                                dict[str(angle)].update({str(number): {'past_layer': past_layer, 'layer': layer, 'i': i, 'j': j, 'pruned': False}})
                            else:
                                dict[str(angle)].update({str(number): {'layer': layer, 'i': i, 'j': j, 'pruned': False}})
                        else:
                            if layer_iter != 0:
                                dict.update({str(angle): {'0': {'past_layer': past_layer, 'layer': layer, 'i': i, 'j': j, 'pruned': False}}})
                            else:
                                dict.update({str(angle): {'0': {'layer': layer, 'i': i, 'j': j, 'pruned': False}}})
                        
                        if angles_per_i_iter == 0:
                            angles_per_i = np.array([angle])
                        else:
                            angles_per_i = np.concatenate([angles_per_i, np.array([angle])])
                        angles_per_i_iter = angles_per_i_iter + 1
                
                if angles_per_i_iter != 0:
                    if angles_iter == 0:
                        angles = angles_per_i
                    else:
                        angles = np.concatenate([angles, angles_per_i])
                    angles_iter = angles_iter + 1
            
            if iter == 0:
                total_angle = angles
            else:
                total_angle = np.concatenate([total_angle, angles])
            iter = iter + 1   
        
        tEnd = time.time()
        print("(cost %f seconds)" %(tEnd - tStart))
    # Sort the angle
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            # past layer
            #if bool(dict_[str(angle)][index].get('past_layer')):
            past_layer = dict_[str(angle)][index]['past_layer']
            past_weights_mask = all_mask[past_layer]
            # current layer
            layer = dict_[str(angle)][index]['layer']
            current_weights_mask = all_mask[layer]
            # depth
            depth_i = dict_[str(angle)][index]['i']
            depth_j = dict_[str(angle)][index]['j']

            if not bool(weights_to_be_pruned.get(layer)):
                is_depth_i_appear = False
                is_depth_j_appear = False
                weights_to_be_pruned.update({layer: {str(depth_i): 1, str(depth_j): 1}})
            else:
                is_depth_i_appear = bool(weights_to_be_pruned[layer].get(str(depth_i)))
                is_depth_j_appear = bool(weights_to_be_pruned[layer].get(str(depth_j)))
                if not is_depth_i_appear:
                    weights_to_be_pruned[layer].update({str(depth_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_i): weights_to_be_pruned[layer][str(depth_i)]+1})
                if not is_depth_j_appear:
                    weights_to_be_pruned[layer].update({str(depth_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_j): weights_to_be_pruned[layer][str(depth_j)]+1})
            if not is_depth_i_appear or not is_depth_j_appear:
                #if bool(dict_[str(angle)][index].get('past_layer')):
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, depth_i]))
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
            #print(pruned_num)
        if pruned_num >= total_num * pruning_propotion:
            break    
            
    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel', 'Computation'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            if dict[str(angle)][index]['pruned']:
                continue
            # past layer
            #if bool(dict[str(angle)][index].get('past_layer')):
            past_layer = dict[str(angle)][index]['past_layer']
            past_weights_mask  = all_mask[past_layer]    
            past_weights       = all_weights[past_layer]
            past_outputs_shape = all_outputs_shape[past_layer]
            # current layer
            layer = dict[str(angle)][index]['layer']
            current_weights_mask  = all_mask[layer]
            current_weights       = all_weights[layer]
            current_outputs_shape = all_outputs_shape[layer]
            # depth
            depth_i = dict[str(angle)][index]['i']
            depth_j = dict[str(angle)][index]['j']
            is_depth_i_pruned = pruned_dict[past_layer][str(depth_i)]
            is_depth_j_pruned = pruned_dict[past_layer][str(depth_j)]
 
            if not is_depth_i_pruned and not is_depth_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                if weights_to_be_pruned[layer][str(depth_i)] < weights_to_be_pruned[layer][str(depth_j)]:
                    pruned_depth = depth_i
                elif weights_to_be_pruned[layer][str(depth_i)] > weights_to_be_pruned[layer][str(depth_j)]:
                    pruned_depth = depth_j
                else:
                    # Magnitude
                    #if bool(dict[str(angle)][index].get('past_layer')):
                    sum_of_past_layer_kernal_i = np.sum(np.abs(past_weights[:, :, :, depth_i] * past_weights_mask[:, :, :, depth_i]))
                    sum_of_past_layer_kernel_j = np.sum(np.abs(past_weights[:, :, :, depth_j] * past_weights_mask[:, :, :, depth_j]))
                    #else:
                    #    sum_of_past_layer_kernal_i = 0
                    #    sum_of_past_layer_kernel_j = 0
                    sum_of_current_layer_depth_i = np.sum(np.abs(current_weights[:, :, depth_i, :] * current_weights_mask[:, :, depth_i, :]))
                    sum_of_current_layer_depth_j = np.sum(np.abs(current_weights[:, :, depth_j, :] * current_weights_mask[:, :, depth_j, :]))
                    sum_of_pruned_depth_i = sum_of_past_layer_kernal_i + sum_of_current_layer_depth_i
                    sum_of_pruned_depth_j = sum_of_past_layer_kernel_j + sum_of_current_layer_depth_j
                    # Compare
                    if sum_of_pruned_depth_i <= sum_of_pruned_depth_j:
                        pruned_depth = depth_i
                    else:
                        pruned_depth = depth_j
                
                # remove the past pruned mask (for computing computation)
                unpruned_past_mask = past_weights_mask
                past_channel_num = np.shape(past_weights_mask)[3]
                past_depth = np.shape(past_weights_mask)[2]
                past_pruned_channels = []
                past_pruned_depth = []
                # (channel)
                for i in range(past_channel_num):
                    if np.sum(past_weights_mask[:, :, :, i]) == 0:
                        past_pruned_channels = np.append(past_pruned_channels, [i])
                unpruned_past_mask = np.delete(unpruned_past_mask, past_pruned_channels, axis = 3)
                # (depth)
                for i in range(past_depth):
                    if np.sum(past_weights_mask[:, :, i, :]) == 0:
                        past_pruned_depth = np.append(past_pruned_depth, [i])
                unpruned_past_mask = np.delete(unpruned_past_mask, past_pruned_depth, axis = 2)
                
                # remove the current pruned mask (for computing computation)
                unpruned_current_mask = current_weights_mask
                current_channel_num = np.shape(current_weights_mask)[3]
                current_depth = np.shape(current_weights_mask)[2]
                current_pruned_channels = []
                current_pruned_depth = []
                # (channel)
                for i in range(current_channel_num):
                    if np.sum(current_weights_mask[:, :, :, i]) == 0:
                        current_pruned_channels = np.append(current_pruned_channels, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_channels, axis = 3)
                # (depth)
                for i in range(current_depth):
                    if np.sum(current_weights_mask[:, :, i, :]) == 0:
                        current_pruned_depth = np.append(current_pruned_depth, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_depth, axis = 2)
                
                # Assign Zero to mask
                tmp = {}
                computation = 0
                pruned_dict[past_layer].update({str(pruned_depth): True})
                #if bool(dict[str(angle)][index].get('past_layer')):
                past_weights_mask[:, :, :, pruned_depth] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, depth_i]))
                past_weights_shape = np.shape(unpruned_past_mask)
                computation = computation + past_weights_shape[0] * past_weights_shape[1] * past_weights_shape[2] * past_outputs_shape[2] * past_outputs_shape[3]
                tmp.update({past_layer: {'channel': pruned_depth}})
                current_weights_mask[:, :, pruned_depth, :] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
                current_weights_shape = np.shape(unpruned_current_mask)
                computation = computation + current_weights_shape[0] * current_weights_shape[1] * current_weights_shape[3] * current_outputs_shape[2] * current_outputs_shape[3]
                tmp.update({layer: {'depth': pruned_depth}})
                tmp.update({'computation': computation})
                pruned_weights_info.append(tmp)
                t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'depth' + str(pruned_depth), computation])
                #print("\r{} / {}	" .format(pruned_num, total_num), end = "")
                #print("{}	{}	{}" .format(angle, layer, 'depth' + str(pruned_depth)))
            break
    print(t)
    
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        #sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    """
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    """
    # See the parameter shape change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        mask = sess.run(mask_tensor)
        # remove the pruned part
        unpruned_mask = mask
        # (channel)
        channel_num = np.shape(mask)[3]
        pruned_channels = []
        for channel in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, channel]) == 0:
                pruned_channels = np.append(pruned_channels, [channel])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # (depth)
        depths = np.shape(unpruned_mask)[2]
        pruned_depths = []
        for depth in range(depths):
            if np.sum(unpruned_mask[:, :, depth, :]) == 0:
                pruned_depths = np.append(pruned_depths, [depth])
        unpruned_mask = np.delete(unpruned_mask, pruned_depths, axis = 2)
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' 
        .format(layer, np.shape(mask), np.shape(unpruned_mask)))
    print("Prune Over!")
    
    return pruned_weights_info
    
def filter_prune_by_angleII(
    prune_info_dict,
    pruning_propotion,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]
    
    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
    
    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})

    # Build the dictionary for angle
    dict = {}
    iter = 0
    tmp = 0
    for layer_iter in range(0, len(sorted_layer)-1):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        # past layer
        if layer_iter != 0:
            past_layer = sorted_layer[layer_iter-1]
            past_weights = all_weights[past_layer]
        #next layer
        if layer_iter != len(sorted_layer)-1:
            next_layer = sorted_layer[layer_iter+1]
            next_weights = all_weights[next_layer]
        
        # remove the pruned depth
        # First layer have no pruned in depth dimension
        if layer_iter != 0:
            depth = np.shape(past_weights)[3] # past layer channel number equals to current layer depth
            pruned_depths = []
            for i in range(depth):
                if pruned_dict[past_layer][str(i)]:
                    pruned_depths = np.append(pruned_depths, [i])
            weights = np.delete(weights, pruned_depths, axis = 2)
        
        # calculate angle
        if weights.size != 0:
            channel = np.shape(weights)[3]
            for i in range(channel):
                for j in range(i+1, channel):
                    # If the channel i or j has been pruned, we don't calculate its angle to others
                    # By doing so, we will not prune the kernel which has been pruned before.
                    if not pruned_dict[layer][str(i)] and not pruned_dict[layer][str(j)]:
                        x = weights[:, :, :, i]
                        y = weights[:, :, :, j]
                        x = np.reshape(x, [np.size(x)])
                        y = np.reshape(y, [np.size(y)])
                        
                        if sum(x*x) == 0 or sum(y*y) == 0:
                            angle = 90.0
                            tmp = tmp + 1
                            print('{}, {}, {}' .format(layer, i, j))
                        else:
                            angle = compute_angle(x, y)
                            angle = abs(angle - 90)
                        
                        if bool(dict.get(str(angle))):
                            number = len(dict[str(angle)])
                            if layer_iter != len(sorted_layer)-1:
                                dict[str(angle)].update({str(number): {'next_layer': next_layer, 'layer': layer, 'i': i, 'j': j, 'pruned': False}})
                            else:
                                dict[str(angle)].update({str(number): {'layer': layer, 'i': i, 'j': j, 'pruned': False}})
                        else:
                            if layer_iter != len(sorted_layer)-1:
                                dict.update({str(angle): {'0': {'next_layer': next_layer, 'layer': layer, 'i': i, 'j': j, 'pruned': False}}})
                            else:
                                dict.update({str(angle): {'0': {'layer': layer, 'i': i, 'j': j, 'pruned': False}}})
                        
                        if iter == 0:
                            total_angle = np.array([angle])
                        else:
                            total_angle = np.concatenate([total_angle, np.array([angle])])
                        iter = iter + 1    
    # Sort the angle
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            # next layer
            if bool(dict_[str(angle)][index].get('next_layer')):
                next_layer = dict_[str(angle)][index]['next_layer']
                next_weights_mask = all_mask[next_layer]
            # current layer
            layer = dict_[str(angle)][index]['layer']
            current_weights_mask = all_mask[layer]
            # channel
            channel_i = dict_[str(angle)][index]['i']
            channel_j = dict_[str(angle)][index]['j']
            
            if not bool(weights_to_be_pruned.get(layer)):
                is_channel_i_appear = False
                is_channel_j_appear = False
                weights_to_be_pruned.update({layer: {str(channel_i): 1, str(channel_j): 1}})
            else:
                is_channel_i_appear = bool(weights_to_be_pruned[layer].get(str(channel_i)))
                is_channel_j_appear = bool(weights_to_be_pruned[layer].get(str(channel_j)))
                if not is_channel_i_appear:
                    weights_to_be_pruned[layer].update({str(channel_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(channel_i): weights_to_be_pruned[layer][str(channel_i)]+1})
                if not is_channel_j_appear:
                    weights_to_be_pruned[layer].update({str(channel_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(channel_j): weights_to_be_pruned[layer][str(channel_j)]+1})
            if not is_channel_i_appear or not is_channel_j_appear:    
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, channel_i]))
                if bool(dict_[str(angle)][index].get('next_layer')):
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(next_weights_mask[:, :, channel_i, :]))
            #print(pruned_num)
        if pruned_num >= total_num * pruning_propotion:
            break    
            
    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            if dict[str(angle)][index]['pruned']:
                continue
            # next layer
            if bool(dict[str(angle)][index].get('next_layer')):
                next_layer = dict[str(angle)][index]['next_layer']
                next_weights_mask = all_mask[next_layer]    
                next_weights      = all_weights[next_layer] 
            # current layer
            layer = dict[str(angle)][index]['layer']
            current_weights_mask = all_mask[layer]         
            current_weights      = all_weights[layer]
            # channel
            channel_i = dict[str(angle)][index]['i']
            channel_j = dict[str(angle)][index]['j']
            is_channel_i_pruned = pruned_dict[layer][str(channel_i)]
            is_channel_j_pruned = pruned_dict[layer][str(channel_j)]
 
            if not is_channel_i_pruned and not is_channel_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                if weights_to_be_pruned[layer][str(channel_i)] < weights_to_be_pruned[layer][str(channel_j)]:
                    pruned_channel = channel_i
                elif weights_to_be_pruned[layer][str(channel_i)] > weights_to_be_pruned[layer][str(channel_j)]:
                    pruned_channel = channel_j
                else:
                    # Magnitude
                    if bool(dict[str(angle)][index].get('next_layer')):
                        sum_of_next_layer_depth_i = np.sum(np.abs(next_weights[:, :, channel_i, :] * next_weights_mask[:, :, channel_i, :]))
                        sum_of_next_layer_depth_j = np.sum(np.abs(next_weights[:, :, channel_j, :] * next_weights_mask[:, :, channel_j, :]))
                    else:
                        sum_of_next_layer_depth_i = 0
                        sum_of_next_layer_depth_j = 0
                    sum_of_current_layer_channel_i = np.sum(np.abs(current_weights[:, :, :, channel_i] * current_weights_mask[:, :, :, channel_i]))
                    sum_of_current_layer_channel_j = np.sum(np.abs(current_weights[:, :, :, channel_j] * current_weights_mask[:, :, :, channel_j]))
                    sum_of_pruned_channel_i = sum_of_next_layer_depth_i + sum_of_current_layer_channel_i
                    sum_of_pruned_channel_j = sum_of_next_layer_depth_j + sum_of_current_layer_channel_j
                    # Compare
                    if sum_of_pruned_channel_i <= sum_of_pruned_channel_j:
                        pruned_channel = channel_i
                    else:
                        pruned_channel = channel_j
                # Assign Zero to mask
                pruned_dict[layer].update({str(pruned_channel): True})
                current_weights_mask[:, :, :, pruned_channel] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, channel_i]))
                if bool(dict[str(angle)][index].get('next_layer')):
                    next_weights_mask[:, :, pruned_channel, :] = 0
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(next_weights_mask[:, :, channel_i, :]))
                t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'channel' + str(pruned_channel)])
            break
    #print(t)
    
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)): 
        layer = sorted_layer[layer_iter]
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    """
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    """
    # See the parameter shape change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        mask = sess.run(mask_tensor)
        # remove the pruned part
        unpruned_mask = mask
        # (channel)
        channel_num = np.shape(mask)[3]
        pruned_channels = []
        for channel in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, channel]) == 0:
                pruned_channels = np.append(pruned_channels, [channel])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # (depth)
        depths = np.shape(unpruned_mask)[2]
        pruned_depths = []
        for depth in range(depths):
            if np.sum(unpruned_mask[:, :, depth, :]) == 0:
                pruned_depths = np.append(pruned_depths, [depth])
        unpruned_mask = np.delete(unpruned_mask, pruned_depths, axis = 2)
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' 
        .format(layer, np.shape(mask), np.shape(unpruned_mask)))
    print("Prune Over!")

def filter_prune_by_angle_with_skip(
    prune_info_dict,
    pruning_propotion,
    skip_layer,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]
    
    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
    
    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(1, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})

    # Build the dictionary for angle
    dict = {}
    iter = 0
    tmp = 0
    for layer_iter in range(1, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        weights = np.transpose(weights, (0,1,3,2))
        # remove the pruned kernel
        channel_num = np.shape(weights)[2]
        pruned_channels = []
        for i in range(channel_num):
            if pruned_dict[layer][str(i)]:
                pruned_channels = np.append(pruned_channels, [i])
        weights = np.delete(weights, pruned_channels, axis = 2)

        if weights.size != 0:
            # calculate angle
            past_layer = sorted_layer[layer_iter-1]
            depth = np.shape(weights)[-1]
            for i in range(depth):
                for j in range(i+1, depth):
                    # If the depth i or j has been pruned, we don't calculate its angle to others
                    # By doing so, we will not prune the same kernel which has been pruned before.
                    if not pruned_dict[past_layer][str(i)] and not pruned_dict[past_layer][str(j)]:
                        x = weights[:, :, :, i]
                        y = weights[:, :, :, j]
                        x = np.reshape(x, [np.size(x)])
                        y = np.reshape(y, [np.size(y)])
                        
                        if sum(x*x) == 0 or sum(y*y) == 0:
                            angle = 90.0
                            tmp = tmp + 1
                            print('{}, {}, {}' .format(layer, i, j))
                        else:
                            angle = compute_angle(x, y)
                            angle = abs(angle - 90)
                        
                        if bool(dict.get(str(angle))):
                            number = len(dict[str(angle)])
                            dict[str(angle)].update({str(number): {'past_layer': past_layer, 'layer': layer, 'i': i, 'j': j, 'pruned': False}})
                        else:
                            dict.update({str(angle): {'0': {'past_layer': past_layer, 'layer': layer, 'i': i, 'j': j, 'pruned': False}}})
                        
                        if iter == 0:
                            total_angle = np.array([angle])
                        else:
                            total_angle = np.concatenate([total_angle, np.array([angle])])
                        iter = iter + 1    
                    
    # Sort the angle
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            past_layer = dict_[str(angle)][index]['past_layer']
            layer = dict_[str(angle)][index]['layer']
            depth_i = dict_[str(angle)][index]['i']
            depth_j = dict_[str(angle)][index]['j']
            past_weights_mask = all_mask[past_layer]
            current_weights_mask = all_mask[layer]
            
            if not bool(weights_to_be_pruned.get(layer)):
                is_depth_i_appear = False
                is_depth_j_appear = False
                weights_to_be_pruned.update({layer: {str(depth_i): 1, str(depth_j): 1}})
            else:
                is_depth_i_appear = bool(weights_to_be_pruned[layer].get(str(depth_i)))
                is_depth_j_appear = bool(weights_to_be_pruned[layer].get(str(depth_j)))
                if not is_depth_i_appear:
                    weights_to_be_pruned[layer].update({str(depth_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_i): weights_to_be_pruned[layer][str(depth_i)]+1})
                if not is_depth_j_appear:
                    weights_to_be_pruned[layer].update({str(depth_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_j): weights_to_be_pruned[layer][str(depth_j)]+1})
            if not is_depth_i_appear or not is_depth_j_appear:
                is_past_layer_skip = any(past_layer == 'layer'+str(skip_layer_) for skip_layer_ in skip_layer)
                is_layer_skip = any(layer == 'layer'+str(skip_layer_) for skip_layer_ in skip_layer)
                if (not is_past_layer_skip) and (not is_layer_skip):
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, depth_i]))
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
            #print(pruned_num)
        if pruned_num >= total_num * pruning_propotion:
            break    

    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            #print(dict[str(angle)].keys())
            if dict[str(angle)][index]['pruned']:
                continue
            past_layer = dict[str(angle)][index]['past_layer']
            layer = dict[str(angle)][index]['layer']
            depth_i = dict[str(angle)][index]['i']
            depth_j = dict[str(angle)][index]['j']
            is_depth_i_pruned = pruned_dict[past_layer][str(depth_i)]
            is_depth_j_pruned = pruned_dict[past_layer][str(depth_j)]
            past_weights_mask    = all_mask[past_layer]    #prune_info_dict[past_layer]['mask']
            past_weights         = all_weights[past_layer] #prune_info_dict[past_layer]['weights']
            current_weights_mask = all_mask[layer]         #prune_info_dict[layer]['mask']
            current_weights      = all_weights[layer]      #prune_info_dict[layer]['weights']
            if not is_depth_i_pruned and not is_depth_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                if weights_to_be_pruned[layer][str(depth_i)] < weights_to_be_pruned[layer][str(depth_j)]:
                    pruned_depth = depth_i
                elif weights_to_be_pruned[layer][str(depth_i)] > weights_to_be_pruned[layer][str(depth_j)]:
                    pruned_depth = depth_j
                else:
                    # Magnitude
                    sum_of_past_layer_kernal_i = np.sum(np.abs(past_weights[:, :, :, depth_i] * past_weights_mask[:, :, :, depth_i]))
                    sum_of_past_layer_kernel_j = np.sum(np.abs(past_weights[:, :, :, depth_j] * past_weights_mask[:, :, :, depth_j]))
                    sum_of_current_layer_depth_i = np.sum(np.abs(current_weights[:, :, depth_i, :] * current_weights_mask[:, :, depth_i, :]))
                    sum_of_current_layer_depth_j = np.sum(np.abs(current_weights[:, :, depth_j, :] * current_weights_mask[:, :, depth_j, :]))
                    sum_of_pruned_depth_i = sum_of_past_layer_kernal_i + sum_of_current_layer_depth_i
                    sum_of_pruned_depth_j = sum_of_past_layer_kernel_j + sum_of_current_layer_depth_j
                    # Compare
                    if sum_of_pruned_depth_i <= sum_of_pruned_depth_j:
                        pruned_depth = depth_i
                    else:
                        pruned_depth = depth_j

                # Assign Zero to mask
                pruned_dict[past_layer].update({str(pruned_depth): True})
                is_past_layer_skip = any(past_layer == 'layer'+str(skip_layer_) for skip_layer_ in skip_layer)
                is_layer_skip = any(layer == 'layer'+str(skip_layer_) for skip_layer_ in skip_layer)
                if (not is_past_layer_skip) and (not is_layer_skip):
                    past_weights_mask[:, :, :, pruned_depth] = 0
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, depth_i]))
                    current_weights_mask[:, :, pruned_depth, :] = 0
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
                    t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'depth'+str(pruned_depth)])
                    print("\r{} / {}	" .format(pruned_num, total_num), end = "")
                    print("{}	{}	{}" .format(angle, layer, 'depth' + str(pruned_depth)))
                break          
    #print(t)
    
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        #sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    """
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    """
    # See the parameter shape change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        mask = sess.run(mask_tensor)
        # remove the pruned part
        unpruned_mask = mask
        # (channel)
        channel_num = np.shape(mask)[3]
        pruned_channels = []
        for channel in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, channel]) == 0:
                pruned_channels = np.append(pruned_channels, [channel])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # (depth)
        depths = np.shape(unpruned_mask)[2]
        pruned_depths = []
        for depth in range(depths):
            if np.sum(unpruned_mask[:, :, depth, :]) == 0:
                pruned_depths = np.append(pruned_depths, [depth])
        unpruned_mask = np.delete(unpruned_mask, pruned_depths, axis = 2)
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' 
        .format(layer, np.shape(mask), np.shape(unpruned_mask)))
    print("Prune Over!")

def filter_prune_by_angle_with_penalty(
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    max_penalty,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]

    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    all_outputs_shape = {}
    all_stride = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})

    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})
    
    # angles
    angle_dict = {}
    all_angles = {}
    for layer_iter in range(1, len(sorted_layer)):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        weights = np.transpose(weights, (0,1,3,2))
        # past layer
        if layer_iter != 0:
            past_layer = sorted_layer[layer_iter-1]
            
        # remove the pruned channel
        channel_num = np.shape(weights)[2]
        pruned_channels = []
        for i in range(channel_num):
            if pruned_dict[layer][str(i)]:
                pruned_channels = np.append(pruned_channels, [i])
        weights = np.delete(weights, pruned_channels, axis = 2)
        
        # calculate angle
        if weights.size != 0:
            depth = np.shape(weights)[-1]
            angles = []
            for i in range(depth):
                for j in range(i+1, depth):
                    # If the depth i or j has been pruned, we don't calculate its angle to others
                    # By doing so, we will not prune the same kernel which has been pruned before.
                    if not pruned_dict[past_layer][str(i)] and not pruned_dict[past_layer][str(j)]:
                        # angle
                        x = weights[:, :, :, i]
                        y = weights[:, :, :, j]
                        x = np.reshape(x, [np.size(x)])
                        y = np.reshape(y, [np.size(y)])
                        
                        if sum(x*x) == 0 or sum(y*y) == 0:
                            angle = 90.0
                            tmp = tmp + 1
                            print('{}, {}, {}' .format(layer, i, j))
                        else:
                            angle = compute_angle(x, y)
                            angle = abs(angle - 90.)
                        
                        # angle_dict
                        if not bool(angle_dict.get(layer)):
                            angle_dict.update({layer: {i: {j: angle}}})
                        else:
                            angle_dict[layer].update({i: {j: angle}})
                        
                        # angles
                        if not sum(x*x) == 0 or sum(y*y) == 0:
                            angles = np.append(angles, compute_angle(x, y))
                            
            all_angles.update({layer: angles})
    
    # Penalty
    all_penalty = {}
    penalty_regression = 'stdev'
    for layer_iter in range(1, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        if penalty_regression == 'linear':
            interval = float(max_penalty-1) / float(len(sorted_layer)-1)
            penalty = 1 + layer_iter * interval
            all_penalty.update({layer: penalty})
        elif penalty_regression == 'stdev':
            angles = all_angles[layer]
            stdev = np.std(angles)
            mean = np.mean(angles)
            max = np.max(angles)
            min = np.min(angles)
            all_penalty.update({layer: stdev})
            print("%8s -> \033[1;32mstdev\033[0m: %10f, \033[1;32mmean\033[0m: %10f" %(layer, stdev, mean) )
    
    # Build the dictionary for angle
    dict = {}
    iter = 0
    tmp = 0
    for layer_iter in range(1, len(sorted_layer)):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        weights = np.transpose(weights, (0,1,3,2))
        penalty = all_penalty[layer]
        # past layer
        if layer_iter != 0:
            past_layer = sorted_layer[layer_iter-1]
            
        # remove the pruned channel
        channel_num = np.shape(weights)[2]
        pruned_channels = []
        for i in range(channel_num):
            if pruned_dict[layer][str(i)]:
                pruned_channels = np.append(pruned_channels, [i])
        weights = np.delete(weights, pruned_channels, axis = 2)
        
        # calculate angle
        if weights.size != 0:
            for i in angle_dict[layer]:
                for j in angle_dict[layer][i]:
                    # angle
                    angle = angle_dict[layer][i][j]
                    angle = angle * penalty
                    
                    # dict
                    if bool(dict.get(str(angle))):
                        number = len(dict[str(angle)])
                        if layer_iter != 0:
                            dict[str(angle)].update({str(number): {'past_layer': past_layer, 'layer': layer, 'i': int(i), 'j': int(j), 'pruned': False}})
                        else:
                            dict[str(angle)].update({str(number): {'layer': layer, 'i': int(i), 'j': int(j), 'pruned': False}})
                    else:
                        if layer_iter != 0:
                            dict.update({str(angle): {'0': {'past_layer': past_layer, 'layer': layer, 'i': int(i), 'j': int(j), 'pruned': False}}})
                        else:
                            dict.update({str(angle): {'0': {'layer': layer, 'i': int(i), 'j': int(j), 'pruned': False}}})
                    
                    if iter == 0:
                        total_angle = np.array([angle])
                    else:
                        total_angle = np.concatenate([total_angle, np.array([angle])])
                    iter = iter + 1    
                    
    # Sort the angle
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            # past layer
            #if bool(dict_[str(angle)][index].get('past_layer')):
            past_layer = dict_[str(angle)][index]['past_layer']
            past_weights_mask = all_mask[past_layer]
            # current layer
            layer = dict_[str(angle)][index]['layer']
            current_weights_mask = all_mask[layer]
            # depth
            depth_i = dict_[str(angle)][index]['i']
            depth_j = dict_[str(angle)][index]['j']

            if not bool(weights_to_be_pruned.get(layer)):
                is_depth_i_appear = False
                is_depth_j_appear = False
                weights_to_be_pruned.update({layer: {str(depth_i): 1, str(depth_j): 1}})
            else:
                is_depth_i_appear = bool(weights_to_be_pruned[layer].get(str(depth_i)))
                is_depth_j_appear = bool(weights_to_be_pruned[layer].get(str(depth_j)))
                if not is_depth_i_appear:
                    weights_to_be_pruned[layer].update({str(depth_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_i): weights_to_be_pruned[layer][str(depth_i)]+1})
                if not is_depth_j_appear:
                    weights_to_be_pruned[layer].update({str(depth_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_j): weights_to_be_pruned[layer][str(depth_j)]+1})
            if not is_depth_i_appear or not is_depth_j_appear:
                #if bool(dict_[str(angle)][index].get('past_layer')):
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, depth_i]))
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
            #print(pruned_num)
        if pruned_num >= total_num * pruning_propotion:
            break    
            
    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel', 'Computation'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            if dict[str(angle)][index]['pruned']:
                continue
            # past layer
            #if bool(dict[str(angle)][index].get('past_layer')):
            past_layer = dict[str(angle)][index]['past_layer']
            past_weights_mask  = all_mask[past_layer]    
            past_weights       = all_weights[past_layer]
            past_outputs_shape = all_outputs_shape[past_layer]
            # current layer
            layer = dict[str(angle)][index]['layer']
            current_weights_mask  = all_mask[layer]
            current_weights       = all_weights[layer]
            current_outputs_shape = all_outputs_shape[layer]
            # depth
            depth_i = dict[str(angle)][index]['i']
            depth_j = dict[str(angle)][index]['j']
            is_depth_i_pruned = pruned_dict[past_layer][str(depth_i)]
            is_depth_j_pruned = pruned_dict[past_layer][str(depth_j)]
 
            if not is_depth_i_pruned and not is_depth_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                if weights_to_be_pruned[layer][str(depth_i)] < weights_to_be_pruned[layer][str(depth_j)]:
                    pruned_depth = depth_i
                elif weights_to_be_pruned[layer][str(depth_i)] > weights_to_be_pruned[layer][str(depth_j)]:
                    pruned_depth = depth_j
                else:
                    # Magnitude
                    #if bool(dict[str(angle)][index].get('past_layer')):
                    sum_of_past_layer_kernal_i = np.sum(np.abs(past_weights[:, :, :, depth_i] * past_weights_mask[:, :, :, depth_i]))
                    sum_of_past_layer_kernel_j = np.sum(np.abs(past_weights[:, :, :, depth_j] * past_weights_mask[:, :, :, depth_j]))
                    #else:
                    #    sum_of_past_layer_kernal_i = 0
                    #    sum_of_past_layer_kernel_j = 0
                    sum_of_current_layer_depth_i = np.sum(np.abs(current_weights[:, :, depth_i, :] * current_weights_mask[:, :, depth_i, :]))
                    sum_of_current_layer_depth_j = np.sum(np.abs(current_weights[:, :, depth_j, :] * current_weights_mask[:, :, depth_j, :]))
                    sum_of_pruned_depth_i = sum_of_past_layer_kernal_i + sum_of_current_layer_depth_i
                    sum_of_pruned_depth_j = sum_of_past_layer_kernel_j + sum_of_current_layer_depth_j
                    # Compare
                    if sum_of_pruned_depth_i <= sum_of_pruned_depth_j:
                        pruned_depth = depth_i
                    else:
                        pruned_depth = depth_j
                
                # remove the past pruned mask (for computing computation)
                unpruned_past_mask = past_weights_mask
                past_channel_num = np.shape(past_weights_mask)[3]
                past_depth = np.shape(past_weights_mask)[2]
                past_pruned_channels = []
                past_pruned_depth = []
                # channel
                for i in range(past_channel_num):
                    if np.sum(past_weights_mask[:, :, :, i]) == 0:
                        past_pruned_channels = np.append(past_pruned_channels, [i])
                unpruned_past_mask = np.delete(unpruned_past_mask, past_pruned_channels, axis = 3)
                # depth
                for i in range(past_depth):
                    if np.sum(past_weights_mask[:, :, i, :]) == 0:
                        past_pruned_depth = np.append(past_pruned_depth, [i])
                unpruned_past_mask = np.delete(unpruned_past_mask, past_pruned_depth, axis = 2)
                
                # remove the current pruned mask (for computing computation)
                unpruned_current_mask = current_weights_mask
                current_channel_num = np.shape(current_weights_mask)[3]
                current_depth = np.shape(current_weights_mask)[2]
                current_pruned_channels = []
                current_pruned_depth = []
                # (channel)
                for i in range(current_channel_num):
                    if np.sum(current_weights_mask[:, :, :, i]) == 0:
                        current_pruned_channels = np.append(current_pruned_channels, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_channels, axis = 3)
                # (depth)
                for i in range(current_depth):
                    if np.sum(current_weights_mask[:, :, i, :]) == 0:
                        current_pruned_depth = np.append(current_pruned_depth, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_depth, axis = 2)
                
                # Assign Zero to mask
                tmp = {}
                computation = 0
                pruned_dict[past_layer].update({str(pruned_depth): True})
                #if bool(dict[str(angle)][index].get('past_layer')):
                past_weights_mask[:, :, :, pruned_depth] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, depth_i]))
                past_weights_shape = np.shape(unpruned_past_mask)
                computation = computation + past_weights_shape[0] * past_weights_shape[1] * past_weights_shape[2] * past_outputs_shape[2] * past_outputs_shape[3]
                tmp.update({past_layer: {'channel': pruned_depth}})
                current_weights_mask[:, :, pruned_depth, :] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
                current_weights_shape = np.shape(unpruned_current_mask)
                computation = computation + current_weights_shape[0] * current_weights_shape[1] * current_weights_shape[3] * current_outputs_shape[2] * current_outputs_shape[3]
                tmp.update({layer: {'depth': pruned_depth}})
                tmp.update({'computation': computation})
                pruned_weights_info.append(tmp)
                t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'depth' + str(pruned_depth), computation])
                #print("\r{} / {}	" .format(pruned_num, total_num), end = "")
                #print("{}	{}	{}" .format(angle, layer, 'depth' + str(pruned_depth)))
            break
    #print(t)
    
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        #sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    """
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    """
    # See the parameter shape change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        mask = sess.run(mask_tensor)
        # remove the pruned part
        unpruned_mask = mask
        # (channel)
        channel_num = np.shape(mask)[3]
        pruned_channels = []
        for channel in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, channel]) == 0:
                pruned_channels = np.append(pruned_channels, [channel])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # (depth)
        depths = np.shape(unpruned_mask)[2]
        pruned_depths = []
        for depth in range(depths):
            if np.sum(unpruned_mask[:, :, depth, :]) == 0:
                pruned_depths = np.append(pruned_depths, [depth])
        unpruned_mask = np.delete(unpruned_mask, pruned_depths, axis = 2)
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' 
        .format(layer, np.shape(mask), np.shape(unpruned_mask)))
    print("Prune Over!")
    
    return pruned_weights_info
    
def plane_prune_by_angle(
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]

    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    all_outputs_shape = {}
    all_stride = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})
    
    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(1, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        depth = np.shape(mask)[2]
        tmp_dict = {}
        for i in range(channel_num):
            for j in range(depth):
                if j == 0:
                    tmp_dict.update({str(i): {str(j): np.mean(mask[:, :, j, i] == 0)}})
                else:
                    tmp_dict[str(i)].update({str(j): np.mean(mask[:, :, j, i] == 0)})
        pruned_dict.update({layer: tmp_dict})

    # Build the dictionary for angle
    dict = {}
    iter = 0
    tmp = 0
    for layer_iter in range(1, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        weights = np.transpose(weights, (0,1,3,2))
        # calculate angle
        past_layer = sorted_layer[layer_iter-1]
        channel_num = np.shape(weights)[2]
        depth = np.shape(weights)[-1]
        
        weights_distance = np.sqrt(np.sum((weights * weights), axis = (0,1)))
        weights_distance = np.expand_dims(weights_distance, axis = 0)
        weights_distance = np.expand_dims(weights_distance, axis = 0)
        normalized_weights = weights / weights_distance   
        max_angle = 0
        for i in range(depth):
            for j in range(i+1, depth):
                channel_x = normalized_weights[:, :, :, i]
                channel_y = normalized_weights[:, :, :, j]
                channel_cos_theta = np.sum(channel_x * channel_y, axis = (0,1))
                channel_theta = (np.arccos(channel_cos_theta) / np.pi) * 180.
                channel_angle = np.abs(90.0 - channel_theta)
                unpruned_angle_iter = 0
                for channel in range(channel_num):
                    if not pruned_dict[layer][str(channel)][str(i)] and not pruned_dict[layer][str(channel)][str(j)]:
                        angle = channel_angle[channel]
                        if max_angle < angle:
                            max_angle = angle
                        if bool(dict.get(str(angle))):
                            number = len(dict[str(angle)])
                            dict[str(angle)].update({str(number): {'past_layer': past_layer, 'layer': layer, 'channel': channel, 'i': i, 'j': j, 'pruned': False}})
                        else:
                            dict.update({str(angle): {'0': {'past_layer': past_layer, 'layer': layer,  'channel': channel, 'i': i, 'j': j, 'pruned': False}}})
                        if unpruned_angle_iter == 0:
                            unpruned_angle = np.array([angle])
                        else:
                            unpruned_angle = np.concatenate([unpruned_angle, np.array([angle])], axis = 0)
                        unpruned_angle_iter = unpruned_angle_iter + 1
                if unpruned_angle_iter != 0:
                    if iter == 0:
                        total_angle = unpruned_angle
                    else:
                        total_angle = np.concatenate([total_angle, unpruned_angle], axis = 0)
                    iter = iter + 1
                
    # Sort the angle
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        total_num += reduce(lambda x, y: x*y, np.shape(mask))

    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            past_layer = dict_[str(angle)][index]['past_layer']
            layer = dict_[str(angle)][index]['layer']
            channel = dict_[str(angle)][index]['channel']
            depth_i = dict_[str(angle)][index]['i']
            depth_j = dict_[str(angle)][index]['j']
            current_weights_mask = all_mask[layer]
            
            if not bool(weights_to_be_pruned.get(layer)):
                is_depth_i_appear = False
                is_depth_j_appear = False
                weights_to_be_pruned.update({layer: {str(channel): {str(depth_i): 1, str(depth_j): 1}}})
            else:
                if not bool(weights_to_be_pruned[layer].get(str(channel))):
                    weights_to_be_pruned[layer].update({str(channel): {str(depth_i): 1, str(depth_j): 1}})
                else:
                    is_depth_i_appear = bool(weights_to_be_pruned[layer][str(channel)].get(str(depth_i)))
                    is_depth_j_appear = bool(weights_to_be_pruned[layer][str(channel)].get(str(depth_j)))
                    if not is_depth_i_appear:
                        weights_to_be_pruned[layer][str(channel)].update({str(depth_i): 1})
                    else:
                        weights_to_be_pruned[layer][str(channel)].update({str(depth_i): weights_to_be_pruned[layer][str(channel)][str(depth_i)]+1})
                    if not is_depth_j_appear:
                        weights_to_be_pruned[layer][str(channel)].update({str(depth_j): 1})
                    else:
                        weights_to_be_pruned[layer][str(channel)].update({str(depth_j): weights_to_be_pruned[layer][str(channel)][str(depth_j)]+1})
            # Estimate the pruned parameter number            
            if not is_depth_i_appear or not is_depth_j_appear:    
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, channel]))
                
        if pruned_num >= total_num * pruning_propotion:
            break 

    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel', 'Computation'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            #print(dict[str(angle)].keys())        
            if dict[str(angle)][index]['pruned']:
                continue
            # past layer
            #if bool(dict[str(angle)][index].get('past_layer')):
            past_layer = dict[str(angle)][index]['past_layer']
            past_weights_mask  = all_mask[past_layer]    
            past_weights       = all_weights[past_layer]
            past_outputs_shape = all_outputs_shape[past_layer]
            # current layer
            layer = dict[str(angle)][index]['layer']
            current_weights_mask  = all_mask[layer]
            current_weights       = all_weights[layer]
            current_outputs_shape = all_outputs_shape[layer]
            
            channel = dict[str(angle)][index]['channel']
            depth_i = dict[str(angle)][index]['i']
            depth_j = dict[str(angle)][index]['j']
            is_depth_i_pruned = pruned_dict[layer][str(channel)][str(depth_i)]
            is_depth_j_pruned = pruned_dict[layer][str(channel)][str(depth_j)]    

            if not is_depth_i_pruned and not is_depth_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                # Choose which depth to be pruned
                if weights_to_be_pruned[layer][str(channel)][str(depth_i)] < weights_to_be_pruned[layer][str(channel)][str(depth_j)]:
                    pruned_depth = depth_i
                elif weights_to_be_pruned[layer][str(channel)][str(depth_i)] > weights_to_be_pruned[layer][str(channel)][str(depth_j)]:
                    pruned_depth = depth_j
                else:
                    # Magnitude
                    sum_of_current_layer_depth_i = np.sum(np.abs(current_weights[:, :, depth_i, channel] * current_weights_mask[:, :, depth_i, channel]))
                    sum_of_current_layer_depth_j = np.sum(np.abs(current_weights[:, :, depth_j, channel] * current_weights_mask[:, :, depth_j, channel]))
                    sum_of_pruned_depth_i = sum_of_current_layer_depth_i
                    sum_of_pruned_depth_j = sum_of_current_layer_depth_j
                    if sum_of_pruned_depth_i <= sum_of_pruned_depth_j:
                        pruned_depth = depth_i
                    else:
                        pruned_depth = depth_j
                
                # remove the past pruned mask (for computing computation)
                unpruned_past_mask = past_weights_mask
                past_channel_num = np.shape(past_weights_mask)[3]
                past_depth = np.shape(past_weights_mask)[2]
                past_pruned_channels = []
                past_pruned_depth = []
                # (channel)
                for i in range(past_channel_num):
                    if np.sum(past_weights_mask[:, :, :, i]) == 0:
                        past_pruned_channels = np.append(past_pruned_channels, [i])
                unpruned_past_mask = np.delete(unpruned_past_mask, past_pruned_channels, axis = 3)
                # (depth)
                for i in range(past_depth):
                    if np.sum(past_weights_mask[:, :, i, :]) == 0:
                        past_pruned_depth = np.append(past_pruned_depth, [i])
                unpruned_past_mask = np.delete(unpruned_past_mask, past_pruned_depth, axis = 2)
                
                # remove the current pruned mask (for computing computation)
                unpruned_current_mask = current_weights_mask
                current_channel_num = np.shape(current_weights_mask)[3]
                current_depth = np.shape(current_weights_mask)[2]
                current_pruned_channels = []
                current_pruned_depth = []
                # (channel)
                for i in range(current_channel_num):
                    if np.sum(current_weights_mask[:, :, :, i]) == 0:
                        current_pruned_channels = np.append(current_pruned_channels, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_channels, axis = 3)
                # (depth)
                for i in range(current_depth):
                    if np.sum(current_weights_mask[:, :, i, :]) == 0:
                        current_pruned_depth = np.append(current_pruned_depth, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_depth, axis = 2)
                
                # Assign Zero to mask
                tmp = {}
                computation = 0
                # update pruned_dict
                pruned_dict[layer][str(channel)].update({str(pruned_depth): True})
                # Set current layer depth to zero
                current_weights_mask[:, :, pruned_depth, channel] = 0.
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, pruned_depth, channel]))
                # Check if there is all weights are pruned in pruned depth
                if np.mean(current_weights_mask[:, :, pruned_depth, :]) == 0:
                    if np.mean(past_weights_mask[:, :, :, pruned_depth]) != 0:
                        past_weights_mask[:, :, :, pruned_depth] = 0
                        pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, pruned_depth]))
                        # computation
                        past_weights_shape = np.shape(unpruned_past_mask)
                        computation = computation + past_weights_shape[0] * past_weights_shape[1] * past_weights_shape[2] * past_outputs_shape[2] * past_outputs_shape[3]
                        current_weights_shape = np.shape(unpruned_current_mask)
                        computation = computation + current_weights_shape[0] * current_weights_shape[1] * current_weights_shape[3] * current_outputs_shape[2] * current_outputs_shape[3]
                        tmp.update({past_layer: {'channel': pruned_depth}})
                        tmp.update({layer: {'depth': pruned_depth}})
                        tmp.update({'computation': computation})
                        pruned_weights_info.append(tmp)
                        
                t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'depth' + str(pruned_depth), computation])                
                #print("\r{} / {}	{}	{}	{} " .format(pruned_num, total_num, angle, layer, 'depth' + str(pruned_depth)), end = "")
                ##print("{} / {}	{}	{}	{} " .format(pruned_num, total_num, angle, layer, 'depth' + str(pruned_depth)))
                break
    #print(t)
    
    # Store all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))

    for layer_iter in range(1, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    print("Prune Over!")
    
    return pruned_weights_info
 
def denseNet_filter_prune_by_angle(
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]

    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    all_outputs_shape = {}
    all_stride = {}
    all_parents = {}
    all_children = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})
        if bool(prune_info_dict[layer].get('parents')):
            all_parents.update({layer: prune_info_dict[layer]['parents']})
            #print("{} -> {}" .format(layer, prune_info_dict[layer]['parents']))
        if bool(prune_info_dict[layer].get('children')):
            all_children.update({layer: prune_info_dict[layer]['children']})
            #print("{} -> {}" .format(layer, prune_info_dict[layer]['children']))
    
    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})
        
    # Build the dictionary for angle
    dict = {}
    iter = 0
    for layer_iter in range(1, len(sorted_layer)):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        weights = np.transpose(weights, (0,1,3,2))
        mask = all_mask[layer]
        # remove the pruned channel
        channel_num = np.shape(weights)[2]
        pruned_channels = []
        for i in range(channel_num):
            if pruned_dict[layer][str(i)]:
                pruned_channels = np.append(pruned_channels, [i])
        weights = np.delete(weights, pruned_channels, axis = 2)
        # calculate angle
        if weights.size != 0:
            depth = np.shape(weights)[-1]
            for i in range(depth):
                for j in range(i+1, depth):
                    # If the depth i or j has been pruned, we don't calculate its angle to others
                    # By doing so, we will not prune the same kernel which has been pruned before.
                    is_depth_i_pruned = np.sum(mask[:, :, i, :]) == 0
                    is_depth_j_pruned = np.sum(mask[:, :, j, :]) == 0
                    if not is_depth_i_pruned and not is_depth_j_pruned:
                        x = weights[:, :, :, i]
                        y = weights[:, :, :, j]
                        x = np.reshape(x, [np.size(x)])
                        y = np.reshape(y, [np.size(y)])
                        
                        if sum(x*x) == 0 or sum(y*y) == 0:
                            angle = 90.0
                            print('{}, {}, {}' .format(layer, i, j))
                        else:
                            angle = compute_angle(x, y)
                            angle = abs(angle - 90)
                        
                        if bool(dict.get(str(angle))):
                            number = len(dict[str(angle)])
                            dict[str(angle)].update({str(number): {'layer': layer, 'i': i, 'j': j, 'pruned': False}})
                        else:
                            dict.update({str(angle): {'0': {'layer': layer, 'i': i, 'j': j, 'pruned': False}}})
                        
                        if iter == 0:
                            total_angle = np.array([angle])
                        else:
                            total_angle = np.concatenate([total_angle, np.array([angle])])
                        iter = iter + 1    
             
    # Sort the angle
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            # current layer
            layer = dict_[str(angle)][index]['layer']
            current_weights_mask = all_mask[layer]
            # depth
            depth_i = dict_[str(angle)][index]['i']
            depth_j = dict_[str(angle)][index]['j']

            if not bool(weights_to_be_pruned.get(layer)):
                is_depth_i_appear = False
                is_depth_j_appear = False
                weights_to_be_pruned.update({layer: {str(depth_i): 1, str(depth_j): 1}})
            else:
                is_depth_i_appear = bool(weights_to_be_pruned[layer].get(str(depth_i)))
                is_depth_j_appear = bool(weights_to_be_pruned[layer].get(str(depth_j)))
                if not is_depth_i_appear:
                    weights_to_be_pruned[layer].update({str(depth_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_i): weights_to_be_pruned[layer][str(depth_i)]+1})
                if not is_depth_j_appear:
                    weights_to_be_pruned[layer].update({str(depth_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_j): weights_to_be_pruned[layer][str(depth_j)]+1})
            if not is_depth_i_appear or not is_depth_j_appear:
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
                
                # Check the parent layer channel should be pruned or not
                if bool(all_parents.get(layer)):
                    # random choose depth i or j to be pruned
                    rand_index = random.randrange(0, 2)
                    # Find the head
                    if rand_index == 0:
                        pruned_depth = depth_i
                    else:
                        pruned_depth = depth_j
                    interval = pruned_depth             
                    for _, parent in enumerate(all_parents[layer]):
                        if pruned_depth >= all_parents[layer][parent] and (pruned_depth - all_parents[layer][parent]) <= interval:
                            interval = pruned_depth - all_parents[layer][parent]
                            head = parent
                            
                    # Check through all parents
                    is_head_channel_to_be_pruned = True
                    for _, child in enumerate(all_children[head]):
                        start_point = all_children[head][child]
                        position = start_point + interval
                        if np.sum(all_mask[child][:, :, position, :]) != 0:
                            is_head_channel_i_to_be_pruned = False
                            break
                if is_head_channel_i_to_be_pruned:
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head][:, :, :, interval]))
                    
        if pruned_num >= total_num * pruning_propotion:
            break    
      
    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel', 'Computation'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            if dict[str(angle)][index]['pruned']:
                continue
            # current layer
            layer = dict[str(angle)][index]['layer']
            current_weights_mask  = all_mask[layer]
            current_weights       = all_weights[layer]
            current_outputs_shape = all_outputs_shape[layer]
            # depth
            depth_i = dict[str(angle)][index]['i']
            depth_j = dict[str(angle)][index]['j']
            is_depth_i_pruned = np.sum(current_weights_mask[:, :, depth_i, :]) == 0
            is_depth_j_pruned = np.sum(current_weights_mask[:, :, depth_j, :]) == 0
 
            if not is_depth_i_pruned and not is_depth_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                if weights_to_be_pruned[layer][str(depth_i)] < weights_to_be_pruned[layer][str(depth_j)]:
                    pruned_depth = depth_i
                elif weights_to_be_pruned[layer][str(depth_i)] > weights_to_be_pruned[layer][str(depth_j)]:
                    pruned_depth = depth_j
                else:
                    # Magnitude
                    #if bool(dict[str(angle)][index].get('past_layer')):
                    #sum_of_past_layer_kernal_i = np.sum(np.abs(past_weights[:, :, :, depth_i] * past_weights_mask[:, :, :, depth_i]))
                    #sum_of_past_layer_kernel_j = np.sum(np.abs(past_weights[:, :, :, depth_j] * past_weights_mask[:, :, :, depth_j]))
                    #else:
                    #    sum_of_past_layer_kernal_i = 0
                    #    sum_of_past_layer_kernel_j = 0
                    sum_of_current_layer_depth_i = np.sum(np.abs(current_weights[:, :, depth_i, :] * current_weights_mask[:, :, depth_i, :]))
                    sum_of_current_layer_depth_j = np.sum(np.abs(current_weights[:, :, depth_j, :] * current_weights_mask[:, :, depth_j, :]))
                    sum_of_pruned_depth_i = sum_of_current_layer_depth_i #+ sum_of_past_layer_kernal_i
                    sum_of_pruned_depth_j = sum_of_current_layer_depth_j #+ sum_of_past_layer_kernel_j 
                    # Compare
                    if sum_of_pruned_depth_i <= sum_of_pruned_depth_j:
                        pruned_depth = depth_i
                    else:
                        pruned_depth = depth_j
                
                # remove the current pruned mask (for computing computation)
                unpruned_current_mask = current_weights_mask
                current_channel_num = np.shape(current_weights_mask)[3]
                current_depth = np.shape(current_weights_mask)[2]
                current_pruned_channels = []
                current_pruned_depth = []
                # (channel)
                for i in range(current_channel_num):
                    if np.sum(current_weights_mask[:, :, :, i]) == 0:
                        current_pruned_channels = np.append(current_pruned_channels, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_channels, axis = 3)
                # (depth)
                for i in range(current_depth):
                    if np.sum(current_weights_mask[:, :, i, :]) == 0:
                        current_pruned_depth = np.append(current_pruned_depth, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_depth, axis = 2)
                
                ## Assign Zero to mask
                tmp = {}
                computation = 0
                # Current
                current_weights_mask[:, :, pruned_depth, :] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
                current_weights_shape = np.shape(unpruned_current_mask)
                computation = computation + current_weights_shape[0] * current_weights_shape[1] * current_weights_shape[3] * current_outputs_shape[2] * current_outputs_shape[3]
                tmp.update({layer: {'depth': pruned_depth}})
                # Past 
                # Check the parent layer channel should be pruned or not
                if bool(all_parents.get(layer)):
                    interval = pruned_depth             
                    for _, parent in enumerate(all_parents[layer]):
                        if pruned_depth >= all_parents[layer][parent] and (pruned_depth - all_parents[layer][parent]) <= interval:
                            interval = pruned_depth - all_parents[layer][parent]
                            head = parent
                            
                    # Check through all parents
                    is_head_channel_to_be_pruned = True
                    for _, child in enumerate(all_children[head]):
                        start_point = all_children[head][child]
                        position = start_point + interval
                        if np.sum(all_mask[child][:, :, position, :]) != 0:
                            is_head_channel_i_to_be_pruned = False
                            break
                if is_head_channel_i_to_be_pruned:
                    pruned_dict[head].update({str(pruned_depth): True})
                    head_weights_mask  = all_mask[head]
                    head_weights_mask[:, :, :, interval] = 0
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head][:, :, :, interval]))
                    # remove the past pruned mask (for computing computation)
                    unpruned_head_mask = head_weights_mask
                    head_channel_num = np.shape(head_weights_mask)[3]
                    head_depth = np.shape(head_weights_mask)[2]
                    head_pruned_channels = []
                    head_pruned_depth = []
                    # (channel)
                    for i in range(head_channel_num):
                        if np.sum(head_weights_mask[:, :, :, i]) == 0:
                            head_pruned_channels = np.append(head_pruned_channels, [i])
                    unpruned_head_mask = np.delete(unpruned_head_mask, head_pruned_channels, axis = 3)
                    # (depth)
                    for i in range(head_depth):
                        if np.sum(head_weights_mask[:, :, i, :]) == 0:
                            head_pruned_depth = np.append(head_pruned_depth, [i])
                    unpruned_head_mask = np.delete(unpruned_head_mask, head_pruned_depth, axis = 2)
                    
                    head_weights_shape = np.shape(unpruned_head_mask)
                    computation = computation + head_weights_shape[0] * head_weights_shape[1] * head_weights_shape[2] * head_weights_shape[2] * head_weights_shape[3]
                    tmp.update({head: {'channel': pruned_depth}})
                tmp.update({'computation': computation})
                pruned_weights_info.append(tmp)
                t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'depth' + str(pruned_depth), computation])
            break
    print(t)
    
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        #sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    """
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    """
    # See the parameter shape change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        mask = sess.run(mask_tensor)
        # remove the pruned part
        unpruned_mask = mask
        # (channel)
        channel_num = np.shape(mask)[3]
        pruned_channels = []
        for channel in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, channel]) == 0:
                pruned_channels = np.append(pruned_channels, [channel])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # (depth)
        depths = np.shape(unpruned_mask)[2]
        pruned_depths = []
        for depth in range(depths):
            if np.sum(unpruned_mask[:, :, depth, :]) == 0:
                pruned_depths = np.append(pruned_depths, [depth])
        unpruned_mask = np.delete(unpruned_mask, pruned_depths, axis = 2)
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' 
        .format(layer, np.shape(mask), np.shape(unpruned_mask)))
    print("Prune Over!")
    
    return pruned_weights_info

def denseNet_filter_prune_by_angleII(
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]

    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    all_outputs_shape = {}
    all_stride = {}
    all_parents = {}
    all_children = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})
        if bool(prune_info_dict[layer].get('parents')):
            all_parents.update({layer: prune_info_dict[layer]['parents']})
            #print("{} -> {}" .format(layer, prune_info_dict[layer]['parents']))
        if bool(prune_info_dict[layer].get('children')):
            all_children.update({layer: prune_info_dict[layer]['children']})
            #print("{} -> {}" .format(layer, prune_info_dict[layer]['children']))
    
    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})
        
    # Build the dictionary for angle
    dict = {}
    iter = 0
    for layer_iter in range(1, len(sorted_layer)-1):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        mask = all_mask[layer]
        # calculate angle
        if weights.size != 0:
            channel = np.shape(weights)[3]
            for i in range(channel):
                for j in range(i+1, channel):
                    # If the channel i or j has been pruned, we don't calculate its angle to others
                    # By doing so, we will not prune the same kernel which has been pruned before.
                    is_channel_i_pruned = np.sum(mask[:, :, :, i]) == 0
                    is_channel_j_pruned = np.sum(mask[:, :, :, j]) == 0
                    if not is_channel_i_pruned and not is_channel_j_pruned:
                        x = weights[:, :, :, i]
                        y = weights[:, :, :, j]
                        x = np.reshape(x, [np.size(x)])
                        y = np.reshape(y, [np.size(y)])
                        
                        if sum(x*x) == 0 or sum(y*y) == 0:
                            angle = 90.0
                            print('{}, {}, {}' .format(layer, i, j))
                        else:
                            angle = compute_angle(x, y)
                            angle = abs(angle - 90)
                        
                        if bool(dict.get(str(angle))):
                            number = len(dict[str(angle)])
                            dict[str(angle)].update({str(number): {'layer': layer, 'i': i, 'j': j, 'pruned': False}})
                        else:
                            dict.update({str(angle): {'0': {'layer': layer, 'i': i, 'j': j, 'pruned': False}}})
                        
                        if iter == 0:
                            total_angle = np.array([angle])
                        else:
                            total_angle = np.concatenate([total_angle, np.array([angle])])
                        iter = iter + 1
             
    # Sort the angle
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            # current layer
            layer = dict_[str(angle)][index]['layer']
            current_weights_mask = all_mask[layer]
            # children
            if bool(all_children.get(layer)):
                children = all_children[layer]
            # channel
            channel_i = dict_[str(angle)][index]['i']
            channel_j = dict_[str(angle)][index]['j']

            if not bool(weights_to_be_pruned.get(layer)):
                is_channel_i_appear = False
                is_channel_j_appear = False
                weights_to_be_pruned.update({layer: {str(channel_i): 1, str(channel_j): 1}})
            else:
                is_channel_i_appear = bool(weights_to_be_pruned[layer].get(str(channel_i)))
                is_channel_j_appear = bool(weights_to_be_pruned[layer].get(str(channel_j)))
                if not is_channel_i_appear:
                    weights_to_be_pruned[layer].update({str(channel_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(channel_i): weights_to_be_pruned[layer][str(channel_i)]+1})
                if not is_channel_j_appear:
                    weights_to_be_pruned[layer].update({str(channel_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(channel_j): weights_to_be_pruned[layer][str(channel_j)]+1})
            if not is_channel_i_appear or not is_channel_j_appear:
                # Current layer
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, channel_i]))
                # Children layer
                if bool(all_children.get(layer)):
                    # random choose depth i or j to be pruned
                    rand_index = random.randrange(0, 2)
                    if rand_index == 0:
                        pruned_channel = channel_i
                    elif rand_index == 1:
                        pruned_channel = channel_j
                    # each child
                    for _, child in enumerate(children):
                        interval = pruned_channel
                        start = children[child]
                        pruned_depth = start + interval
                        pruned_num = pruned_num + np.sum(all_mask[child][:, :, pruned_depth, :])
                    
        if pruned_num >= total_num * pruning_propotion:
            break    
    
    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel', 'Computation'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            if dict[str(angle)][index]['pruned']:
                continue
            # current layer
            layer = dict[str(angle)][index]['layer']
            current_weights_mask  = all_mask[layer]
            current_weights       = all_weights[layer]
            current_outputs_shape = all_outputs_shape[layer]
            # channel
            channel_i = dict[str(angle)][index]['i']
            channel_j = dict[str(angle)][index]['j']
            is_channel_i_pruned = np.sum(current_weights_mask[:, :, :, channel_i]) == 0
            is_channel_j_pruned = np.sum(current_weights_mask[:, :, :, channel_j]) == 0
 
            if not is_channel_i_pruned and not is_channel_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                if weights_to_be_pruned[layer][str(channel_i)] < weights_to_be_pruned[layer][str(channel_j)]:
                    pruned_channel = channel_i
                elif weights_to_be_pruned[layer][str(channel_i)] > weights_to_be_pruned[layer][str(channel_j)]:
                    pruned_channel = channel_j
                else:
                    # Magnitude
                    sum_of_children_i = 0
                    sum_of_children_j = 0
                    if bool(all_children.get(layer)):
                        children = all_children[layer]
                        for _, child in enumerate(children):
                            child_weight = all_weights[child]
                            child_mask = all_mask[child]
                            sum_of_children_i = sum_of_children_i + np.sum(np.abs(child_weight[:, :, channel_i, :] * child_mask[:, :, channel_i, :]))
                            sum_of_children_j = sum_of_children_j + np.sum(np.abs(child_weight[:, :, channel_j, :] * child_mask[:, :, channel_j, :]))
                    sum_of_current_layer_channel_i = np.sum(np.abs(current_weights[:, :, :, channel_i] * current_weights_mask[:, :, :, channel_i]))
                    sum_of_current_layer_channel_j = np.sum(np.abs(current_weights[:, :, :, channel_j] * current_weights_mask[:, :, :, channel_j]))
                    sum_of_pruned_channel_i = sum_of_current_layer_channel_i + sum_of_children_i
                    sum_of_pruned_channel_j = sum_of_current_layer_channel_j + sum_of_children_j 
                    # Compare
                    if sum_of_pruned_channel_i <= sum_of_pruned_channel_j:
                        pruned_channel = channel_i
                    else:
                        pruned_channel = channel_j
                

                ## Assign Zero to mask
                tmp = {}
                computation = 0
                # Current
                current_weights_mask[:, :, :, pruned_channel] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, pruned_channel]))
                tmp.update({layer: {'channel': pruned_channel}})
                # (computation)
                # remove the current pruned mask (for computing computation)
                unpruned_current_mask = current_weights_mask
                current_channel_num = np.shape(current_weights_mask)[3]
                current_depth = np.shape(current_weights_mask)[2]
                current_pruned_channels = []
                current_pruned_depth = []
                # (channel)
                for i in range(current_channel_num):
                    if np.sum(current_weights_mask[:, :, :, i]) == 0:
                        current_pruned_channels = np.append(current_pruned_channels, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_channels, axis = 3)
                # (depth)
                for i in range(current_depth):
                    if np.sum(current_weights_mask[:, :, i, :]) == 0:
                        current_pruned_depth = np.append(current_pruned_depth, [i])
                #
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_depth, axis = 2)
                current_weights_shape = np.shape(unpruned_current_mask)
                computation = computation + current_weights_shape[0] * current_weights_shape[1] * current_weights_shape[2] * current_outputs_shape[2] * current_outputs_shape[3]
                
                # Children
                if bool(all_children.get(layer)):
                    children = all_children[layer]
                    for _, child in enumerate(children):
                        interval = pruned_channel
                        start = children[child]
                        pruned_depth = start + interval
                        all_mask[child][:, :, pruned_depth, :] = 0
                        pruned_num = pruned_num + np.sum(all_mask[child][:, :, pruned_depth, :])
                        tmp.update({child: {'depth': pruned_depth}})
                        # (computation)
                        # remove the pruned mask
                        unpruned_child_mask = all_mask[child]
                        child_channel_num = np.shape(unpruned_child_mask)[3]
                        child_depth = np.shape(unpruned_child_mask)[2]
                        child_pruned_channels = []
                        child_pruned_depth = []
                        # (channel)
                        for i in range(child_channel_num):
                            if np.sum(unpruned_child_mask[:, :, :, i]) == 0:
                                child_pruned_channels = np.append(child_pruned_channels, [i])
                        unpruned_child_mask = np.delete(unpruned_child_mask, child_pruned_channels, axis = 3)
                        # (depth)
                        for i in range(child_depth):
                            if np.sum(unpruned_child_mask[:, :, i, :]) == 0:
                                child_pruned_depth = np.append(child_pruned_depth, [i])
                        unpruned_child_mask = np.delete(unpruned_child_mask, child_pruned_depth, axis = 2)
                        # 
                        child_weights_shape = np.shape(unpruned_child_mask)
                        computation = computation + child_weights_shape[0] * child_weights_shape[1] * child_weights_shape[3] * child_weights_shape[2] * child_weights_shape[3]
                    
                tmp.update({'computation': computation})        
                pruned_weights_info.append(tmp)
                t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'channel' + str(pruned_channel), computation])
            break
    print(t)
    
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        #sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    # See the parameter shape change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        mask = sess.run(mask_tensor)
        # remove the pruned part
        unpruned_mask = mask
        # (channel)
        channel_num = np.shape(mask)[3]
        pruned_channels = []
        for channel in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, channel]) == 0:
                pruned_channels = np.append(pruned_channels, [channel])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # (depth)
        depths = np.shape(unpruned_mask)[2]
        pruned_depths = []
        for depth in range(depths):
            if np.sum(unpruned_mask[:, :, depth, :]) == 0:
                pruned_depths = np.append(pruned_depths, [depth])
        unpruned_mask = np.delete(unpruned_mask, pruned_depths, axis = 2)
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' 
        .format(layer, np.shape(mask), np.shape(unpruned_mask)))
    
    print("Prune Over!")
    
    return pruned_weights_info
   
def denseNet_filter_prune_by_angleII_with_skip(
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    skip_layer,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]

    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    all_outputs_shape = {}
    all_stride = {}
    all_parents = {}
    all_children = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})
        if bool(prune_info_dict[layer].get('parents')):
            all_parents.update({layer: prune_info_dict[layer]['parents']})
            #print("{} -> {}" .format(layer, prune_info_dict[layer]['parents']))
        if bool(prune_info_dict[layer].get('children')):
            all_children.update({layer: prune_info_dict[layer]['children']})
            #print("{} -> {}" .format(layer, prune_info_dict[layer]['children']))
    
    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})
 
    # Build the dictionary for angle
    dict = {}
    iter = 0
    for layer_iter in range(1, len(sorted_layer)-1):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        mask = all_mask[layer]
        # Angle of layer in skip layer will not be calculated. 
        if not layer in skip_layer:
            # calculate angle
            if weights.size != 0:
                channel = np.shape(weights)[3]
                for i in range(channel):
                    for j in range(i+1, channel):
                        # If the channel i or j has been pruned, we don't calculate its angle to others
                        # By doing so, we will not prune the same kernel which has been pruned before.
                        is_channel_i_pruned = np.sum(mask[:, :, :, i]) == 0
                        is_channel_j_pruned = np.sum(mask[:, :, :, j]) == 0
                        if not is_channel_i_pruned and not is_channel_j_pruned:
                            x = weights[:, :, :, i]
                            y = weights[:, :, :, j]
                            x = np.reshape(x, [np.size(x)])
                            y = np.reshape(y, [np.size(y)])
                            
                            if sum(x*x) == 0 or sum(y*y) == 0:
                                angle = 90.0
                                print('{}, {}, {}' .format(layer, i, j))
                            else:
                                angle = compute_angle(x, y)
                                angle = abs(angle - 90)
                            
                            if bool(dict.get(str(angle))):
                                number = len(dict[str(angle)])
                                dict[str(angle)].update({str(number): {'layer': layer, 'i': i, 'j': j, 'pruned': False}})
                            else:
                                dict.update({str(angle): {'0': {'layer': layer, 'i': i, 'j': j, 'pruned': False}}})
                            
                            if iter == 0:
                                total_angle = np.array([angle])
                            else:
                                total_angle = np.concatenate([total_angle, np.array([angle])])
                            iter = iter + 1
             
    # Sort the angle
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            # current layer
            layer = dict_[str(angle)][index]['layer']
            current_weights_mask = all_mask[layer]
            # children
            if bool(all_children.get(layer)):
                children = all_children[layer]
            # channel
            channel_i = dict_[str(angle)][index]['i']
            channel_j = dict_[str(angle)][index]['j']

            if not bool(weights_to_be_pruned.get(layer)):
                is_channel_i_appear = False
                is_channel_j_appear = False
                weights_to_be_pruned.update({layer: {str(channel_i): 1, str(channel_j): 1}})
            else:
                is_channel_i_appear = bool(weights_to_be_pruned[layer].get(str(channel_i)))
                is_channel_j_appear = bool(weights_to_be_pruned[layer].get(str(channel_j)))
                if not is_channel_i_appear:
                    weights_to_be_pruned[layer].update({str(channel_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(channel_i): weights_to_be_pruned[layer][str(channel_i)]+1})
                if not is_channel_j_appear:
                    weights_to_be_pruned[layer].update({str(channel_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(channel_j): weights_to_be_pruned[layer][str(channel_j)]+1})
            if not is_channel_i_appear or not is_channel_j_appear:
                # Current layer
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, channel_i]))
                # Children layer
                if bool(all_children.get(layer)):
                    # random choose depth i or j to be pruned
                    rand_index = random.randrange(0, 2)
                    if rand_index == 0:
                        pruned_channel = channel_i
                    elif rand_index == 1:
                        pruned_channel = channel_j
                    # each child
                    for _, child in enumerate(children):
                        interval = pruned_channel
                        start = children[child]
                        pruned_depth = start + interval
                        pruned_num = pruned_num + np.sum(all_mask[child][:, :, pruned_depth, :])
                    
        if pruned_num >= total_num * pruning_propotion:
            break    
    
    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel', 'Computation'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            if dict[str(angle)][index]['pruned']:
                continue
            # current layer
            layer = dict[str(angle)][index]['layer']
            current_weights_mask  = all_mask[layer]
            current_weights       = all_weights[layer]
            current_outputs_shape = all_outputs_shape[layer]
            # channel
            channel_i = dict[str(angle)][index]['i']
            channel_j = dict[str(angle)][index]['j']
            is_channel_i_pruned = np.sum(current_weights_mask[:, :, :, channel_i]) == 0
            is_channel_j_pruned = np.sum(current_weights_mask[:, :, :, channel_j]) == 0
 
            if not is_channel_i_pruned and not is_channel_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                if weights_to_be_pruned[layer][str(channel_i)] < weights_to_be_pruned[layer][str(channel_j)]:
                    pruned_channel = channel_i
                elif weights_to_be_pruned[layer][str(channel_i)] > weights_to_be_pruned[layer][str(channel_j)]:
                    pruned_channel = channel_j
                else:
                    # Magnitude
                    sum_of_children_i = 0
                    sum_of_children_j = 0
                    if bool(all_children.get(layer)):
                        children = all_children[layer]
                        for _, child in enumerate(children):
                            child_weight = all_weights[child]
                            child_mask = all_mask[child]
                            sum_of_children_i = sum_of_children_i + np.sum(np.abs(child_weight[:, :, channel_i, :] * child_mask[:, :, channel_i, :]))
                            sum_of_children_j = sum_of_children_j + np.sum(np.abs(child_weight[:, :, channel_j, :] * child_mask[:, :, channel_j, :]))
                    sum_of_current_layer_channel_i = np.sum(np.abs(current_weights[:, :, :, channel_i] * current_weights_mask[:, :, :, channel_i]))
                    sum_of_current_layer_channel_j = np.sum(np.abs(current_weights[:, :, :, channel_j] * current_weights_mask[:, :, :, channel_j]))
                    sum_of_pruned_channel_i = sum_of_current_layer_channel_i + sum_of_children_i
                    sum_of_pruned_channel_j = sum_of_current_layer_channel_j + sum_of_children_j 
                    # Compare
                    if sum_of_pruned_channel_i <= sum_of_pruned_channel_j:
                        pruned_channel = channel_i
                    else:
                        pruned_channel = channel_j
                

                ## Assign Zero to mask
                tmp = {}
                computation = 0
                # Current
                current_weights_mask[:, :, :, pruned_channel] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, pruned_channel]))
                tmp.update({layer: {'channel': pruned_channel}})
                # (computation)
                # remove the current pruned mask (for computing computation)
                unpruned_current_mask = current_weights_mask
                current_channel_num = np.shape(current_weights_mask)[3]
                current_depth = np.shape(current_weights_mask)[2]
                current_pruned_channels = []
                current_pruned_depth = []
                # (channel)
                for i in range(current_channel_num):
                    if np.sum(current_weights_mask[:, :, :, i]) == 0:
                        current_pruned_channels = np.append(current_pruned_channels, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_channels, axis = 3)
                # (depth)
                for i in range(current_depth):
                    if np.sum(current_weights_mask[:, :, i, :]) == 0:
                        current_pruned_depth = np.append(current_pruned_depth, [i])
                #
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_depth, axis = 2)
                current_weights_shape = np.shape(unpruned_current_mask)
                computation = computation + current_weights_shape[0] * current_weights_shape[1] * current_weights_shape[2] * current_outputs_shape[2] * current_outputs_shape[3]
                
                # Children
                if bool(all_children.get(layer)):
                    children = all_children[layer]
                    for _, child in enumerate(children):
                        interval = pruned_channel
                        start = children[child]
                        pruned_depth = start + interval
                        all_mask[child][:, :, pruned_depth, :] = 0
                        pruned_num = pruned_num + np.sum(all_mask[child][:, :, pruned_depth, :])
                        tmp.update({child: {'depth': pruned_depth}})
                        # (computation)
                        # remove the pruned mask
                        unpruned_child_mask = all_mask[child]
                        child_channel_num = np.shape(unpruned_child_mask)[3]
                        child_depth = np.shape(unpruned_child_mask)[2]
                        child_pruned_channels = []
                        child_pruned_depth = []
                        # (channel)
                        for i in range(child_channel_num):
                            if np.sum(unpruned_child_mask[:, :, :, i]) == 0:
                                child_pruned_channels = np.append(child_pruned_channels, [i])
                        unpruned_child_mask = np.delete(unpruned_child_mask, child_pruned_channels, axis = 3)
                        # (depth)
                        for i in range(child_depth):
                            if np.sum(unpruned_child_mask[:, :, i, :]) == 0:
                                child_pruned_depth = np.append(child_pruned_depth, [i])
                        unpruned_child_mask = np.delete(unpruned_child_mask, child_pruned_depth, axis = 2)
                        # 
                        child_weights_shape = np.shape(unpruned_child_mask)
                        computation = computation + child_weights_shape[0] * child_weights_shape[1] * child_weights_shape[3] * child_weights_shape[2] * child_weights_shape[3]
                    
                tmp.update({'computation': computation})        
                pruned_weights_info.append(tmp)
                t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'channel' + str(pruned_channel), computation])
            break
    print(t)
    
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        #sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    # See the parameter shape change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        mask = sess.run(mask_tensor)
        # remove the pruned part
        unpruned_mask = mask
        # (channel)
        channel_num = np.shape(mask)[3]
        pruned_channels = []
        for channel in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, channel]) == 0:
                pruned_channels = np.append(pruned_channels, [channel])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # (depth)
        depths = np.shape(unpruned_mask)[2]
        pruned_depths = []
        for depth in range(depths):
            if np.sum(unpruned_mask[:, :, depth, :]) == 0:
                pruned_depths = np.append(pruned_depths, [depth])
        unpruned_mask = np.delete(unpruned_mask, pruned_depths, axis = 2)
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' 
        .format(layer, np.shape(mask), np.shape(unpruned_mask)))
    
    print("Prune Over!")
    
    return pruned_weights_info

def denseNet_filter_prune_by_angleII_with_skip_with_penalty(
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    skip_layer,
    sess
    ):
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]
    
    
    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    all_outputs_shape = {}
    all_stride = {}
    all_parents = {}
    all_children = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})
        if bool(prune_info_dict[layer].get('parents')):
            all_parents.update({layer: prune_info_dict[layer]['parents']})
            #print("{} -> {}" .format(layer, prune_info_dict[layer]['parents']))
        if bool(prune_info_dict[layer].get('children')):
            all_children.update({layer: prune_info_dict[layer]['children']})
            #print("{} -> {}" .format(layer, prune_info_dict[layer]['children']))
            
    # Record the original weights parameter size
    original_weights_size = {}
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        original_weights_size.update({layer: np.sum(all_mask[layer])})
    
    # Build Pruned dict
    pruned_dict = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})
 
    # Calculate Penalty
    angle_dict = {}
    all_angles = {}
    for layer_iter in range(1, len(sorted_layer)-1):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        mask = all_mask[layer]
        # Angle of layer in skip layer will not be calculated. 
        if not layer in skip_layer:
            # calculate angle
            channel = np.shape(weights)[3]
            angles = []
            for i in range(channel):
                for j in range(i+1, channel):
                    # If the channel i or j has been pruned, we don't calculate its angle to others
                    # By doing so, we will not prune the same kernel which has been pruned before.
                    is_channel_i_pruned = np.sum(mask[:, :, :, i]) == 0
                    is_channel_j_pruned = np.sum(mask[:, :, :, j]) == 0
                    if not is_channel_i_pruned and not is_channel_j_pruned:
                        x = weights[:, :, :, i]
                        y = weights[:, :, :, j]
                        x = np.reshape(x, [np.size(x)])
                        y = np.reshape(y, [np.size(y)])
                        # angle
                        if sum(x*x) == 0 or sum(y*y) == 0:
                            angle = 90.0
                            print('{}, {}, {}' .format(layer, i, j))
                        else:
                            angle = compute_angle(x, y)
                            angle = abs(angle - 90)
                        # angle_dict
                        if bool(angle_dict.get(str(layer))):
                            angle_dict[layer].update({str(i): {str(j): angle}})
                        else:
                            angle_dict.update({layer: {str(i): {str(j): angle}}})
                        # angles
                        if not sum(x*x) == 0 or sum(y*y) == 0:
                            angles = np.append(angles, compute_angle(x, y))
            # Angles Per Layer
            all_angles.update({layer: angles})
        
    all_penalty = {}
    penalty_regression = 'stdev'
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        if not layer in skip_layer:
            if penalty_regression == 'stdev':
                angles = all_angles[layer]
                stdev = np.std(angles)
                mean = np.mean(angles)
                max = np.max(angles)
                min = np.min(angles)
                print("%8s -> \033[1;32mstdev\033[0m: %10f, \033[1;32mmean\033[0m: %10f" %(layer, stdev, mean) )
                all_penalty.update({layer: stdev})
    
    # Build the dictionary for angle
    dict = {}
    iter = 0
    for layer_iter in range(1, len(sorted_layer)-1):
        # current layer
        layer = sorted_layer[layer_iter]
        weights = all_weights[layer]
        mask = all_mask[layer]
        # Angle of layer in skip layer will not be calculated. 
        if not layer in skip_layer:
            # calculate angle
            penalty = all_penalty[layer]
            for i in angle_dict[layer]:
                for j in angle_dict[layer][i]:
                    angle = angle_dict[layer][i][j]
                    angle = angle * penalty

                    # Dict
                    if bool(dict.get(str(angle))):
                        number = len(dict[str(angle)])
                        dict[str(angle)].update({str(number): {'layer': layer, 'i': int(i), 'j': int(j), 'pruned': False}})
                    else:
                        dict.update({str(angle): {'0': {'layer': layer, 'i': int(i), 'j': int(j), 'pruned': False}}})
                    
                    # Total Angles
                    if iter == 0:
                        total_angle = np.array([angle])
                    else:
                        total_angle = np.concatenate([total_angle, np.array([angle])])
                    iter = iter + 1 
    
    # Sort the angle with penalty
    sorted_angle = np.sort(total_angle)[::-1]

    # Calculate the total parameters number
    total_num = 0
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        total_num += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # Get the weights to be pruned
    dict_ = copy.deepcopy(dict)
    weights_to_be_pruned = {}
    pruned_num = 0
    pruned_angle_num = 0
    for _, angle in enumerate(sorted_angle):
        pruned_angle_num = pruned_angle_num + 1
        for _, index in enumerate(dict_[str(angle)].keys()):
            if dict_[str(angle)][index]['pruned']:
                continue
            dict_[str(angle)][index].update({'pruned': True})
            # current layer
            layer = dict_[str(angle)][index]['layer']
            current_weights_mask = all_mask[layer]
            # children
            if bool(all_children.get(layer)):
                children = all_children[layer]
            # channel
            channel_i = dict_[str(angle)][index]['i']
            channel_j = dict_[str(angle)][index]['j']

            if not bool(weights_to_be_pruned.get(layer)):
                is_channel_i_appear = False
                is_channel_j_appear = False
                weights_to_be_pruned.update({layer: {str(channel_i): 1, str(channel_j): 1}})
            else:
                is_channel_i_appear = bool(weights_to_be_pruned[layer].get(str(channel_i)))
                is_channel_j_appear = bool(weights_to_be_pruned[layer].get(str(channel_j)))
                if not is_channel_i_appear:
                    weights_to_be_pruned[layer].update({str(channel_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(channel_i): weights_to_be_pruned[layer][str(channel_i)]+1})
                if not is_channel_j_appear:
                    weights_to_be_pruned[layer].update({str(channel_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(channel_j): weights_to_be_pruned[layer][str(channel_j)]+1})
            if not is_channel_i_appear or not is_channel_j_appear:
                # Current layer
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, channel_i]))
                # Children layer
                if bool(all_children.get(layer)):
                    # random choose depth i or j to be pruned
                    rand_index = random.randrange(0, 2)
                    if rand_index == 0:
                        pruned_channel = channel_i
                    elif rand_index == 1:
                        pruned_channel = channel_j
                    # each child
                    for _, child in enumerate(children):
                        interval = pruned_channel
                        start = children[child]
                        pruned_depth = start + interval
                        pruned_num = pruned_num + np.sum(all_mask[child][:, :, pruned_depth, :])
                    
        if pruned_num >= total_num * pruning_propotion:
            break    
    
    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Magnitude', 'layer', 'channel', 'Computation'])
    t.align = 'l'
    pruned_num = 0
    for iter in range(pruned_angle_num):
        angle = sorted_angle[iter]
        for _, index in enumerate(dict[str(angle)].keys()):
            if dict[str(angle)][index]['pruned']:
                continue
            # current layer
            layer = dict[str(angle)][index]['layer']
            current_weights_mask  = all_mask[layer]
            current_weights       = all_weights[layer]
            current_outputs_shape = all_outputs_shape[layer]
            # channel
            channel_i = dict[str(angle)][index]['i']
            channel_j = dict[str(angle)][index]['j']
            is_channel_i_pruned = np.sum(current_weights_mask[:, :, :, channel_i]) == 0
            is_channel_j_pruned = np.sum(current_weights_mask[:, :, :, channel_j]) == 0
 
            if not is_channel_i_pruned and not is_channel_j_pruned:
                dict[str(angle)][index].update({'pruned': True})
                if weights_to_be_pruned[layer][str(channel_i)] < weights_to_be_pruned[layer][str(channel_j)]:
                    pruned_channel = channel_i
                elif weights_to_be_pruned[layer][str(channel_i)] > weights_to_be_pruned[layer][str(channel_j)]:
                    pruned_channel = channel_j
                else:
                    # Magnitude
                    sum_of_children_i = 0
                    sum_of_children_j = 0
                    if bool(all_children.get(layer)):
                        children = all_children[layer]
                        for _, child in enumerate(children):
                            child_weight = all_weights[child]
                            child_mask = all_mask[child]
                            sum_of_children_i = sum_of_children_i + np.sum(np.abs(child_weight[:, :, channel_i, :] * child_mask[:, :, channel_i, :]))
                            sum_of_children_j = sum_of_children_j + np.sum(np.abs(child_weight[:, :, channel_j, :] * child_mask[:, :, channel_j, :]))
                    sum_of_current_layer_channel_i = np.sum(np.abs(current_weights[:, :, :, channel_i] * current_weights_mask[:, :, :, channel_i]))
                    sum_of_current_layer_channel_j = np.sum(np.abs(current_weights[:, :, :, channel_j] * current_weights_mask[:, :, :, channel_j]))
                    sum_of_pruned_channel_i = sum_of_current_layer_channel_i + sum_of_children_i
                    sum_of_pruned_channel_j = sum_of_current_layer_channel_j + sum_of_children_j 
                    # Compare
                    if sum_of_pruned_channel_i <= sum_of_pruned_channel_j:
                        pruned_channel = channel_i
                    else:
                        pruned_channel = channel_j
                

                ## Assign Zero to mask
                tmp = {}
                computation = 0
                # Current
                current_weights_mask[:, :, :, pruned_channel] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, :, pruned_channel]))
                tmp.update({layer: {'channel': pruned_channel}})
                # (computation)
                # remove the current pruned mask (for computing computation)
                unpruned_current_mask = current_weights_mask
                current_channel_num = np.shape(current_weights_mask)[3]
                current_depth = np.shape(current_weights_mask)[2]
                current_pruned_channels = []
                current_pruned_depth = []
                # (channel)
                for i in range(current_channel_num):
                    if np.sum(current_weights_mask[:, :, :, i]) == 0:
                        current_pruned_channels = np.append(current_pruned_channels, [i])
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_channels, axis = 3)
                # (depth)
                for i in range(current_depth):
                    if np.sum(current_weights_mask[:, :, i, :]) == 0:
                        current_pruned_depth = np.append(current_pruned_depth, [i])
                #
                unpruned_current_mask = np.delete(unpruned_current_mask, current_pruned_depth, axis = 2)
                current_weights_shape = np.shape(unpruned_current_mask)
                computation = computation + current_weights_shape[0] * current_weights_shape[1] * current_weights_shape[2] * current_outputs_shape[2] * current_outputs_shape[3]
                
                # Children
                if bool(all_children.get(layer)):
                    children = all_children[layer]
                    for _, child in enumerate(children):
                        interval = pruned_channel
                        start = children[child]
                        pruned_depth = start + interval
                        all_mask[child][:, :, pruned_depth, :] = 0
                        pruned_num = pruned_num + np.sum(all_mask[child][:, :, pruned_depth, :])
                        tmp.update({child: {'depth': pruned_depth}})
                        # (computation)
                        # remove the pruned mask
                        unpruned_child_mask = all_mask[child]
                        child_channel_num = np.shape(unpruned_child_mask)[3]
                        child_depth = np.shape(unpruned_child_mask)[2]
                        child_pruned_channels = []
                        child_pruned_depth = []
                        # (channel)
                        for i in range(child_channel_num):
                            if np.sum(unpruned_child_mask[:, :, :, i]) == 0:
                                child_pruned_channels = np.append(child_pruned_channels, [i])
                        unpruned_child_mask = np.delete(unpruned_child_mask, child_pruned_channels, axis = 3)
                        # (depth)
                        for i in range(child_depth):
                            if np.sum(unpruned_child_mask[:, :, i, :]) == 0:
                                child_pruned_depth = np.append(child_pruned_depth, [i])
                        unpruned_child_mask = np.delete(unpruned_child_mask, child_pruned_depth, axis = 2)
                        # 
                        child_weights_shape = np.shape(unpruned_child_mask)
                        computation = computation + child_weights_shape[0] * child_weights_shape[1] * child_weights_shape[3] * child_weights_shape[2] * child_weights_shape[3]
                    
                tmp.update({'computation': computation})        
                pruned_weights_info.append(tmp)
                t.add_row([str(pruned_num)+' / '+str(total_num), angle, layer, 'channel' + str(pruned_channel), computation])
            break
    print(t)
    
    # Update all masks
    print('Updating Masks ... ')
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        #sess.run(tf.assign(prune_info_dict[layer]['weights'], all_weights[layer]))
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
    # See the parameter change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        after_pruned_weights_size = np.sum(sess.run(mask_tensor))
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' .format(layer, int(original_weights_size[layer]), int(after_pruned_weights_size)))
    # See the parameter shape change
    for layer_iter in range(0, len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        mask_tensor = prune_info_dict[layer]['mask']
        mask = sess.run(mask_tensor)
        # remove the pruned part
        unpruned_mask = mask
        # (channel)
        channel_num = np.shape(mask)[3]
        pruned_channels = []
        for channel in range(channel_num):
            if np.sum(unpruned_mask[:, :, :, channel]) == 0:
                pruned_channels = np.append(pruned_channels, [channel])
        unpruned_mask = np.delete(unpruned_mask, pruned_channels, axis = 3)
        # (depth)
        depths = np.shape(unpruned_mask)[2]
        pruned_depths = []
        for depth in range(depths):
            if np.sum(unpruned_mask[:, :, depth, :]) == 0:
                pruned_depths = np.append(pruned_depths, [depth])
        unpruned_mask = np.delete(unpruned_mask, pruned_depths, axis = 2)
        print('{} : \033[0;32m{}\033[0m -> \033[0;32m{}\033[0m' 
        .format(layer, np.shape(mask), np.shape(unpruned_mask)))
    
    print("Prune Over!")
    
    return pruned_weights_info
 
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
    data_num,
    BATCH_SIZE, 
    sess
    ):
    
    test_batch_size = BATCH_SIZE
    #batch_num = int(len(v_xs) / test_batch_size)
    batch_num = int(data_num/BATCH_SIZE)
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
        dict = dict_test
        dict_test.update({is_training: False}) #xs: v_xs_part,
        Y_pre_list, v_ys_part = sess.run([prediction_list, ys], feed_dict = dict_test)	
        
        # Combine Different Device Prediction
        for d in range(len(Y_pre_list)):
            if d == 0:
                Y_pre = Y_pre_list[d]
            else:
                Y_pre = np.concatenate([Y_pre, Y_pre_list[d]], axis = 0)
                
        if Y_pre.ndim == 3:
            Y_pre = np.expand_dims(Y_pre, axis = 0)
        
        #if i%1000 == 0:
        #    Y_pre_part = Y_pre
        #else:
        #    Y_pre_part = np.concatenate([Y_pre_part, Y_pre], axis=0)
        #
        #if i == 0:
        #    Y_pre_total = Y_pre_part
        #else:
        #    Y_pre_total = np.concatenate([Y_pre_total, Y_pre_part], axis=0)
        Y_pre_total = []
        
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
        #else:
        #    result = np.concatenate([result, np.argmax(Y_pre, -1)], axis=0)
        
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
        print(accuracy_top1)
        #print("\r{} / {} -> Accuracy:{}" .format((i+1)*test_batch_size, data_num, accuracy_top1), end = "") 

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
	
#=============================#
#    Rebuilding Components    #
#=============================#
def middle_floor_decision(
    pruned_weights_info, 
    Model_Path_base
    ):
    # Original Computation
    with open(Model_Path_base + 'computation.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for _, row in enumerate(reader):
            original_computation = int(row[0])
    # Decide Middle Floors
    delete_num = 0
    selected_index = []
    error = 0
    while(1):
        t = PrettyTable(['Index', 'Computation', 'Computation (%)'])
        t.align = 'l'
        # Increase Computation 
        increase_computation = 0
        computation_record = []
        for i in range(len(pruned_weights_info)-delete_num)[::-1]:
            increase_computation = increase_computation + pruned_weights_info[i]['computation']
            computation_record.append(increase_computation)
        computation_record = computation_record[::-1]
        # Print Computation
        for i in range(len(pruned_weights_info)-delete_num):
            increase_computation = computation_record[i]
            increase_percent = float(increase_computation) / float(original_computation) * 100
            t.add_row([i, increase_computation, increase_percent])
        print(t)
        print("Middle Index : {}".format(selected_index))
        if error == 1:
            print("\033[1;31mError1\033[0m: You have to key the index at first time!")
            error = 0
        elif error == 2:
            print("\033[1;31mError2\033[0m: Not a number!")
            error = 0
        elif error == 3:
            print("\033[1;31mError3\033[0m: Over the highest index!")
            error = 0
        elif error == 4:
            print("\033[1;31mError4\033[0m: Index can not smaller than 0!")
            error = 0
        # Cin
        index = raw_input('Key one \033[1;32mindex\033[0m or \033[1;32m"c"\033[0m to start or \033[1;32m"e"\033[0m to exit :')
        if index == 'c':
            # Error1
            if not selected_index:
                error = 1
                continue
            print('Start Rebuilding!')
            break
        elif index == 'e':
            exit()
        else:
            try:
                index = int(index)
            except:
                # Error2
                error = 2
                continue
        if index == 0:
            print('Start Rebuilding!')
            selected_index.append(0)
            print("Middle Index : {}".format(selected_index))
            break
        elif index >= len(pruned_weights_info) - delete_num:
            # Error3
            error = 3
            continue
        elif index < 0:
            # Error4
            error = 4
            continue
        else:
            delete_num = len(pruned_weights_info) - index
            selected_index.append(index)
    return selected_index

def get_mask(
    model_path,
    model,
    H_Resize,
    W_Resize,
    class_num
    ):
    # Model Dictionary
    Model_dict = Model_dict_Generator(model_path + 'model.csv', class_num)
    # is_training
    is_training = tf.placeholder(tf.bool)
    # is_quantized_activation
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
    # is_ternary
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})  
    # net
    net = tf.ones([1, H_Resize, W_Resize, 3])
    # model
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = 0.0,
        data_format             = "NCHW",
        reuse                   = None)
    # mask tensor
    weights_mask_collection = tf.get_collection("float32_weights_mask", scope=None)
    # saver
    saver = tf.train.Saver()
    # Session
    with tf.Session() as sess:
        save_path = saver.restore(sess, model_path + model)
        mask = sess.run(weights_mask_collection)
    # reset graph    
    tf.reset_default_graph()
    
    return mask

def get_weights(
    model_path,
    model,
    H_Resize,
    W_Resize,
    class_num
    ):
    # Model Dictionary
    Model_dict = Model_dict_Generator(model_path + 'model.csv', class_num)
    # is_training
    is_training = tf.placeholder(tf.bool)
    # is_quantized_activation
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
    # is_ternary
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})  
    # net
    net = tf.ones([1, H_Resize, W_Resize, 3])
    # model
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = 0.0,
        data_format             = "NCHW",
        reuse                   = None)
    # mask tensor
    weights_collection = tf.get_collection("float32_weights", scope=None)
    # saver
    saver = tf.train.Saver()
    # Session
    with tf.Session() as sess:
        save_path = saver.restore(sess, model_path + model)
        weights = sess.run(weights_collection)
    # reset graph    
    tf.reset_default_graph()
    
    return weights
    
def update_var_list( # Garbage
    var_list,
    base_mask
    ):
    iter = 0
    var_list_new = []
    for _, var in enumerate(var_list):
        if var.name.split('_')[-1].split(':')[0] == 'weights' and iter == 0:  
            base_mask_now = base_mask[iter]
            [H, W, input_channel, output_channel] = np.shape(base_mask_now)
            #------------------#
            #   Channel Part   #
            #------------------#
            remove_channel = []
            # Get the channel to be removed
            for channel in range(output_channel):
                if np.mean(base_mask_now[:, :, :, channel]) != 0:
                    remove_channel.append(channel)
            base_mask_now = np.delete(base_mask_now, remove_channel, axis = 3)
            
            # Remove the channel
            start = 0
            end = output_channel
            channel_iter = 0
            if len(remove_channel) != 0:
                for _, channel in enumerate(remove_channel):
                    if start != channel:
                        interval = channel - start
                        slice = tf.slice(var, [0, 0, 0, start], [H, W, input_channel, interval])
                        if channel_iter == 0:
                            var_tmp = slice
                            channel_iter = channel_iter + 1
                        else:
                            var_tmp = tf.concat([var_tmp, slice], axis = 3)
                            channel_iter = channel_iter + 1
                    start = channel + 1
                if start < end:
                    interval = end - start
                    slice = tf.slice(var, [0, 0, 0, start], [H, W, input_channel, interval])
                    if channel_iter == 0:
                        var_tmp = slice
                        channel_iter = channel_iter + 1
                    else:
                        var_tmp = tf.concat([var_tmp, slice], axis = 3)
                        channel_iter = channel_iter + 1
            else:
                var_tmp = var
                
            assert output_channel - len(remove_channel) == var_tmp.get_shape().as_list()[-1], "Channel number is incorrect after remove!"

            #----------------#
            #   Depth Part   #
            #----------------#
            var_ = var_tmp
            [H, W, input_channel, output_channel] = var_tmp.get_shape().as_list()
            remove_depth = []
            # Get the depth to be removed
            for depth in range(input_channel):
                if np.mean(base_mask_now[:, :, depth, :]) != 0:
                    remove_depth.append(depth)
            # Remove the depth
            start = 0
            end = input_channel
            depth_iter = 0
            if len(remove_depth) != 0:
                for _, depth in enumerate(remove_depth):
                    print(depth)
                    if start != depth:
                        interval = depth - start
                        slice = tf.slice(var_, [0, 0, start, 0], [H, W, interval, output_channel])
                        if depth_iter == 0:
                            var_tmp = slice
                            depth_iter = depth_iter + 1
                        else:
                            var_tmp = tf.concat([var_tmp, slice], axis = 2)
                            depth_iter = depth_iter + 1
                    start = depth + 1
                if start < end:
                    interval = end - start
                    slice = tf.slice(var_, [0, 0, start, 0], [H, W, interval, output_channel])
                    if depth_iter == 0:
                        var_tmp = slice
                        depth_iter = depth_iter + 1
                    else:
                        var_tmp = tf.concat([var_tmp, slice], axis = 2)
                        depth_iter = depth_iter + 1
            else:
                var_tmp = var_

            iter = iter + 1
            var_list_new.append(var_tmp)
        #else:
            #var_list_new.append(var)
    return var_list_new

def update_gra_and_var( # Something Wrong 
    gra_and_var,
    base_mask
    ):
    mask_iter = 0
    gra_and_var_new = []
    for iter in range(len(gra_and_var)):
        gra = gra_and_var[iter][0]
        var = gra_and_var[iter][1]
        
        if var.name.split('_')[-1].split(':')[0] == 'weights':
            mask = base_mask[mask_iter]
            mask = 1. - mask
            mask_tensor = tf.constant(mask)
            mask_iter = mask_iter + 1
            gra_new = tf.multiply(gra, tf.zeros_like(gra))
            gra_and_var_new.append([gra_new, var])
        else:
            gra_and_var_new.append(gra_and_var[iter])
            
    return gra_and_var_new
        
#=============#
#    Debug    #
#=============#
def get_is_train_mask(
    model_path,
    model,
    H_Resize,
    W_Resize,
    class_num
    ):
    # Model Dictionary
    Model_dict = Model_dict_Generator(model_path + 'model.csv', class_num)
    # is_training
    is_training = tf.placeholder(tf.bool)
    # is_quantized_activation
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
    # is_ternary
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})  
    # net
    net = tf.ones([1, H_Resize, W_Resize, 3])
    # model
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = 0.0,
        data_format             = "NCHW",
        reuse                   = None)
    # mask tensor
    weights_mask_collection = tf.get_collection("is_train_float32_weights_mask", scope=None)
    # saver
    saver = tf.train.Saver()
    # Session
    with tf.Session() as sess:
        save_path = saver.restore(sess, model_path + model)
        mask = sess.run(weights_mask_collection)
    # reset graph    
    tf.reset_default_graph()
    
    return mask

def get_final_weights(
    model_path,
    model,
    H_Resize,
    W_Resize,
    class_num
    ):
    # Model Dictionary
    Model_dict = Model_dict_Generator(model_path + 'model.csv', class_num)
    # is_training
    is_training = tf.placeholder(tf.bool)
    # is_quantized_activation
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
    # is_ternary
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})  
    # net
    net = tf.ones([1, H_Resize, W_Resize, 3])
    # model
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = 0.0,
        data_format             = "NCHW",
        reuse                   = None)
    # mask tensor
    weights_collection = tf.get_collection("weights", scope=None)
    # saver
    saver = tf.train.Saver()
    # Session
    with tf.Session() as sess:
        save_path = saver.restore(sess, model_path + model)
        weights = sess.run(weights_collection)
    # reset graph    
    tf.reset_default_graph()
    
    return weights

def check_untrainable_variable_no_change(
    Dataset,
    Model_Path_1,
    Model_Path_2,
    Model_1,
    Model_2,
    ):
    
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
    
    mask_1 = get_is_train_mask(Model_Path_1, Model_1, H_Resize, W_Resize, class_num)
    mask_2 = get_is_train_mask(Model_Path_2, Model_2, H_Resize, W_Resize, class_num)
    weights_1 = get_final_weights(Model_Path_1, Model_1, H_Resize, W_Resize, class_num)
    weights_2 = get_final_weights(Model_Path_2, Model_2, H_Resize, W_Resize, class_num)

    # Check whether the untrainable variables are the same or not
    for iter in range(len(mask_1)):
        untrainable_weights_1 = (1 - mask_1[iter]) * weights_1[iter]
        untrainable_weights_2 = (1 - mask_1[iter]) * weights_2[iter]
        if np.sum(untrainable_weights_1 - untrainable_weights_2) != 0.0:
            print("\033[1;31mError\033[0m: Untrainable variables is different in iter {}" .format(iter))
            exit()
            
    print("\033[1;32mPass\033[0m")
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

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow

import Model_binary as Model
import vgg_preprocessing
import imagenet_preprocessing
import resnet_model
import cifar10

SHOW_MODEL = False
SHOW_FAST_MODEL = False
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
        BATCH_SIZE        = FLAGs['BatchSize']# / len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        Dataset           = FLAGs.Dataset
        Model_first_name  = FLAGs.Model_1st
        Model_second_name = FLAGs.Model_2nd
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs.BatchSize# / len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    
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
            if Model_second_name == '50':
                HP.update({'LR'                        : 0.0125          })  
                HP.update({'LR_Strategy'               : 'Normal'        })
                HP.update({'LR_Final'                  : 1e-3            })
                HP.update({'LR_Decade'                 : 10              }) 
                HP.update({'LR_Decade_1st_Epoch'       : 30              })
                HP.update({'LR_Decade_2nd_Epoch'       : 60              })
                HP.update({'LR_Decade_3rd_Epoch'       : 80              })
                HP.update({'LR_Decade_4th_Epoch'       : 90              })
                HP.update({'L2_Lambda'                 : 2e-4            })
                HP.update({'Opt_Method'                : 'Momentum'      })
                HP.update({'Momentum_Rate'             : 0.9             })
                HP.update({'IS_STUDENT'                : False           })
                HP.update({'Ternary_Epoch'             : 50              })
                HP.update({'Quantized_Activation_Epoch': 100             })
                HP.update({'Dropout_Rate'              : 0.0             })
            else:
                HP.update({'LR'                         : 0.1            })  
                HP.update({'LR_Strategy'                : 'Normal'       })
                HP.update({'LR_Final'                   : 1e-3           })
                HP.update({'LR_Decade'                  : 10             }) 
                HP.update({'LR_Decade_1st_Epoch'        : 80             })
                HP.update({'LR_Decade_2nd_Epoch'        : 120            })
                HP.update({'LR_Decade_3rd_Epoch'        : 180            })
                HP.update({'LR_Decade_4th_Epoch'        : 220            })
                HP.update({'L2_Lambda'                  : 1e-5             })
                HP.update({'Opt_Method'                 : 'Momentum'         })
                HP.update({'Momentum_Rate'              : 0.9            })
                HP.update({'IS_STUDENT'                 : False          })
                HP.update({'Ternary_Epoch'              : 0              })
                HP.update({'Ternary_LR_Strategy'        : 'Normal'       })
                HP.update({'Ternary_LR_Decade'          : 10             })
                HP.update({'Ternary_LR'                 : 1e-1           })
                HP.update({'Ternary_LR_Final'           : 1e-3           })
                HP.update({'Ternary_LR_Decade_1st_Epoch': 80             })
                HP.update({'Ternary_LR_Decade_2nd_Epoch': 120            })
                HP.update({'Ternary_LR_Decade_3rd_Epoch': 150            })
                HP.update({'Ternary_LR_Decade_4th_Epoch': 200            })
                HP.update({'Binary_Epoch'               : 0              })
                HP.update({'Binary_LR_Strategy'         : 'Normal'       })
                HP.update({'Binary_LR'                  : 1e-1           })
                HP.update({'Binary_LR_Final'            : 1e-4           })
                HP.update({'Binary_LR_Decade'           : 10             })
                HP.update({'Binary_LR_Decade_1st_Epoch' : 80             })
                HP.update({'Binary_LR_Decade_2nd_Epoch' : 120            })
                HP.update({'Binary_LR_Decade_3rd_Epoch' : 160            })
                HP.update({'Binary_LR_Decade_4th_Epoch' : 200            })
                HP.update({'Quantized_Activation_Epoch' : 0              })
                HP.update({'Dropout_Rate'               : 0.0            })
                
        elif Model_first_name == 'DenseNet':
            HP.update({'LR'                        : 1e-1      })
            HP.update({'LR_Strategy'               : 'Normal'  })
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
            HP.update({'LR'                        : 0.0000125 }) 
            HP.update({'LR_Strategy'               : 'Normal'  })
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
        elif Model_first_name == 'BinaryConnect':
            HP.update({'LR'                         : 0.01       })  
            HP.update({'LR_Strategy'                : 'Normal'   })
            HP.update({'LR_Final'                   : 1e-3       })
            HP.update({'LR_Decade'                  : 10         }) 
            HP.update({'LR_Decade_1st_Epoch'        : 80         })
            HP.update({'LR_Decade_2nd_Epoch'        : 120        })
            HP.update({'LR_Decade_3rd_Epoch'        : 200        })
            HP.update({'LR_Decade_4th_Epoch'        : 200        })
            HP.update({'L2_Lambda'                  : 0.         })
            HP.update({'Opt_Method'                 : 'Adam'     })
            HP.update({'Momentum_Rate'              : 0.9        })
            HP.update({'IS_STUDENT'                 : False      })
            HP.update({'Ternary_Epoch'              : 161        })
            HP.update({'Ternary_LR'                 : 1e-1       })
            HP.update({'Ternary_LR_Decade_1st_Epoch': 80         })
            HP.update({'Ternary_LR_Decade_2nd_Epoch': 120        })
            HP.update({'Ternary_LR_Decade_3rd_Epoch': 150        })
            HP.update({'Ternary_LR_Decade_4th_Epoch': 200        })
            HP.update({'Binary_Epoch'               : 0          })
            HP.update({'Binary_LR_Strategy'         : 'Normal'   })
            HP.update({'Binary_LR'                  : 1e-1       })
            HP.update({'Binary_LR_Final'            : 1e-4       })
            HP.update({'Binary_LR_Decade'           : 10         })
            HP.update({'Binary_LR_Decade_1st_Epoch' : 80         })
            HP.update({'Binary_LR_Decade_2nd_Epoch' : 120        })
            HP.update({'Binary_LR_Decade_3rd_Epoch' : 160        })
            HP.update({'Binary_LR_Decade_4th_Epoch' : 200        })
            HP.update({'Quantized_Activation_Epoch' : 100        })
            HP.update({'Dropout_Rate'               : 0.0        })
        
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
    test_Y_pre_path      ,
    training_type        ,
    diversify_layers     ,
    is_find_best_model   ,
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
        Model_first_name           = Model_first_name      ,
        Model_second_name          = Model_second_name     ,
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
        test_Y_pre_path            = test_Y_pre_path       ,
        training_type              = training_type         ,
        diversify_layers           = diversify_layers      ,
        is_find_best_model         = is_find_best_model    )
    
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
        Dataset             = FLAGs['Dataset']
        Model_first_name    = FLAGs['Model_1st']
        Model_second_name   = FLAGs['Model_2nd']
        EPOCH               = Epoch
        BATCH_SIZE          = FLAGs['BatchSize']
        Pruning_Strategy    = FLAGs['Pruning_Strategy']
        Pruning_Propotion   = FLAGs['Pruning_Propotion']
        Pruning_Times       = FLAGs['Pruning_Times']  
    else:
        Dataset             = FLAGs.Dataset
        Model_first_name    = FLAGs.Model_1st
        Model_second_name   = FLAGs.Model_2nd
        EPOCH               = Epoch
        BATCH_SIZE          = FLAGs.BatchSize
        Pruning_Strategy    = FLAGs.Pruning_Strategy
        Pruning_Propotion   = FLAGs.Pruning_Propotion
        Pruning_Times       = FLAGs.Pruning_Times
    
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
    HP.update({'Batch_Size'         : BATCH_SIZE})
    HP.update({'Epoch'              : EPOCH})
    HP.update({'H_Resize'           : H_Resize})
    HP.update({'W_Resize'           : W_Resize})
    HP.update({'train_num'          : train_num})
    HP.update({'valid_num'          : valid_num})
    HP.update({'test_num'           : test_num})
    HP.update({'Pruning_Strategy'   : Pruning_Strategy})
    HP.update({'Pruning_Propotion'  : Pruning_Propotion})
    HP.update({'Pruning_Times'      : Pruning_Times})
    if Model_first_name == 'ResNet':
        if Model_second_name == '50':
            HP.update({'LR'                        : 0.000125        })  
            HP.update({'LR_Strategy'               : 'Normal'        })
            HP.update({'LR_Final'                  : 1e-3            })
            HP.update({'LR_Decade'                 : 10              }) 
            HP.update({'LR_Decade_1st_Epoch'       : 30              })
            HP.update({'LR_Decade_2nd_Epoch'       : 150             })
            HP.update({'LR_Decade_3rd_Epoch'       : 150             })
            HP.update({'L2_Lambda'                 : 2e-4            })
            HP.update({'Opt_Method'                : 'Momentum'      })
            HP.update({'Momentum_Rate'             : 0.9             })
            HP.update({'Ternary_Epoch'             : 50              })
            HP.update({'Quantized_Activation_Epoch': 100             })
            HP.update({'Dropout_Rate'              : 0.0             })
        else:
            HP.update({'LR'                         : 0.001           })  
            HP.update({'LR_Strategy'                : 'Normal'        })
            HP.update({'LR_Final'                   : 1e-3            })
            HP.update({'LR_Decade'                  : 10              }) 
            HP.update({'LR_Decade_1st_Epoch'        : 50              })
            HP.update({'LR_Decade_2nd_Epoch'        : 90              })
            HP.update({'LR_Decade_3rd_Epoch'        : 130             })
            HP.update({'LR_Decade_4th_Epoch'        : 200             })
            HP.update({'L2_Lambda'                  : 2e-4            })
            HP.update({'Opt_Method'                 : 'Momentum'      })
            HP.update({'Momentum_Rate'              : 0.9             })
            HP.update({'Ternary_Epoch'              : 161             })
            HP.update({'Ternary_LR'                 : 1e-1            })
            HP.update({'Ternary_LR_Decade_1st_Epoch': 80              })
            HP.update({'Ternary_LR_Decade_2nd_Epoch': 120             })
            HP.update({'Ternary_LR_Decade_3rd_Epoch': 150             })
            HP.update({'Ternary_LR_Decade_4th_Epoch': 200             })
            HP.update({'Binary_Epoch'               : 0               })
            HP.update({'Binary_LR_Strategy'         : 'Normal'        })
            HP.update({'Binary_LR'                  : 1e-2            })
            HP.update({'Binary_LR_Final'            : 1e-4            })
            HP.update({'Binary_LR_Decade'           : 10              })
            HP.update({'Binary_LR_Decade_1st_Epoch' : 80              })
            HP.update({'Binary_LR_Decade_2nd_Epoch' : 120             })
            HP.update({'Binary_LR_Decade_3rd_Epoch' : 160             })
            HP.update({'Binary_LR_Decade_4th_Epoch' : 200             })
            HP.update({'Quantized_Activation_Epoch' : 100             })
            HP.update({'Dropout_Rate'               : 0.0             })
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
        HP.update({'Ternary_Epoch'             : 50              })
        HP.update({'Quantized_Activation_Epoch': 100             })
        HP.update({'Dropout_Rate'              : 0.0             })
    elif Model_first_name == 'MobileNet':
        HP.update({'LR'                        : 0.000125        })
        HP.update({'LR_Strategy'               : '3times'        })
        HP.update({'LR_Final'                  : 1e-3            })
        HP.update({'LR_Decade'                 : 30              })
        HP.update({'LR_Decade_1st_Epoch'       : 100             })
        HP.update({'LR_Decade_2nd_Epoch'       : 300             })
        HP.update({'LR_Decade_3rd_Epoch'       : 300             })
        HP.update({'L2_Lambda'                 : 2e-4            })
        HP.update({'Opt_Method'                : 'Momentum'      })
        HP.update({'Momentum_Rate'             : 0.9             })
        HP.update({'Ternary_Epoch'             : 50              })
        HP.update({'Quantized_Activation_Epoch': 100             })
        HP.update({'Dropout_Rate'              : 0.0             }) 
        
    print("\033[1;32mBATCH SIZE\033[0m : {}" .format(HP['Batch_Size']))
    
    #---------------------------#
    #    Hyperparameter Save    #
    #---------------------------#
    components = np.array(['Batch_Size'                 ,
                           'Epoch'                      ,
                           'H_Resize'                   ,
                           'W_Resize'                   ,
                           'LR'                         ,
                           'LR_Strategy'                ,
                           'LR_Final'                   ,
                           'LR_Decade'                  ,
                           'LR_Decade_1st_Epoch'        ,
                           'LR_Decade_2nd_Epoch'        ,
                           'LR_Decade_3rd_Epoch'        ,
                           'L2_Lambda'                  ,
                           'Opt_Method'                 ,
                           'Momentum_Rate'              ,
                           'Dropout_Rate'               ,
                           'Pruning_Strategy'           ,
                           'Pruning_Propotion'          ,
                           'Pruning_Times'              ])
    
    for iter, component in enumerate(components):
        if iter == 0:
            HP_csv = np.array([HP[component]])
        else:
            HP_csv = np.concatenate([HP_csv, np.array([HP[component]])], axis=0)
            
    components = np.expand_dims(components, axis=1)
    HP_csv = np.expand_dims(HP_csv, axis=1)
    HP_csv = np.concatenate([HP_csv, components], axis=1)
    
    #------------------#
    #   Global_Epoch   #
    #------------------#
    Global_Epoch = Global_Epoch % FLAGs.Epoch
    if Global_Epoch == 0:
        try:
            with open(pruning_model_path + 'info.csv', 'rb') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                best_accuracy = float(csvreader.next()[0])
                best_model = csvreader.next()[0]
                pruning_model_path = best_model.split(best_model.split('/')[-1])[0]
                pruning_model = best_model.split('/')[-1]
        except:
            None
    
    Model_Path, Model, Global_Epoch = Pruning(
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
    
    return Model_Path, Model, Global_Epoch
    
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
        mode              = FLAGs['mode']
    else:
        Dataset           = FLAGs.Dataset
        Model_first_name  = FLAGs.Model_1st
        Model_second_name = FLAGs.Model_2nd
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs.BatchSize
        mode              = FLAGs.mode
    
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
    HP.update({'mode'                      : mode            })
    
    HP.update({'LR'                        : 1e-3            }) 
    HP.update({'LR_Strategy'               : 'Normal'        })
    HP.update({'LR_Final'                  : 1e-3            })
    HP.update({'LR_Decade'                 : 10              })   
    HP.update({'LR_Decade_1st_Epoch'       : 80              })
    HP.update({'LR_Decade_2nd_Epoch'       : 120             })
    HP.update({'LR_Decade_3rd_Epoch'       : 200             })
    HP.update({'L2_Lambda'                 : 2e-4            })
    HP.update({'Opt_Method'                : 'Momentum'      })
    HP.update({'Momentum_Rate'             : 0.9             })
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
                           'Dropout_Rate'              ])
    
    for iter, component in enumerate(components):
        if iter == 0:
            HP_csv = np.array([HP[component]])
        else:
            HP_csv = np.concatenate([HP_csv, np.array([HP[component]])], axis=0)
    components = np.expand_dims(components, axis=1)
    HP_csv = np.expand_dims(HP_csv, axis=1)
    HP_csv = np.concatenate([HP_csv, components], axis=1)
    
    # Masks of untrainable model
    base_mask = get_mask(
        model_path = rebuilding_model_path_base,
        model      = rebuilding_model_base,
        H_Resize   = H_Resize,
        W_Resize   = W_Resize,
        class_num  = class_num)
    
    # Weights of untrainable model
    base_weights = get_weights(
        model_path = rebuilding_model_path_base,
        model      = rebuilding_model_base,
        H_Resize   = H_Resize,
        W_Resize   = W_Resize,
        class_num  = class_num)
    
    Model_Path, Model, Global_Epoch = Rebuilding(
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
    
    return Model_Path, Model, Global_Epoch

def run_diversifying(
    Hyperparameter          , 
    # Info                  
    FLAGs                   ,
    Epoch                   ,
    Global_Epoch            ,
    IS_HYPERPARAMETER_OPT   ,
    error                   ,
    # Path                  
    Dataset_Path            ,
    Y_pre_Path              ,
    teacher_model_path      ,
    teacher_model           ,
    student_model_path      ,
    student_model           ,
    pruned_model_path       ,
    pruned_model            ,
    PR_iter                 ,
    diversify_layer       
    ):
    
    if IS_IN_IPYTHON:
        Dataset           = FLAGs['Dataset']
        Model_first_name  = FLAGs['Model_1st']
        Model_second_name = FLAGs['Model_2nd']
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs['BatchSize']# / len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        Pruning_Strategy  = FLAGs[Pruning_Strategy]
    else:
        Dataset           = FLAGs.Dataset
        Model_first_name  = FLAGs.Model_1st
        Model_second_name = FLAGs.Model_2nd
        EPOCH             = Epoch
        BATCH_SIZE        = FLAGs.BatchSize# / len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        Pruning_Strategy  = FLAGs.Pruning_Strategy
    
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
    Model_dict = Model_dict_Generator('Model/' + Model_first_name + '_Model/' + Model_Name + '.csv', class_num)
    Model_dict_s = Model_dict_Generator(student_model_path + 'model.csv', class_num)
    
    #-----------------------------------#
    #   Hyperparameter : User Defined   #
    #-----------------------------------#
    if not IS_HYPERPARAMETER_OPT:
        HP = {}
        HP.update({'Batch_Size'         : BATCH_SIZE      })
        HP.update({'Epoch'              : EPOCH           })
        HP.update({'H_Resize'           : H_Resize        })
        HP.update({'W_Resize'           : W_Resize        })
        HP.update({'train_num'          : train_num       })
        HP.update({'valid_num'          : valid_num       })
        HP.update({'test_num'           : test_num        })
        HP.update({'Pruning_Strategy'   : Pruning_Strategy})
        
        if Model_first_name == 'ResNet':
            if Model_second_name == '50':
                HP.update({'LR'                        : 0.0125          })  
                HP.update({'LR_Strategy'               : 'Normal'        })
                HP.update({'LR_Final'                  : 1e-3            })
                HP.update({'LR_Decade'                 : 10              }) 
                HP.update({'LR_Decade_1st_Epoch'       : 30              })
                HP.update({'LR_Decade_2nd_Epoch'       : 60              })
                HP.update({'LR_Decade_3rd_Epoch'       : 80              })
                HP.update({'LR_Decade_4th_Epoch'       : 90              })
                HP.update({'L2_Lambda'                 : 2e-4            })
                HP.update({'Opt_Method'                : 'Momentum'      })
                HP.update({'Momentum_Rate'             : 0.9             })
                HP.update({'IS_STUDENT'                : False           })
                HP.update({'Ternary_Epoch'             : 50              })
                HP.update({'Quantized_Activation_Epoch': 100             })
                HP.update({'Dropout_Rate'              : 0.0             })
            else:
                HP.update({'LR'                         : 0.01           })  
                HP.update({'LR_Strategy'                : 'Normal'       })
                HP.update({'LR_Final'                   : 1e-3           })
                HP.update({'LR_Decade'                  : 10             }) 
                HP.update({'LR_Decade_1st_Epoch'        : 20             })
                HP.update({'LR_Decade_2nd_Epoch'        : 80             })
                HP.update({'LR_Decade_3rd_Epoch'        : 100            })
                HP.update({'LR_Decade_4th_Epoch'        : 220            })
                HP.update({'L2_Lambda'                  : 1.             })
                HP.update({'Opt_Method'                 : 'Momentum'     })
                HP.update({'Momentum_Rate'              : 0.9            })
                HP.update({'IS_STUDENT'                 : False          })
                HP.update({'Ternary_Epoch'              : 0              })
                HP.update({'Ternary_LR_Strategy'        : 'Normal'       })
                HP.update({'Ternary_LR_Decade'          : 10             })
                HP.update({'Ternary_LR'                 : 1e-1           })
                HP.update({'Ternary_LR_Final'           : 1e-3           })
                HP.update({'Ternary_LR_Decade_1st_Epoch': 80             })
                HP.update({'Ternary_LR_Decade_2nd_Epoch': 120            })
                HP.update({'Ternary_LR_Decade_3rd_Epoch': 150            })
                HP.update({'Ternary_LR_Decade_4th_Epoch': 200            })
                HP.update({'Binary_Epoch'               : 0              })
                HP.update({'Binary_LR_Strategy'         : 'Normal'       })
                HP.update({'Binary_LR'                  : 1e-4           })
                HP.update({'Binary_LR_Final'            : 1e-4           })
                HP.update({'Binary_LR_Decade'           : 10             })
                HP.update({'Binary_LR_Decade_1st_Epoch' : 80             })
                HP.update({'Binary_LR_Decade_2nd_Epoch' : 120            })
                HP.update({'Binary_LR_Decade_3rd_Epoch' : 160            })
                HP.update({'Binary_LR_Decade_4th_Epoch' : 200            })
                HP.update({'Quantized_Activation_Epoch' : 0              })
                HP.update({'Dropout_Rate'               : 0.0            })                
        elif Model_first_name == 'DenseNet':
            HP.update({'LR'                        : 1e-1      })
            HP.update({'LR_Strategy'               : 'Normal'  })
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
            HP.update({'LR'                        : 0.0000125 }) 
            HP.update({'LR_Strategy'               : 'Normal'  })
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
        elif Model_first_name == 'BinaryConnect':
            HP.update({'LR'                         : 0.01       })  
            HP.update({'LR_Strategy'                : 'Normal'   })
            HP.update({'LR_Final'                   : 1e-3       })
            HP.update({'LR_Decade'                  : 10         }) 
            HP.update({'LR_Decade_1st_Epoch'        : 80         })
            HP.update({'LR_Decade_2nd_Epoch'        : 120        })
            HP.update({'LR_Decade_3rd_Epoch'        : 200        })
            HP.update({'LR_Decade_4th_Epoch'        : 200        })
            HP.update({'L2_Lambda'                  : 0.         })
            HP.update({'Opt_Method'                 : 'Adam'     })
            HP.update({'Momentum_Rate'              : 0.9        })
            HP.update({'IS_STUDENT'                 : False      })
            HP.update({'Ternary_Epoch'              : 161        })
            HP.update({'Ternary_LR'                 : 1e-1       })
            HP.update({'Ternary_LR_Decade_1st_Epoch': 80         })
            HP.update({'Ternary_LR_Decade_2nd_Epoch': 120        })
            HP.update({'Ternary_LR_Decade_3rd_Epoch': 150        })
            HP.update({'Ternary_LR_Decade_4th_Epoch': 200        })
            HP.update({'Binary_Epoch'               : 0          })
            HP.update({'Binary_LR_Strategy'         : 'Normal'   })
            HP.update({'Binary_LR'                  : 1e-1       })
            HP.update({'Binary_LR_Final'            : 1e-4       })
            HP.update({'Binary_LR_Decade'           : 10         })
            HP.update({'Binary_LR_Decade_1st_Epoch' : 80         })
            HP.update({'Binary_LR_Decade_2nd_Epoch' : 120        })
            HP.update({'Binary_LR_Decade_3rd_Epoch' : 160        })
            HP.update({'Binary_LR_Decade_4th_Epoch' : 200        })
            HP.update({'Quantized_Activation_Epoch' : 100        })
            HP.update({'Dropout_Rate'               : 0.0        })
            
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
    
    Global_Epoch = Global_Epoch % FLAGs.Epoch
    
    #----------------#
    #    Training    #
    #----------------#        
    Model_Path, Model, error, Global_Epoch = Diversifying( 
        Model_dict              = Model_dict            ,
        Model_dict_s            = Model_dict_s          ,
        Dataset                 = Dataset               ,
        Dataset_Path            = Dataset_Path          ,
        Y_pre_Path              = Y_pre_Path            ,
        class_num               = class_num             ,
        diversify_layer         = diversify_layer       ,
        HP                      = HP                    ,
        Global_Epoch            = Global_Epoch          ,
        weights_bd_ratio        = 50                    ,
        biases_bd_ratio         = 50                    ,         
        HP_csv                  = HP_csv                ,
        Model_first_name        = Model_first_name      ,
        Model_second_name       = Model_second_name     ,
        teacher_model_path      = teacher_model_path    ,
        teacher_model           = teacher_model         ,
        student_model_path      = student_model_path    ,
        student_model           = student_model         ,
        pruned_model_path       = pruned_model_path     ,
        pruned_model            = pruned_model          ,
        PR_iter                 = PR_iter               ,
        IS_HYPERPARAMETER_OPT   = IS_HYPERPARAMETER_OPT ,
        error                   = error                 )
        
    return Model_Path, Model, error, Global_Epoch
   
   
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
    
    #-----------------------------#
    #   Some Control Parameters   #
    #-----------------------------#
    IS_TERNARY = False
    IS_BINARY = False
    IS_QUANTIZED_ACTIVATION = False
    for layer in range(len(Model_dict)):
        if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
            IS_TERNARY = True
        if Model_dict['layer'+str(layer)]['IS_BINARY'] == 'TRUE':
            IS_BINARY = True
        if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
            IS_QUANTIZED_ACTIVATION = True

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
    #xs, ys = cifar10.distorted_inputs()
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
    
    ## is_binary
    is_binary = {}
    for layer in range(len(Model_dict)):
        is_binary.update({'layer%d'%layer : tf.placeholder(tf.bool)})

    #----------------------#
    #    Building Model    #
    #----------------------# 
    print("Building Model ...")
    data_format = "NCHW"
    ## -- Deivce --
    GPUs = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    device_num = len(GPUs)
    #batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
    #      [xs, ys], capacity=2 * device_num)
    ## -- Optimizer --
    if HP['Opt_Method']=='Adam':
        opt = tf.train.AdamOptimizer(learning_rate, HP['Momentum_Rate'])
    elif HP['Opt_Method']=='Momentum':
        opt = tf.train.MomentumOptimizer(
            learning_rate = learning_rate, 
            momentum      = HP['Momentum_Rate'])
    if device_num > 1:
        opt = tf.contrib.estimator.TowerOptimizer(opt)
    
    ## -- model --
    net = xs
    net_ = tf.split(net, num_or_size_splits=device_num, axis=0)
    labels = ys
    labels_ = tf.split(labels, num_or_size_splits=device_num, axis=0)
    Model_dict_ = copy.deepcopy(Model_dict)
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for d, device in enumerate(['/device:GPU:%s' %(GPU) for GPU in range(device_num)]):
            with tf.device(device):
                print(device)
                """
                ## -- Dequeues one batch for the GPU --
                image_batch, label_batch = batch_queue.dequeue()
                label_batch = tf.one_hot(label_batch, 10)
                """
                image_batch = net_[d]
                label_batch = labels_[d]
                
                ## -- Build Model --
                prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
                net                     = image_batch, 
                Model_dict              = Model_dict_, 
                is_training             = is_training,
                is_ternary              = is_ternary,
                is_binary               = is_binary,
                is_quantized_activation = is_quantized_activation,
                DROPOUT_RATE            = HP['Dropout_Rate'],
                data_format             = data_format,
                reuse                   = None)
        
                ## -- Loss --
                # L2 Regularization
                l2_norm   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                        if 'float32_weights' in v.name])
                l2_lambda = tf.constant(HP['L2_Lambda'])
                l2_norm   = tf.multiply(l2_lambda, l2_norm)
                
                # Cross Entropy
                cross_entropy = tf.losses.softmax_cross_entropy(
                    onehot_labels = label_batch,
                    logits        = prediction)
                # Total Loss
                if IS_TERNARY and (Global_Epoch+1) >= HP['Ternary_Epoch']:
                    loss = cross_entropy
                elif IS_BINARY and (Global_Epoch+1) >= HP['Binary_Epoch']:
                    l2_norm   = tf.add_n([tf.reduce_sum(tf.square(tf.ones_like(v)-tf.abs(v))) for v in tf.trainable_variables()
                                        if 'float32_weights' in v.name])
                    l2_lambda = tf.constant(HP['L2_Lambda'])
                    l2_norm   = tf.multiply(l2_lambda, l2_norm)
                    loss = cross_entropy + l2_norm
                else:
                    loss = cross_entropy + l2_norm
                
                # Setting variable to reuse mode
                tf.get_variable_scope().reuse_variables()
                
                ## -- Gradients --
                # Batch norm requires update ops to be added as a dependency to the train_op
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Compute Gradients
                    var_list = tf.trainable_variables()
                    gra_and_var = opt.compute_gradients(loss, var_list = var_list)
                tower_grads.append(gra_and_var)

    ## Apply Gradients
    gra_and_vars = average_gradients(tower_grads)
    train_step  = opt.apply_gradients(gra_and_vars, global_step)
    
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
        if 'final_weights' not in variable.name and 'final_biases' not in variable.name:
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
            #print("{}, {}" .format(i, variable))
            all_variables.append([variable.name, str(variable.get_shape().as_list())])
            #pdb.set_trace()
            i = i + 1
    
    #np.savetxt('../pre-trained_model/all_variables.csv', all_variables, delimiter=",", fmt="%s")
    #exit()
    print("\033[0;36m=======================\033[0m")
    print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size))
    print("\033[0;36m=======================\033[0m")
    
    ## -- Collection --
    float32_weights_collection          = tf.get_collection("float32_weights"        , scope=None)
    float32_biases_collection           = tf.get_collection("float32_biases"         , scope=None)
    ## Ternary
    ternary_weights_bd_collection       = tf.get_collection("ternary_weights_bd"     , scope=None)
    ternary_biases_bd_collection        = tf.get_collection("ternary_biases_bd"      , scope=None)
    ## Binary
    binary_weights_bd_collection        = tf.get_collection("binary_weights_bd"      , scope=None)
    binary_biases_bd_collection         = tf.get_collection("binary_biases_bd"       , scope=None)
    clip_weights_collection             = tf.get_collection("clip_weights"           , scope=None)
    ## assign ternary or float32 weights/biases to final weights/biases  
    assign_var_list_collection          = tf.get_collection("assign_var_list"        , scope=None)  
    ## Actvation Quantization    
    float32_net_collection              = tf.get_collection("float32_net"            , scope=None)
    is_quantized_activation_collection  = tf.get_collection("is_quantized_activation", scope=None)
    mantissa_collection                 = tf.get_collection("mantissa"               , scope=None)
    fraction_collection                 = tf.get_collection("fraction"               , scope=None)
    quantized_net_collection            = tf.get_collection("quantized_net"          , scope=None)
    ## Gradient Update
    var_list_collection                 = tf.get_collection("var_list"               , scope=None)
    float32_params                      = tf.get_collection("float32_params"         , scope=None) 
    ## Final weights
    weights_collection = tf.get_collection("weights", scope=None)
    biases_collection  = tf.get_collection("biases", scope=None)

    # Update gra_and_var
    """
    for iter, gra_and_var_ in enumerate(gra_and_var):
        if_is_ternary = Model_dict[gra_and_var_[1].name.split('/')[1].split('_')[0]]['IS_TERNARY'] == 'TRUE'
        if_is_binary  = Model_dict[gra_and_var_[1].name.split('/')[1].split('_')[0]]['IS_BINARY'] == 'TRUE'
        is_add_biases = Model_dict[gra_and_var_[1].name.split('/')[1].split('_')[0]]['is_add_biases'] == 'TRUE'
        if if_is_ternary or if_is_binary:
            if gra_and_var_[1].name.split('/')[-1] == 'float32_weights:0':
                if is_add_biases:
                    gra = gra_and_var[iter+2][0]
                else:
                    gra = gra_and_var[iter+1][0]
                lst = list(gra_and_var_)
                lst[0] = gra
                lst = tuple(lst)
                gra_and_var[iter] = lst
    """
    """    
    for iter, gra_and_var_ in enumerate(gra_and_var):
        print(gra_and_var_)
    """

    #-----------#
    #   Saver   #
    #-----------#	
    saver = tf.train.Saver()
    
    TERNARY_NOW = False
    BINARY_NOW = False
    QUANTIZED_NOW = False
    #---------------#
    #    Session    #
    #---------------#
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.allow_soft_placement = True
    #config.intra_op_parallelism_threads = 256
    with tf.Session(config = config) as sess: 
        #----------------------#
        #    Initialization    #
        #----------------------#
        print("Initializing ...")
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Start the queue runners.
        #tf.train.start_queue_runners(sess=sess)
        
        #kernel_values_per_layer = similar_group(inputs_and_kernels, sess)
        #--------------------------#
        #   Load trained weights   #
        #--------------------------#
        if trained_model_path!=None and trained_model!=None:
            print("Loading the trained weights ... ")
            print("\033[0;35m{}\033[0m" .format(trained_model_path + trained_model))
            save_path = saver.restore(sess, trained_model_path + trained_model)
            
        """
        path = '/home/2016/b22072117/tmp/mobilenet_model/'
        name = 'mobilenet_100_100'
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
            if HP['LR_Strategy'] == 'Normal':
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
            
            # Ternary Learning Rate
            if IS_TERNARY and (Global_Epoch+epoch+1) >= HP['Ternary_Epoch']:
                if HP['Ternary_LR_Strategy'] == 'Normal':
                    if   (Global_Epoch+epoch+1) <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_1st_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 0)
                    elif (Global_Epoch+epoch+1) <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_2nd_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 1)
                    elif (Global_Epoch+epoch+1) <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_3rd_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 2)
                    elif (Global_Epoch+epoch+1) <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_4th_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 3)
                    else:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 4)

            # Binary Learning Rate
            if IS_BINARY and (Global_Epoch+epoch+1) >= HP['Binary_Epoch']:
                if HP['Binary_LR_Strategy'] == 'Normal':
                    if   (Global_Epoch+epoch+1) <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_1st_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 0)
                    elif (Global_Epoch+epoch+1) <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_2nd_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 1)
                    elif (Global_Epoch+epoch+1) <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_3rd_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 2)
                    elif (Global_Epoch+epoch+1) <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_4th_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 3)
                    else:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 4)
                elif HP['Binary_LR_Strategy'] == 'Exponential':
                    decay_rate = float(HP['Binary_LR_Final'] / HP['Binary_LR'])
                    decay_step = float(Global_Epoch+epoch-HP['Binary_Epoch']) / float(500-HP['Binary_Epoch'])
                    lr = HP['Binary_LR'] * pow(decay_rate, decay_step)
            
            ## -- Quantizad Activation --
            if IS_QUANTIZED_ACTIVATION and (epoch+1+Global_Epoch)>=HP['Quantized_Activation_Epoch']:
                """
                batch_xs = train_data[0:HP['Batch_Size']]
                # Calculate Each Activation's appropriate mantissa and fractional bit
                m, f = quantized_m_and_f(float32_net_collection, is_quantized_activation_collection, xs, Model_dict, batch_xs, sess)	
                # Assign mantissa and fractional bit to the tensor
                assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
                """
                # Start Quantize Activation
                QUANTIZED_NOW = True
            
            ## -- Ternary --
            if IS_TERNARY and (epoch+1+Global_Epoch)>=HP['Ternary_Epoch']:
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
            
            ## -- Binary --
            if IS_BINARY and (epoch+1+Global_Epoch)>=HP['Binary_Epoch']:
                """
                # Calculate the binary boundary of each layer's weights
                weights_bd, biases_bd, weights_table, biases_table = binarize_bd(
                    float32_weights_collection,  
                    float32_biases_collection, 
                    weights_bd_ratio, 
                    biases_bd_ratio, 
                    sess)
                # assign binary boundary to tensor
                assign_binary_boundary(binary_weights_bd_collection, binary_biases_bd_collection, weights_bd, biases_bd, sess)
                """
                # Start Quantize Activation
                BINARY_NOW = True
                
            ## -- Set feed_dict --
            # train_step
            feed_dict_train = {}
            for layer in range(len(Model_dict)):
                feed_dict_train.update({is_ternary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_train.update({is_binary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_BINARY']=='TRUE' and BINARY_NOW})
                feed_dict_train.update({is_quantized_activation['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION']=='TRUE' and QUANTIZED_NOW}) 
                
            # Assign float32 or ternary/binary weight and biases to final weights
            feed_dict_assign = {}
            for layer in range(len(Model_dict)):
                feed_dict_assign.update({is_ternary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_assign.update({is_binary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_BINARY']=='TRUE' and BINARY_NOW})
            
            ## -- Training --
            tStart = time.time()
            tStart_Batch = time.time()
            total_batch_iter = int(train_data_num / HP['Batch_Size'])
            for batch_iter in range(total_batch_iter):
                iteration = iteration + 1
                
                # Assign float32 or ternary/bianry weight and biases to final weights
                # This may can be placed into GraphKeys.UPDATE_OPS, but i am not sure.
                for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                    sess.run(assign_var_list, feed_dict = feed_dict_assign)
                
                # Run Training Step
                feed_dict_train.update({is_training: True, learning_rate: lr}) # xs: batch_xs, ys: batch_ys,
                
                time1 = time.time()
                ###################################################
                _, Loss, Prediction, L2_norm, batch_ys, quantized_net = sess.run(
                    [train_step, loss, prediction, l2_norm, labels, quantized_net_collection], 
                    feed_dict = feed_dict_train)
                ####################################################
                time2 = time.time()

                #print(time2-time1) 

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
                """  
                # Per Class Accuracy
                """
                per_class_accuracy(Prediction, batch_ys)
                """
            # Assign float32 or ternary/binary weight and biases to final weights
            # This may can be placed into GraphKeys.UPDATE_OPS, but i am not sure.
            for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                sess.run(assign_var_list, feed_dict = feed_dict_assign)
            
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
    Model_first_name           ,
    Model_second_name          ,
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
    test_Y_pre_path            ,
    training_type              ,
    diversify_layers           ,
    is_find_best_model
    ):
    
    print("Testing ... ")
    Model_Name = Model_first_name + '_' + Model_second_name
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
    
    # is_binary
    is_binary = {}
    for layer in range(len(Model_dict)):
        is_binary.update({'layer%d'%layer : tf.placeholder(tf.bool)})  
    
    # is_fast_mode
    is_fast_mode = {}
    for layer in range(len(Model_dict)):
        is_fast_mode.update({'layer%d'%layer : tf.placeholder(tf.bool, name='is_fast_mode_layer'+str(layer))})
    
    # complexity_mode
    complexity_mode = {}
    for layer in range(len(Model_dict)):
        complexity_mode.update({'layer%d'%layer : tf.placeholder(tf.float32, name='complexity_mode_layer'+str(layer))})

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
    data_format = "NCHW"
    ## -- Build Model --
    net = xs[0 : BATCH_SIZE]
    Model_dict_ = copy.deepcopy(Model_dict)

    DIVERSIFY_MODE_list = [Model_dict_[D]['DIVERSIFY_MODE'] for D in Model_dict.keys()]
    if all(D=='None' for D in DIVERSIFY_MODE_list):
        prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
            net                     = net, 
            Model_dict              = Model_dict_, 
            is_training             = is_training,
            is_ternary              = is_ternary,
            is_binary               = is_binary,
            is_quantized_activation = is_quantized_activation,
            DROPOUT_RATE            = None,
            data_format             = data_format,
            reuse                   = None)
    else:
        with tf.variable_scope('student'):
            is_training = {}
            for layer in range(len(Model_dict)):
                is_training.update({'layer%d'%layer : tf.placeholder(tf.bool)})
            complexity_mode_now = None
            prediction, _, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder_DiversifyVersion(
                net                     = net, 
                Model_dict              = Model_dict_,
                diversify_layer         = None,
                is_testing              = True,
                is_student              = True,
                is_restoring            = False,
                is_training             = is_training,
                is_ternary              = is_ternary,
                is_binary               = is_binary,
                is_quantized_activation = is_quantized_activation,
                is_fast_mode            = is_fast_mode,
                complexity_mode         = complexity_mode,
                complexity_mode_now     = complexity_mode_now,
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
    IS_BINARY = False
    IS_QUANTIZED_ACTIVATION = False
    for layer in range(len(Model_dict)):
        if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
            IS_TERNARY = True
        if Model_dict['layer'+str(layer)]['IS_BINARY'] == 'TRUE':
            IS_BINARY = True    
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
        
        # Binary Debug
        float32_weights_collection = tf.get_collection("float32_weights", scope=None)
        float32_biases_collection = tf.get_collection("float32_biases", scope=None)
        clip_weights_collection = tf.get_collection("clip_weights", scope=None)
        binary_weights_collection = tf.get_collection("binary_weights", scope=None)
        biases_collection = tf.get_collection("biases", scope=None)
        
        DIVERSIFY_MODE_list = [Model_dict_[D]['DIVERSIFY_MODE'] for D in Model_dict.keys()]
        if not all(D=='None' for D in DIVERSIFY_MODE_list):
            feed_dict = {}
            for layer in range(len(Model_dict)):
                feed_dict.update({is_training['layer'+str(layer)]: False})
            float32_weights = sess.run(float32_weights_collection, feed_dict=feed_dict)
            binary_weights = sess.run(binary_weights_collection, feed_dict=feed_dict)
            feed_dict = {}
        else:
            float32_weights = sess.run(float32_weights_collection, feed_dict={is_training: False})
            binary_weights = sess.run(binary_weights_collection, feed_dict={is_training: False})
        print([np.mean(np.abs(binary_weights[v])) for v in range(len(binary_weights))])
        
        
        # Testing
        print("Testing Data Result ... ")
        if training_type == 'diversify':
            ## Pruning Complexity Mode
            complexity_mode_dict = {}
            for layer in range(len(Model_dict)):
                if Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '4':
                    complexity_mode_dict.update({complexity_mode['layer'+str(layer)]: float(Model_dict['layer'+str(layer)]['REBUILING_MODE_MAX'])-1.0})
                elif Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '5':
                    complexity_mode_dict.update({complexity_mode['layer'+str(layer)]: float(Model_dict['layer'+str(layer)]['REBUILING_MODE_MAX'])-1.0})
                elif Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '6':
                    complexity_mode_dict.update({complexity_mode['layer'+str(layer)]: float(Model_dict['layer'+str(layer)]['REBUILING_MODE_MAX'])-1.0})
                            
            ## Diversity
            key = Model_dict.keys()
            sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
            sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]
            diversify_layer_num = sum([Model_dict[layer]['IS_DIVERSITY']=='TRUE' for layer in sorted_layer])
            diversify_layers = [layer for layer in sorted_layer if Model_dict[layer]['IS_DIVERSITY']=='TRUE']
            #diversify_layer_num = len(diversify_layers)
            test_accuracy_total = []
            test_accuracy_total.append(np.concatenate([np.array(diversify_layers), np.array(['Accuracy'])], axis=0))
            for d in range(pow(2, diversify_layer_num)):
                d2b = '{0:0' + str(diversify_layer_num) + 'b}'
                b = d2b.format(d)
                # Initial
                is_fast_mode_dict = {}
                for layer in sorted_layer:
                    is_fast_mode_dict.update({is_fast_mode[layer]: False})
                # Update
                if diversify_layer_num != 0:
                    for bit, value in enumerate(b):
                        # mode
                        if bit == 0:
                            mode = np.array([1 - int(value)])
                        else:
                            mode = np.concatenate([mode, np.array([1 - int(value)])])
                        # is_fast_mode_dict
                        is_fast_mode_dict.update({is_fast_mode[diversify_layers[bit]]: value != '0' })
                        # show
                        if value != '0':
                            print_mode = 'F'
                            print("%s: \033[1;31m%d\033[0m" %(diversify_layers[bit], 1-int(value)), end = " ")
                        else:
                            print_mode = 'N'
                            print("%s: \033[1;34m%d\033[0m" %(diversify_layers[bit], 1-int(value)), end = " ")
                
                ## Rebuilding Complexity Mode
                # total_C_mode
                REBUILING_MODE_MAX_list = []
                total_C_mode = 1
                for layer in range(len(Model_dict)):
                    if Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '7':
                        REBUILING_MODE_MAX_list.append(total_C_mode)
                        total_C_mode = total_C_mode * (int(Model_dict['layer'+str(layer)]['REBUILING_MODE_MAX'])+1)
                        
                # Each C_mode
                for C_mode in range(total_C_mode)[::-1]:
                    # Decode C_mode
                    decode_C_mode = []
                    dividend = C_mode
                    for divisor in REBUILING_MODE_MAX_list[::-1]:
                        decode_C_mode.append(dividend/divisor)
                        dividend = dividend % divisor
                    decode_C_mode = decode_C_mode[::-1]
                    
                    # complexity_mode_dict
                    black_box_iter = 0
                    feed_dict_assign = {}
                    for layer in range(len(Model_dict)):
                        if Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '7':
                            complexity_mode_value = decode_C_mode[black_box_iter]
                            feed_dict_assign.update({complexity_mode['layer'+str(layer)]: complexity_mode_value})
                            complexity_mode_dict.update({complexity_mode['layer'+str(layer)]: complexity_mode_value})
                            print("%s: \033[1;31m%d\033[0m" %('layer'+str(layer), complexity_mode_value), end = " ")
                        if Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '8':
                            complexity_mode_value = decode_C_mode[black_box_iter]
                            feed_dict_assign.update({complexity_mode['layer'+str(layer)]: complexity_mode_value})
                            complexity_mode_dict.update({complexity_mode['layer'+str(layer)]: complexity_mode_value})
                        if Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '9':
                            complexity_mode_value = decode_C_mode[black_box_iter]
                            feed_dict_assign.update({complexity_mode['layer'+str(layer)]: complexity_mode_value})
                            complexity_mode_dict.update({complexity_mode['layer'+str(layer)]: complexity_mode_value})
                            black_box_iter = black_box_iter + 1

                    # Assign
                    assign_var_list_collection = tf.get_collection("assign_var_list", scope=None)  
                    for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                        sess.run(assign_var_list, feed_dict = feed_dict_assign)
                
                
                    test_result, test_accuracy, test_accuracy_top2, test_accuracy_top3, test_Y_pre = compute_accuracy(
                        xs                      = xs, 
                        ys                      = ys, 
                        is_training             = is_training,
                        is_quantized_activation = is_quantized_activation,
                        is_fast_mode_dict       = is_fast_mode_dict,
                        complexity_mode         = complexity_mode,
                        complexity_mode_dict    = complexity_mode_dict,
                        Model_dict              = Model_dict,                
                        QUANTIZED_NOW           = True, 
                        prediction_list         = [prediction], 
                        data_num                = test_data_num,
                        BATCH_SIZE              = BATCH_SIZE, 
                        sess                    = sess)
                    
                    if diversify_layer_num != 0:
                        mode = np.concatenate([mode, np.array([test_accuracy])])
                        test_accuracy_total.append(mode)
                    
                    print("\033[0;32mTesting Data Accuracy\033[0m = {test_Accuracy}"
                    .format(test_Accuracy = test_accuracy))
                
                # Debug (Check if final weights equals to constant weights)
                """
                for layer in range(len(Model_dict)):
                    cond0 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '7'
                    cond1 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '8'
                    cond2 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '9'
                    if (cond0 or cond1 or cond2) and Model_dict['layer'+str(layer)]['type']=="CONV":
                        constant_weights_collection = tf.get_collection('constant_float32_weights', scope='student/Model/layer%d'%(layer)+'/')
                        is_train_float32_weights_mask_collection = tf.get_collection('is_train_float32_weights_mask', scope='student/Model/layer%d'%(layer)+'/')
                        weights_collection = tf.get_collection('weights', scope='student/Model/layer%d'%(layer)+'/')
                        constant_weights = sess.run(constant_weights_collection[0])
                        is_train_float32_weights_mask = sess.run(is_train_float32_weights_mask_collection[0])
                        weights = sess.run(weights_collection[0])
                        print("{} -> {}" .format(weights_collection, np.sum(constant_weights*(1-is_train_float32_weights_mask)==weights)))
                """
        else:  
            test_result, test_accuracy, test_accuracy_top2, test_accuracy_top3, test_Y_pre = compute_accuracy(
                xs                      = xs, 
                ys                      = ys, 
                is_training             = is_training,
                is_quantized_activation = is_quantized_activation,
                is_fast_mode_dict       = {},
                Model_dict              = Model_dict,                
                QUANTIZED_NOW           = True, 
                prediction_list         = [prediction], 
                data_num                = test_data_num,
                BATCH_SIZE              = BATCH_SIZE, 
                sess                    = sess)
            print("\033[0;32mTesting Data Accuracy\033[0m = {test_Accuracy}".format(test_Accuracy = test_accuracy))
        
        #-----------------------#
        #   Saving best Model   #
        #-----------------------#
        if training_type == 'train':
            Dir = 'Model/'
            Dir = Dir + Model_first_name + '_Model/'
            Dir = Dir + Model_Name + '_best/'
        elif training_type == 'prune':
            Dir = testing_model_path
        elif training_type == 'rebuild':
            Dir = testing_model_path
        elif training_type == 'diversify':
            Dir = testing_model_path
            np.savetxt(Dir+"info.csv", test_accuracy_total, delimiter=",", fmt='%s')
            #with open(Dir + 'info.csv', 'wb') as csvfile:
            #    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            #    for acc in test_accuracy_total:
            #        csvwriter.writerow([str(acc)])
                    
        if is_find_best_model:
            if not os.path.isdir(Dir) or not os.path.exists(Dir+'info.csv'):
                print("Saving Best Model ...")
                save_path = saver.save(sess,  Dir + "best.ckpt")
                Model_csv_Generator(Model_dict, Dir + 'model')
                print("\033[0;35m{}\033[0m" .format(save_path))
                with open(Dir + 'info.csv', 'wb') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow([str(test_accuracy)])
                    csvwriter.writerow([testing_model_path + testing_model])
            else:   
                with open(Dir + 'info.csv', 'rb') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    best_accuracy = float(csvreader.next()[0])
                if test_accuracy >= best_accuracy:
                    print("Saving Best Model ...")
                    save_path = saver.save(sess,  Dir + "best.ckpt")
                    Model_csv_Generator(Model_dict, Dir + 'model')
                    print("\033[0;35m{}\033[0m" .format(save_path))
                    with open(Dir + 'info.csv', 'wb') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow([str(test_accuracy)])
                        csvwriter.writerow([testing_model_path + testing_model])
        
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
    
    #-----------------------------#
    #   Some Control Parameters   #
    #-----------------------------#
    IS_TERNARY = False
    IS_BINARY = False
    IS_QUANTIZED_ACTIVATION = False
    for layer in range(len(Model_dict)):
        if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
            IS_TERNARY = True
        if Model_dict['layer'+str(layer)]['IS_BINARY'] == 'TRUE':
            IS_BINARY = True
        if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
            IS_QUANTIZED_ACTIVATION = True

    #---------------#
    #    Dataset    #
    #---------------#
    print("Parsing Data ... ")
    ## -- Cifar10 --
    if Dataset == 'cifar10':
        filenames = [os.path.join(Dataset_Path, 'data_batch_%d.bin'%(i+1)) for i in range(5)]
    ## -- Imagenet ILSVRC_2012 --
    elif Dataset == 'ILSVRC2012':
        _NUM_TRAIN_FILES = 1024
        filenames = [os.path.join(Dataset_Path, 'train-%05d-of-01024' % i) for i in range(_NUM_TRAIN_FILES)]
    
    # tensor input
    xs, ys = input_fn(filenames, class_num, True, HP, Dataset)
    train_data_num = HP['train_num'] 
    print("\033[0;32mTrain Data Number\033[0m : {}" .format(train_data_num))
    
    #------------------#
    #    Placeholder   #
    #------------------#
    data_shape = [None, HP['H_Resize'], HP['W_Resize'], 3]
    global_step = tf.train.get_or_create_global_step()
    batches_per_epoch = train_data_num / HP['Batch_Size'] 
    ## -- is_training --
    is_training = tf.placeholder(tf.bool)
    
    ## -- learning Rate --
    learning_rate = tf.placeholder(tf.float32)

    ## -- is_quantized_activation --
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
        
    ## -- is_ternary --
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    ## -- is_binary --
    is_binary = {}
    for layer in range(len(Model_dict)):
        is_binary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    #-------------#
    #    Model    #
    #-------------# 
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
        is_binary               = is_binary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = HP['Dropout_Rate'],
        data_format             = data_format,
        reuse                   = None)
    
    ## -- Model Size --
    Model_Size = 0
    for iter, variable in enumerate(tf.trainable_variables()):
        if 'final_weights' not in variable.name and 'final_biases' not in variable.name:
            Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
            #print("{}, {}" .format(iter, variable))

    ## -- Collection --
    float32_weights_collection          = tf.get_collection("float32_weights"        , scope=None)
    float32_biases_collection           = tf.get_collection("float32_biases"         , scope=None)
    # Ternary
    ternary_weights_bd_collection       = tf.get_collection("ternary_weights_bd"     , scope=None)
    ternary_biases_bd_collection        = tf.get_collection("ternary_biases_bd"      , scope=None)
    # Binary
    binary_weights_bd_collection        = tf.get_collection("binary_weights_bd"     , scope=None)
    binary_biases_bd_collection         = tf.get_collection("binary_biases_bd"      , scope=None)     
    # assign ternary or float32 weights/biases to final weights/biases  
    assign_var_list_collection          = tf.get_collection("assign_var_list"        , scope=None)  
    # Actvation Quantization    
    float32_net_collection              = tf.get_collection("float32_net"            , scope=None)
    is_quantized_activation_collection  = tf.get_collection("is_quantized_activation", scope=None)
    mantissa_collection                 = tf.get_collection("mantissa"               , scope=None)
    fraction_collection                 = tf.get_collection("fraction"               , scope=None)
    # Gradient Update
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
    if IS_TERNARY and (Global_Epoch+1) >= HP['Ternary_Epoch']:
        loss = cross_entropy
    elif IS_BINARY and (Global_Epoch+1) >= HP['Binary_Epoch']:
        l2_norm   = tf.add_n([tf.reduce_sum(tf.square(tf.ones_like(v)-tf.abs(v))) for v in tf.trainable_variables()
                            if 'batch_normalization' not in v.name and 
                               'final_weights'       not in v.name and 
                               'final_biases'        not in v.name])
        l2_lambda = tf.constant(HP['L2_Lambda'])
        l2_norm   = tf.multiply(l2_lambda, l2_norm)
        loss = cross_entropy + l2_norm
    else:
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
        # Compute Gradients
        var_list = tf.trainable_variables()
        gra_and_var = opt.compute_gradients(loss, var_list = var_list)

    # Apply Gradients
    train_step  = opt.apply_gradients(gra_and_var, global_step)

    #-----------#
    #   Saver   #
    #-----------#
    saver = tf.train.Saver()
    TERNARY_NOW = False
    BINARY_NOW = False
    QUANTIZED_NOW = False
    
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
        if Global_Epoch == 0:
            print("Pruning ...")
            pruned_weights_info = load_obj(pruning_model_path, "pruned_info")
            pruning_propotion = HP['Pruning_Propotion'] / 100.
            kernel_values_per_layer = similar_group(inputs_and_kernels, sess)
           
            
            if HP['Pruning_Strategy'] != HP['Pruning_Strategy'].split('Filter_Similar')[0]:
                info = list(HP['Pruning_Strategy'].split('Filter_Similar')[1])
                
                if 'C' in info:
                    print("Connection", end = " / ")
                if 'B' in info:
                    print("Beta", end = " / ")
                if 'F' in info:
                    print("Skip_Fisrt_Layer", end = " / ")
                if 'L' in info:
                    print("Skip_Last_Layer", end = " / ")
                if 'D' in info:
                    print("Discard_Feature", end = " / ")
                if 'A' in info:
                    print("All_Prune", end = " / ")
                if 'W' in info:
                    print("Cut_Connection_From_Less_2_More", end = " / ")
                print("")

                pruned_weights_info = filter_prune_by_similarity(
                    prune_info_dict             = prune_info_dict, 
                    pruning_propotion           = pruning_propotion, 
                    pruned_weights_info         = pruned_weights_info, 
                    sess                        = sess,
                    is_connection               = 'C' in info,
                    is_beta                     = 'B' in info,
                    skip_first_layer            = 'F' in info,
                    skip_last_layer             = 'L' in info,
                    discard_feature             = 'D' in info,
                    all_be_pruned               = 'A' in info,
                    cut_connection_less_to_more = 'W' in info,
                    beta_threshold              = pow(10, -int(info[info.index('B')-1])) if 'B' in info else 0.01)
            else:
                print("\033[1;31mError\033[0m : No such strategy!")
                exit()
            """
            elif HP['Pruning_Strategy'] == 'Filter_Angle':
                if Model_first_name == 'MobileNet':
                    pruned_weights_info = mobileNet_filter_prune_by_angle(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
                elif Model_first_name == 'DenseNet':
                    pruned_weights_info = denseNet_filter_prune_by_angle(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
                else:
                    pruned_weights_info = filter_prune_by_angle(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
            elif HP['Pruning_Strategy'] == 'Filter_AngleII':
                if Model_first_name == 'DenseNet':
                    pruned_weights_info = denseNet_filter_prune_by_angleII(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
                else:
                    filter_prune_by_angleII(prune_info_dict, pruning_propotion, sess)
            elif HP['Pruning_Strategy'] == 'Filter_AngleIII':
                pruned_weights_info = filter_prune_by_angleIII(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
            elif HP['Pruning_Strategy'] == 'Filter_AngleIV':
                pruned_weights_info = filter_prune_by_angleIV(prune_info_dict, pruning_propotion, pruned_weights_info, sess)
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
            """
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
            Global_Epoch = Global_Epoch + 1

            ## -- Learning Rate --
            if HP['LR_Strategy'] == 'Normal':
                if Global_Epoch <= HP['LR_Decade_1st_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 0)
                elif Global_Epoch <= HP['LR_Decade_2nd_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 1)
                elif Global_Epoch <= HP['LR_Decade_3rd_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 2)
                elif Global_Epoch <= HP['LR_Decade_4th_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 3)
                else:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 4)
            
            ## -- Ternary Learning Rate --
            if IS_TERNARY and Global_Epoch >= HP['Ternary_Epoch']:
                if HP['Ternary_LR_Strategy'] == 'Normal':
                    if Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_1st_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 0)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_2nd_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 1)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_3rd_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 2)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_4th_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 3)
                    else:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 4)

            ## -- Binary Learning Rate --
            if IS_BINARY and Global_Epoch >= HP['Binary_Epoch']:
                if HP['Binary_LR_Strategy'] == 'Normal':
                    if Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_1st_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 0)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_2nd_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 1)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_3rd_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 2)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_4th_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 3)
                    else:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 4)
                
                elif HP['Binary_LR_Strategy'] == 'Exponential':
                    decay_rate = float(HP['Binary_LR_Final'] / HP['Binary_LR'])
                    decay_step = float(Global_Epoch-1-HP['Binary_Epoch']) / float(500-HP['Binary_Epoch'])
                    lr = HP['Binary_LR'] * pow(decay_rate, decay_step)

            ## -- Quantizad Activation --
            if IS_QUANTIZED_ACTIVATION and Global_Epoch == HP['Quantized_Activation_Epoch']:
                batch_xs = train_data[0:HP['Batch_Size']]
                # Calculate Each Activation's appropriate mantissa and fractional bit
                m, f = quantized_m_and_f(float32_net_collection, is_quantized_activation_collection, xs, Model_dict, batch_xs, sess)	
                # Assign mantissa and fractional bit to the tensor
                assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
                # Start Quantize Activation
                QUANTIZED_NOW = True
            
            ## -- Ternary --
            if IS_TERNARY and Global_Epoch == HP['Ternary_Epoch']:
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
                loss = cross_entropy
            
            ## -- Binary --
            if IS_BINARY and Global_Epoch >= HP['Binary_Epoch']:
                BINARY_NOW = True  
            
            ## -- Set feed_dict --
            # train_step
            feed_dict_train = {}
            for layer in range(len(Model_dict)):
                feed_dict_train.update({is_ternary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_train.update({is_binary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_BINARY']=='TRUE' and BINARY_NOW})
                feed_dict_train.update({is_quantized_activation['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION']=='TRUE' and QUANTIZED_NOW}) 
            
            # Assign float32 or ternary weight and biases to final weights
            feed_dict_assign = {}
            for layer in range(len(Model_dict)):
                feed_dict_assign.update({is_ternary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_assign.update({is_binary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_BINARY']=='TRUE' and BINARY_NOW})

            ## -- Training --
            total_correct_num = 0
            total_error_num = 0
            Train_loss = 0
            tStart = time.time()
            total_batch_iter = int(train_data_num / HP['Batch_Size'])
            for batch_iter in range(total_batch_iter):
                tStart_Batch = time.time()
                # Assign float32 or ternary weight and biases to final weights
                for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                    sess.run(assign_var_list, feed_dict = feed_dict_assign)
                
                # Run Training Step
                feed_dict_train.update({is_training: True, learning_rate: lr})
                _, Loss, Prediction, L2_norm, batch_ys = sess.run(
                    [train_step, loss, prediction, l2_norm, ys], 
                    feed_dict = feed_dict_train)
                
                # Result
                y_pre             = np.argmax(Prediction, -1)
                correct_num       = np.sum(np.equal(np.argmax(batch_ys, -1), y_pre) == True , dtype = np.float32)
                error_num         = np.sum(np.equal(np.argmax(batch_ys, -1), y_pre) == False, dtype = np.float32)
                batch_accuracy    = correct_num / (correct_num + error_num)
                total_correct_num = total_correct_num + correct_num
                total_error_num   = total_error_num + error_num
                Train_loss        = Train_loss + np.mean(Loss)  
                
                # Per Batch Info
                """
                tEnd_Batch = time.time()
                print("\033[1;34;40mEpoch\033[0m : %3d" %(Global_Epoch), end=" ")
                print("\033[1;34;40mData Iteration\033[0m : %7d" %(batch_iter*HP['Batch_Size']), end=" ")
                print("\033[1;32;40mBatch Accuracy\033[0m : %5f" %(batch_accuracy), end=" ")
                print("\033[1;32;40mLoss\033[0m : %5f" %(np.mean(Loss)), end=" ")
                print("\033[1;32;40mL2-norm\033[0m : %5f" %(L2_norm), end=" ")
                print("\033[1;32;40mLearning Rate\033[0m : %4f" %(lr), end=" ")
                print("(%2f sec)" %(tEnd_Batch-tStart_Batch))
                """
                
                # Per Class Accuracy
                """
                per_class_accuracy(Prediction, batch_ys)
                """
            
            # Assign float32 or ternary weight and biases to final weights
            for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                sess.run(assign_var_list, feed_dict = feed_dict_assign) 
            
            # Per Epoch Info
            tEnd = time.time()            
            Train_acc  = total_correct_num  / (total_correct_num + total_error_num)
            Train_loss = Train_loss / total_batch_iter 
            print("\r\033[0;33mEpoch{}\033[0m" .format(Global_Epoch), end = "")
            print(" (Cost {TIME} sec)" .format(TIME = tEnd - tStart))
            print("\033[0;32mLearning Rate    \033[0m : {}".format(lr))
            print("\033[0;32mTraining Accuracy\033[0m : {}".format(Train_acc))
            print("\033[0;32mTraining Loss    \033[0m : {} (l2_norm: {})".format(Train_loss, L2_norm))
            
            # train_info
            if epoch == 0:
                Train_acc_per_epoch = np.array([Train_acc])
                Train_loss_per_epoch = np.array([Train_loss])
            else:
                Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc])], axis=0)
                Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
            
            ## -- Saving Model --
            if (epoch+1) == HP['Epoch']:
                Dir = pruning_model_path[0:len(pruning_model_path)-1]
                Pruned_Size = 0.
                for iter, mask in enumerate(tf.get_collection('float32_weights_mask')):
                    Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)
                Pruning_Propotion_Now = int(Pruned_Size / Model_Size * 100)
                Dir = Dir.split('_' + HP['Pruning_Strategy'])[0] + '_' + HP['Pruning_Strategy'] + str(int(HP['Pruning_Propotion'])) + '_' + str(Pruning_Propotion_Now) + '/'
                if (not os.path.exists(Dir)):
                    print("\033[0;35m%s\033[0m is not exist!" %Dir)
                    print("\033[0;35m%s\033[0m is created!" %Dir)
                    os.makedirs(Dir)
                
                # Model.ckpt
                print("Saving Trained Weights ...")
                save_path = saver.save(sess, Dir + str(Global_Epoch) + ".ckpt")
                print("\033[0;35m{}\033[0m" .format(save_path))
                
                # Hyperparameter.csv
                np.savetxt(Dir + 'Hyperparameter.csv', HP_csv, delimiter=",", fmt="%s")
                
                # Model.csv
                Model_csv_Generator(Model_dict, Dir + 'model')
                
                # Analysis.csv
                #Save_Analyzsis_as_csv(Analysis, Dir + 'Analysis')
                
                # Computation.csv
                np.savetxt(Dir + 'computation.csv', np.array([computation]), delimiter=",", fmt="%d")
        
                # pruned_info.pkl
                if Global_Epoch == HP['Epoch']:
                    save_dict(pruned_weights_info, Dir, "pruned_info")
                
                # train_info.csv
                if Global_Epoch != HP['Epoch']:
                    with open(Dir + 'train_info.csv') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                        for i, row in enumerate(reader):
                            if i == 0:
                                Train_acc_per_epoch_ = np.array([float(row[0])])
                                Train_loss_per_epoch_ = np.array([float(row[1])])
                            elif i < Global_Epoch-HP['Epoch']:
                                Train_acc_per_epoch_ = np.concatenate([Train_acc_per_epoch_ , np.array([float(row[0])])], axis=0)
                                Train_loss_per_epoch_ = np.concatenate([Train_loss_per_epoch_, np.array([float(row[1])])], axis=0)
                                
                    Train_acc_per_epoch = np.concatenate([Train_acc_per_epoch_ , Train_acc_per_epoch], axis=0)
                    Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch_, Train_loss_per_epoch], axis=0)
                
                Train_acc_per_epoch = np.expand_dims(Train_acc_per_epoch, axis=1)
                Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
                Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
                np.savetxt(Dir + 'train_info.csv' , Train_info, delimiter=",", fmt="%f")
                
        # End Epoch
        tEnd_All = time.time()
        print("Total costs {TIME} sec\n" .format(TIME = tEnd_All - tStart_All))
    
    # Reset the tensorflow graph
    tf.reset_default_graph()
    
    # Update Model_path, Model
    Model_Path = Dir
    Model = str(Global_Epoch) + '.ckpt'
    
    return Model_Path, Model, Global_Epoch

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
    
    #-----------------------------#
    #   Some Control Parameters   #
    #-----------------------------#
    IS_TERNARY = False
    IS_BINARY = False
    IS_QUANTIZED_ACTIVATION = False
    for layer in range(len(Model_dict)):
        if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
            IS_TERNARY = True
        if Model_dict['layer'+str(layer)]['IS_BINARY'] == 'TRUE':
            IS_BINARY = True
        if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
            IS_QUANTIZED_ACTIVATION = True
    
    #---------------#
    #    Dataset    #
    #---------------#
    print("Parsing Data ... ")
    ## -- Cifar10 --
    if Dataset == 'cifar10':
        filenames = [os.path.join(Dataset_Path, 'data_batch_%d.bin'%(i+1)) for i in range(5)]
    ## -- Imagenet ILSVRC_2012 --
    elif Dataset == 'ILSVRC2012':
        _NUM_TRAIN_FILES = 1024
        filenames = [os.path.join(Dataset_Path, 'train-%05d-of-01024' % i) for i in range(_NUM_TRAIN_FILES)]
    
    # tensor input
    xs, ys = input_fn(filenames, class_num, True, HP, Dataset)
    train_data_num = HP['train_num'] 
    print("\033[0;32mTrain Data Number\033[0m : {}" .format(train_data_num))
    
    #------------------#
    #    Placeholder   #
    #------------------#
    data_shape = [None, HP['H_Resize'], HP['W_Resize'], 3]
    global_step = tf.train.get_or_create_global_step()
    batches_per_epoch = train_data_num / HP['Batch_Size'] 
    ## -- is_training --
    is_training = tf.placeholder(tf.bool)
    
    ## -- learning Rate --
    learning_rate = tf.placeholder(tf.float32)

    ## -- is_quantized_activation --
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
        
    ## -- is_ternary --
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    ## -- is_binary --
    is_binary = {}
    for layer in range(len(Model_dict)):
        is_binary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    #-------------#
    #    Model    #
    #-------------# 
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
        is_binary               = is_binary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = HP['Dropout_Rate'],
        data_format             = data_format,
        reuse                   = None)
    
    ## -- Model Size --
    Model_Size = 0
    for iter, variable in enumerate(tf.trainable_variables()):
        if 'final_weights' not in variable.name and 'final_biases' not in variable.name:
            Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
            #print("{}, {}" .format(iter, variable))

    ## -- Collection --
    float32_weights_collection          = tf.get_collection("float32_weights"        , scope=None)
    float32_biases_collection           = tf.get_collection("float32_biases"         , scope=None)
    # Ternary
    ternary_weights_bd_collection       = tf.get_collection("ternary_weights_bd"     , scope=None)
    ternary_biases_bd_collection        = tf.get_collection("ternary_biases_bd"      , scope=None)
    # Binary
    binary_weights_bd_collection        = tf.get_collection("binary_weights_bd"     , scope=None)
    binary_biases_bd_collection         = tf.get_collection("binary_biases_bd"      , scope=None)     
    # assign ternary or float32 weights/biases to final weights/biases  
    assign_var_list_collection          = tf.get_collection("assign_var_list"        , scope=None)  
    # Actvation Quantization    
    float32_net_collection              = tf.get_collection("float32_net"            , scope=None)
    is_quantized_activation_collection  = tf.get_collection("is_quantized_activation", scope=None)
    mantissa_collection                 = tf.get_collection("mantissa"               , scope=None)
    fraction_collection                 = tf.get_collection("fraction"               , scope=None)
    # Gradient Update
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
    if IS_TERNARY and (Global_Epoch+1) >= HP['Ternary_Epoch']:
        loss = cross_entropy
    elif IS_BINARY and (Global_Epoch+1) >= HP['Binary_Epoch']:
        l2_norm   = tf.add_n([tf.reduce_sum(tf.square(tf.ones_like(v)-tf.abs(v))) for v in tf.trainable_variables()
                            if 'batch_normalization' not in v.name and 
                               'final_weights'       not in v.name and 
                               'final_biases'        not in v.name])
        l2_lambda = tf.constant(HP['L2_Lambda'])
        l2_norm   = tf.multiply(l2_lambda, l2_norm)
        loss = cross_entropy + l2_norm
    else:
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
        # Compute Gradients
        var_list = tf.trainable_variables()
        gra_and_var = opt.compute_gradients(loss, var_list = var_list)

    # Apply Gradients
    train_step  = opt.apply_gradients(gra_and_var, global_step)

    #-----------#
    #   Saver   #
    #-----------#
    saver = tf.train.Saver()
    TERNARY_NOW = False
    BINARY_NOW = False
    QUANTIZED_NOW = False
    
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
        if HP['mode'] == '1' or HP['mode'] == '2':
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
        
        #----------------------------#
        #    Constatnt Model Size    #
        #----------------------------#
        Constant_Size = 0
        for iter, mask in enumerate(tf.get_collection('is_train_float32_weights_mask')):
            Constant_Size = Constant_Size + np.sum(sess.run(mask) == 0)
        print("\033[0;36m=======================\033[0m")
        print("\033[0;36m Constant Size\033[0m = {}" .format(Constant_Size))
        print("\033[0;36m=======================\033[0m")
        
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
            Global_Epoch = Global_Epoch + 1
            
            ## -- Learning Rate --
            if HP['LR_Strategy'] == 'Normal':
                if Global_Epoch <= HP['LR_Decade_1st_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 0)
                elif Global_Epoch <= HP['LR_Decade_2nd_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 1)
                elif Global_Epoch <= HP['LR_Decade_3rd_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 2)
                elif Global_Epoch <= HP['LR_Decade_4th_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 3)
                else:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 4)
            
            ## -- Ternary Learning Rate --
            if IS_TERNARY and Global_Epoch >= HP['Ternary_Epoch']:
                if HP['Ternary_LR_Strategy'] == 'Normal':
                    if Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_1st_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 0)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_2nd_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 1)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_3rd_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 2)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_4th_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 3)
                    else:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 4)

            ## -- Binary Learning Rate --
            if IS_BINARY and Global_Epoch >= HP['Binary_Epoch']:
                if HP['Binary_LR_Strategy'] == 'Normal':
                    if Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_1st_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 0)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_2nd_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 1)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_3rd_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 2)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_4th_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 3)
                    else:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 4)
                
                elif HP['Binary_LR_Strategy'] == 'Exponential':
                    decay_rate = float(HP['Binary_LR_Final'] / HP['Binary_LR'])
                    decay_step = float(Global_Epoch-1-HP['Binary_Epoch']) / float(500-HP['Binary_Epoch'])
                    lr = HP['Binary_LR'] * pow(decay_rate, decay_step)
            
            ## -- Quantizad Activation --
            if IS_QUANTIZED_ACTIVATION and Global_Epoch == HP['Quantized_Activation_Epoch']:
                batch_xs = train_data[0:HP['Batch_Size']]
                # Calculate Each Activation's appropriate mantissa and fractional bit
                m, f = quantized_m_and_f(float32_net_collection, is_quantized_activation_collection, xs, Model_dict, batch_xs, sess)	
                # Assign mantissa and fractional bit to the tensor
                assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
                # Start Quantize Activation
                QUANTIZED_NOW = True
                
            ## -- Ternary --
            if IS_TERNARY and Global_Epoch == HP['Ternary_Epoch']:
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
                loss = cross_entropy
            ## -- Binary --
            if IS_BINARY and Global_Epoch >= HP['Binary_Epoch']:
                BINARY_NOW = True  
                
            ## -- Set feed_dict --
            # train_step
            feed_dict_train = {}
            for layer in range(len(Model_dict)):
                feed_dict_train.update({is_ternary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_train.update({is_binary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_BINARY']=='TRUE' and BINARY_NOW})
                feed_dict_train.update({is_quantized_activation['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION']=='TRUE' and QUANTIZED_NOW}) 
            # Assign float32 or ternary weight and biases to final weights
            feed_dict_assign = {}
            for layer in range(len(Model_dict)):
                feed_dict_assign.update({is_ternary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_assign.update({is_binary['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_BINARY']=='TRUE' and BINARY_NOW})
            
            ## -- Training --
            total_correct_num = 0
            total_error_num = 0
            Train_loss = 0
            tStart = time.time()
            total_batch_iter = int(train_data_num / HP['Batch_Size'])
            for batch_iter in range(total_batch_iter):
                tStart_Batch = time.time()
                # Assign float32 or ternary weight and biases to final weights
                for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                    sess.run(assign_var_list, feed_dict = feed_dict_assign)
                
                # Run Training Step
                feed_dict_train.update({is_training: True, learning_rate: lr})
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
                
                # Per Batch Info
                """
                tEnd_Batch = time.time()
                print("\033[1;34;40mEpoch\033[0m : %3d" %(Global_Epoch), end=" ")
                print("\033[1;34;40mData Iteration\033[0m : %7d" %(batch_iter*HP['Batch_Size']), end=" ")
                print("\033[1;32;40mBatch Accuracy\033[0m : %5f" %(batch_accuracy), end=" ")
                print("\033[1;32;40mLoss\033[0m : %5f" %(np.mean(Loss)), end=" ")
                print("\033[1;32;40mL2-norm\033[0m : %5f" %(L2_norm), end=" ")
                print("\033[1;32;40mLearning Rate\033[0m : %4f" %(lr), end=" ")
                print("(%2f sec)" %(tEnd_Batch-tStart_Batch))
                """
                
                # Per Class Accuracy
                """
                per_class_accuracy(Prediction, batch_ys)
                """
            # Assign float32 or ternary weight and biases to final weights
            for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                sess.run(assign_var_list, feed_dict = feed_dict_assign)
                
            # Per Epoch Info
            tEnd = time.time()            
            Train_acc  = total_correct_num  / (total_correct_num + total_error_num)
            Train_loss = Train_loss / total_batch_iter 
            print("\r\033[0;33mEpoch{}\033[0m" .format(Global_Epoch), end = "")
            print(" (Cost {TIME} sec)" .format(TIME = tEnd - tStart))
            print("\033[0;32mLearning Rate    \033[0m : {}".format(lr))
            print("\033[0;32mTraining Accuracy\033[0m : {}".format(Train_acc))
            print("\033[0;32mTraining Loss    \033[0m : {} (l2_norm: {})".format(Train_loss, L2_norm))

            # train_info
            if epoch == 0:
                Train_acc_per_epoch = np.array([Train_acc])
                Train_loss_per_epoch = np.array([Train_loss])
            else:
                Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc])], axis=0)
                Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
            
            ## -- Saving Model --
            if (epoch+1) == HP['Epoch']:
                # Update the constant weights to trainable weights
                for iter in range(len(is_train_mask_collection)):
                    constant_weights_value = base_mask[iter] * base_weights[iter]
                    float32_weights = float32_weights_collection[iter]
                    float32_weights_value = (1. - base_mask[iter]) * sess.run(float32_weights) + constant_weights_value
                    sess.run(tf.assign(float32_weights, float32_weights_value))
            
                # Dir
                Dir = rebuilding_model_path_base[0:len(rebuilding_model_path_base)-1]
                Pruned_Size = 0.
                for iter, mask in enumerate(tf.get_collection('float32_weights_mask')):
                    Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)
                Pruning_Propotion_Now = int(Pruned_Size / Model_Size * 100)
                if len(Dir.split('Re')) == 1:
                    Dir = Dir + '_' + 'Re' + str(Pruning_Propotion_Now) + '/'
                else:
                    Dir = Dir + '_' + str(Pruning_Propotion_Now) + '/'
                if (not os.path.exists(Dir)):
                    print("\033[0;35m%s\033[0m is not exist!" %Dir)
                    print("\033[0;35m%s\033[0m is created!" %Dir)
                    os.makedirs(Dir)
                
                # Model.ckpt
                print("Saving Trained Weights ...")
                save_path = saver.save(sess, Dir + str(Global_Epoch) + ".ckpt")
                print("\033[0;35m{}\033[0m" .format(save_path))
                
                # Hyperparameter.csv
                np.savetxt(Dir + 'Hyperparameter.csv', HP_csv, delimiter=",", fmt="%s")
                
                # Model.csv
                Model_csv_Generator(Model_dict, Dir + 'model')
                
                # Analysis.csv
                #Save_Analyzsis_as_csv(Analysis, Dir + 'Analysis')
                
                # Computation.csv
                np.savetxt(Dir + 'computation.csv', np.array([computation]), delimiter=",", fmt="%d")
                
                # pruned_info.pkl
                if Global_Epoch == HP['Epoch']:
                    save_dict(pruned_weights_info, Dir, "pruned_info")
                
                # train_info.csv
                if Global_Epoch != HP['Epoch']:
                    with open(Dir + 'train_info.csv') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                        for i, row in enumerate(reader):
                            if i == 0:
                                Train_acc_per_epoch_ = np.array([float(row[0])])
                                Train_loss_per_epoch_ = np.array([float(row[1])])
                            elif i < Global_Epoch-HP['Epoch']:
                                Train_acc_per_epoch_ = np.concatenate([Train_acc_per_epoch_ , np.array([float(row[0])])], axis=0)
                                Train_loss_per_epoch_ = np.concatenate([Train_loss_per_epoch_, np.array([float(row[1])])], axis=0)
                    Train_acc_per_epoch = np.concatenate([Train_acc_per_epoch_ , Train_acc_per_epoch], axis=0)
                    Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch_, Train_loss_per_epoch], axis=0)
                
                Train_acc_per_epoch = np.expand_dims(Train_acc_per_epoch, axis=1)
                Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
                Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
                np.savetxt(Dir + 'train_info.csv' , Train_info, delimiter=",", fmt="%f")
            
            # Debug
            constant_weights_collection = tf.get_collection('constant_float32_weights', scope = None)
            is_train_mask_collection = tf.get_collection('is_train_float32_weights_mask', scope = None)
            final_weights_collection = tf.get_collection('weights', scope = None)
            for c in range(len(constant_weights_collection)):
                constant_weights = sess.run(constant_weights_collection[c])
                final_weights = sess.run(final_weights_collection[c])
                is_train_mask = sess.run(is_train_mask_collection[c])
                #print(np.mean((1.-is_train_mask) * (constant_weights-final_weights)))
                assert np.mean((1.-is_train_mask) * (constant_weights-final_weights)) == 0., "Final weights must be equal to constant weights!"
                
        # End Epoch
        tEnd_All = time.time()
        print("Total costs {TIME} sec\n" .format(TIME = tEnd_All - tStart_All))

    tf.reset_default_graph()
        
    Model_Path = Dir
    Model = str(Global_Epoch) + '.ckpt'
    
    return Model_Path, Model, Global_Epoch

def Diversifying(
    # Model
    Model_dict              ,
    Model_dict_s            ,
    # Dataset               
    Dataset	                ,
    Dataset_Path            ,
    Y_pre_Path              ,
    class_num               ,
    # Parameter	    
    diversify_layer         ,
    HP                      ,
    Global_Epoch            ,
    weights_bd_ratio        ,
    biases_bd_ratio         ,
    # Model Save            
    HP_csv                  ,
    Model_first_name        ,
    Model_second_name       ,
    # Pre-trained Weights   
    teacher_model_path      ,
    teacher_model           ,
    # Be-trained Weights    
    student_model_path      ,
    student_model           ,
    # Mask Weights
    pruned_model_path       ,
    pruned_model            ,
    # Rebuilding
    PR_iter                 ,
    # Hyperparameter Opt
    IS_HYPERPARAMETER_OPT   ,
    # 
    error
    ):
    
    Model_Name = Model_first_name + '_' + Model_second_name
    
    #-----------------------------#
    #   Some Control Parameters   #
    #-----------------------------#
    IS_TERNARY = False
    IS_BINARY = False
    IS_QUANTIZED_ACTIVATION = False
    for layer in range(len(Model_dict)):
        if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
            IS_TERNARY = True
        if Model_dict['layer'+str(layer)]['IS_BINARY'] == 'TRUE':
            IS_BINARY = True
        if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
            IS_QUANTIZED_ACTIVATION = True

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
    #xs, ys = cifar10.distorted_inputs()
    train_data_num = HP['train_num'] 
    print("\033[0;32mTrain Data Number\033[0m : {}" .format(train_data_num))
    
    #------------------#
    #    Placeholder   #
    #------------------#
    data_shape = [None, HP['H_Resize'], HP['W_Resize'], 3]
    global_step = tf.train.get_or_create_global_step()
    batches_per_epoch = train_data_num / HP['Batch_Size'] 
    ## Is_training
    is_training = {}
    for layer in range(len(Model_dict)):
        is_training.update({'layer%d'%layer : tf.placeholder(tf.bool)})
        
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
    
    ## is_binary
    is_binary = {}
    for layer in range(len(Model_dict)):
        is_binary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    ## is_fast_mode
    is_fast_mode = {}
    for layer in range(len(Model_dict)):
        is_fast_mode.update({'layer%d'%layer : tf.placeholder(tf.bool, name='is_fast_mode_layer'+str(layer))})
    
    ## complexity_mode
    complexity_mode = {}
    for layer in range(len(Model_dict)):
        complexity_mode.update({'layer%d'%layer : tf.placeholder(tf.float32, name='complexity_mode_layer'+str(layer))})

    
    #----------------------#
    #    Building Model    #
    #----------------------#
    print("Building Model ...")
    data_format = "NCHW"
    ## -- Deivce --
    GPUs = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    device_num = len(GPUs)
    #batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
    #      [xs, ys], capacity=2 * device_num)
    
    ## -- Optimizer --
    if HP['Opt_Method']=='Adam':
        opt = tf.train.AdamOptimizer(learning_rate, HP['Momentum_Rate'])
    elif HP['Opt_Method']=='Momentum':
        opt = tf.train.MomentumOptimizer(
            learning_rate = learning_rate, 
            momentum      = HP['Momentum_Rate'])
    if device_num > 1:
        opt = tf.contrib.estimator.TowerOptimizer(opt)
    
    ## -- model --
    net = xs
    net_ = tf.split(net, num_or_size_splits=device_num, axis=0)
    labels = ys
    labels_ = tf.split(labels, num_or_size_splits=device_num, axis=0)
    Model_dict_ = copy.deepcopy(Model_dict)
    Model_dict_s_ = copy.deepcopy(Model_dict_s)
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for d, device in enumerate(['/device:GPU:%s' %(GPU) for GPU in range(device_num)]):
            with tf.device(device):
                print(device)
                """
                ## -- Dequeues one batch for the GPU --
                image_batch, label_batch = batch_queue.dequeue()
                label_batch = tf.one_hot(label_batch, 10)
                """
                image_batch = net_[d]
                label_batch = labels_[d]
                
                ## -- Build Model --
                # Update Model_dict_s
                if diversify_layer['type'] == 'restoring':
                    for layer in diversify_layer['layer']:
                        Model_dict_s[layer].update({'IS_DIVERSITY': 'FALSE'})
                        Model_dict_s[layer].update({'DIVERSIFY_MODE': None})
                else:
                    for layer in diversify_layer['layer']:
                        # IS_DIVERSITY
                        if diversify_layer['mode'] == '0':
                            Model_dict_s[layer].update({'IS_DIVERSITY': 'TRUE'})
                        else:
                            Model_dict_s[layer].update({'IS_DIVERSITY': 'FALSE'})
                            
                    # DIVERSIFY_MODE
                    if diversify_layer['mode'] == '0':
                        for middle_layer in diversify_layer['layer']:
                            Model_dict_s[middle_layer].update({'DIVERSIFY_MODE': diversify_layer['mode']})
                    elif diversify_layer['mode'] == '1':
                        for middle_layer in diversify_layer['layer']:
                            Model_dict_s[middle_layer].update({'DIVERSIFY_MODE': '2'})
                        Model_dict_s[diversify_layer['layer'][0]].update({'DIVERSIFY_MODE': '1'})
                        Model_dict_s[diversify_layer['layer'][-1]].update({'DIVERSIFY_MODE': '3'})
                        Model_dict_s[diversify_layer['layer'][-1]].update({'IS_DIVERSITY': 'TRUE'})
                    elif diversify_layer['mode'] == '4':
                        for middle_layer in diversify_layer['layer']:
                            Model_dict_s[middle_layer].update({'DIVERSIFY_MODE': '5'})
                        Model_dict_s[diversify_layer['layer'][0]].update({'DIVERSIFY_MODE': '4'})
                        Model_dict_s[diversify_layer['layer'][-1]].update({'DIVERSIFY_MODE': '6'})
                    elif diversify_layer['mode'] == '7':
                        for middle_layer in diversify_layer['layer']:
                            Model_dict_s[middle_layer].update({'DIVERSIFY_MODE': '8'})
                        Model_dict_s[diversify_layer['layer'][0]].update({'DIVERSIFY_MODE': '7'})
                        Model_dict_s[diversify_layer['layer'][-1]].update({'DIVERSIFY_MODE': '9'})
                        #Model_dict_s[diversify_layer['layer'][-1]].update({'IS_DIVERSITY': 'TRUE'})
                    
                    # REBUILING_MODE_MAX
                    if diversify_layer['mode'] == '4':
                        for middle_layer in diversify_layer['layer']:
                            Model_dict_s[middle_layer].update({'REBUILING_MODE_MAX': str(PR_iter+1)})
                            
                # Student Model
                complexity_mode_now = PR_iter + 1 if diversify_layer['mode'] == '4' else int(diversify_layer['times'])-PR_iter
                with tf.variable_scope('student'):
                    prediction_s, train_net_s, Analysis, max_parameter, inputs_and_kernels, prune_info_dict_s = Model_dict_Decoder_DiversifyVersion(
                    net                     = image_batch, 
                    Model_dict              = Model_dict_s,
                    diversify_layer         = diversify_layer,
                    is_testing              = False,
                    is_student              = True,
                    is_restoring            = diversify_layer['type'] == 'restoring',
                    complexity_mode         = complexity_mode,
                    is_training             = is_training,
                    is_ternary              = is_ternary,
                    is_binary               = is_binary,
                    is_quantized_activation = is_quantized_activation,
                    is_fast_mode            = is_fast_mode,
                    complexity_mode_now     = complexity_mode_now,
                    DROPOUT_RATE            = HP['Dropout_Rate'],
                    data_format             = data_format,
                    reuse                   = None)
                
                # Teacher Model
                prediction, train_net, _, _, _, _ = Model_dict_Decoder_DiversifyVersion(
                net                     = image_batch, 
                Model_dict              = Model_dict_,
                diversify_layer         = diversify_layer,
                is_testing              = False,
                is_student              = False,
                is_restoring            = False,
                complexity_mode         = complexity_mode,
                is_training             = is_training,
                is_ternary              = is_ternary,
                is_binary               = is_binary,
                is_quantized_activation = is_quantized_activation,
                is_fast_mode            = is_fast_mode,
                complexity_mode_now     = complexity_mode_now,
                DROPOUT_RATE            = HP['Dropout_Rate'],
                data_format             = data_format,
                reuse                   = None)
                                
                ## -- Loss --
                # L2 Regularization
                l2_norm   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                            if 'batch_normalization' not in v.name])
                l2_lambda = tf.constant(HP['L2_Lambda'])
                l2_norm   = tf.multiply(l2_lambda, l2_norm)
                
                # Cross Entropy
                if diversify_layer['layer'][-1] == 'layer'+str(len(Model_dict)-1):
                    cross_entropy = tf.losses.softmax_cross_entropy(
                        onehot_labels = label_batch,
                        logits        = train_net_s)
                
                # Mean-Square Error
                mean_square = tf.losses.mean_squared_error(
                    labels      = train_net,
                    predictions = train_net_s)
                
                root_mean_square = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(train_net, train_net_s))))
                
                # Total Loss
                if IS_TERNARY and (Global_Epoch+1) >= HP['Ternary_Epoch']:
                    loss = cross_entropy
                elif IS_BINARY and (Global_Epoch+1) >= HP['Binary_Epoch']:
                    l2_norm   = tf.add_n([tf.reduce_sum(tf.square(tf.ones_like(v)-tf.abs(v))) for v in tf.trainable_variables()
                                        if 'float32_weights' in v.name])
                    l2_lambda = tf.constant(HP['L2_Lambda'])
                    l2_norm   = tf.multiply(l2_lambda, l2_norm)
                    loss = cross_entropy + l2_norm
                else:
                    if diversify_layer['layer'][-1] == 'layer'+str(len(Model_dict)-1):
                        loss = cross_entropy + l2_norm
                    else:
                        loss = mean_square #+ l2_norm
                    
                ## Setting variable to reuse mode
                tf.get_variable_scope().reuse_variables()
                
                ## -- Gradient --
                # Batch norm requires update ops to be added as a dependency to the train_op
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Compute Gradients
                    var_list = tf.trainable_variables()
                    gra_and_var = opt.compute_gradients(loss, var_list = var_list)
                    
                    # Delete 'None' gradient element in gra_and_var list
                    gv_iter = 0
                    while(1):
                        if gra_and_var[gv_iter][0] == None:
                            gra_and_var.pop(gv_iter)
                        else:
                            gv_iter = gv_iter + 1
                            
                        if gv_iter >= len(gra_and_var):
                            break
                #tower_grads.append(gra_and_var)
    
    # Apply Gradients
    #gra_and_vars = average_gradients(tower_grads)
    train_step  = opt.apply_gradients(gra_and_var, global_step)

    ## -- Model Size --
    Model_Size = 0
    trainable_variables = []
    for iter, variable in enumerate(tf.trainable_variables()):
        if 'final_weights' not in variable.name and 'final_biases' not in variable.name:
            Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
            # See all your variables in termainl	
            print("{}, {}" .format(iter, variable))
            trainable_variables.append(variable.name)
            
    print("\033[0;36m=======================\033[0m")
    print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size))
    print("\033[0;36m=======================\033[0m")
    
    # For Loading trained Model
    all_variables = []
    i = 0
    for iter, variable in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=None)):
        # See all your variables in termainl	
        if not any("mask" in s for s in variable.name.split('_:/')) and not any("constant" in s for s in variable.name.split('_:/')):
            #print("{}, {}" .format(i, variable))
            all_variables.append([variable.name, str(variable.get_shape().as_list())])
            i = i + 1
            
    ## -- Collection --
    float32_weights_collection          = tf.get_collection("float32_weights"           , scope=None)
    float32_biases_collection           = tf.get_collection("float32_biases"            , scope=None)
    float32_weights_fast_collection     = tf.get_collection("float32_weights_fast"      , scope=None)
    float32_biases_fast_collection      = tf.get_collection("float32_biases_fast"       , scope=None)
    ## Ternary  
    ternary_weights_bd_collection       = tf.get_collection("ternary_weights_bd"        , scope=None)
    ternary_biases_bd_collection        = tf.get_collection("ternary_biases_bd"         , scope=None)
    ## Binary   
    binary_weights_bd_collection        = tf.get_collection("binary_weights_bd"         , scope=None)
    binary_biases_bd_collection         = tf.get_collection("binary_biases_bd"          , scope=None)
    clip_weights_collection             = tf.get_collection("clip_weights"              , scope=None)
    ## assign ternary or float32 weights/biases to final weights/biases     
    assign_var_list_collection          = tf.get_collection("assign_var_list"           , scope=None)  
    ## Actvation Quantization       
    float32_net_collection              = tf.get_collection("float32_net"               , scope=None)
    is_quantized_activation_collection  = tf.get_collection("is_quantized_activation"   , scope=None)
    mantissa_collection                 = tf.get_collection("mantissa"                  , scope=None)
    fraction_collection                 = tf.get_collection("fraction"                  , scope=None)
    quantized_net_collection            = tf.get_collection("quantized_net"             , scope=None)
    ## Gradient Update  
    var_list_collection                 = tf.get_collection("var_list"                  , scope=None)
    float32_params                      = tf.get_collection("float32_params"            , scope=None) 
    ## Final weights
    weights_collection                  = tf.get_collection("weights"                   , scope=None)
    biases_collection                   = tf.get_collection("biases"                    , scope=None)

    #-----------#
    #   Saver   #
    #-----------#	
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Model/'))
    saver_s = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='student/'))
    TERNARY_NOW = False
    BINARY_NOW = False
    QUANTIZED_NOW = False
    
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
        
        # Start the queue runners.
        #tf.train.start_queue_runners(sess=sess)
        
        #--------------------------#
        #   Load trained weights   #
        #--------------------------#
        if teacher_model_path!=None and teacher_model!=None:
            print("Loading the trained weights (teacher) ... ")
            print("\033[0;35m{}\033[0m" .format(teacher_model_path + teacher_model))
            save_path = saver.restore(sess, teacher_model_path + teacher_model)
        
        print("Loading the trained weights (student) ... ")
        print("\033[0;35m{}\033[0m" .format(student_model_path + student_model))
        if Global_Epoch != 0:# or PR_iter != 0:
            save_path = saver_s.restore(sess, student_model_path + student_model)
        else:
            checkpoint_path = os.path.join(student_model_path, student_model)
            reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            if pruned_model_path != None and pruned_model != None:
                pruned_checkpoint_path = os.path.join(pruned_model_path, pruned_model)
                pruned_reader = pywrap_tensorflow.NewCheckpointReader(pruned_checkpoint_path)
                pruned_var_to_shape_map = pruned_reader.get_variable_to_shape_map()
            for j, s in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='student')): 
                found = False
                layer = str(s.name.split('/')[2])
                cond_0 = True #s.name.split('/')[-1] == 'float32_weights_mask:0'
                cond_1 = 'fast' in s.name.split('/')
                cond_2 = diversify_layer['mode'] == '1'
                cond_3 = layer in diversify_layer['layer']
                if cond_0 and cond_1 and cond_2 and cond_3:
                    for key in sorted(pruned_var_to_shape_map.keys()):
                        #if key.split('/')[-1] == 'Momentum':
                        #    print(key)
                        value = pruned_reader.get_tensor(key)
                        if teacher_model_path == student_model_path:
                            name = s.name.split('student/')[-1]
                        else:
                            name = s.name
                        #name = name.split('_fast')[0] + name.split('_fast')[1]
                        name = name.split('fast/')[0] + name.split('fast/')[1]
                        if name == key + ':0':
                            #print('[\033[1;32mO\033[0m] %s \033[1;32m->\033[0m %s' %(s.name, key))
                            found = True
                            sess.run(tf.assign(s, value))
                            #break
                else:
                    for key in sorted(var_to_shape_map.keys()):
                        value = reader.get_tensor(key)
                        if teacher_model_path == student_model_path:
                            name = s.name.split('student/')[-1]
                        else:
                            name = s.name
                        if name == key + ':0':
                            #print('%s \033[1;32m->\033[0m %s' %(key, s.name))
                            found = True
                            sess.run(tf.assign(s, value))
                            break
                if found:
                    #print('[\033[1;32mO\033[0m] %s \033[1;32m->\033[0m %s' %(s.name, key))
                    None
                else:
                    print('[\033[1;31mX\033[0m] %s \033[1;32m->\033[0m \033[1;33m%s\033[0m' %(s.name, 'Not Found'))
         
        #---------------#
        #    Pruning    #
        #---------------#
        if Global_Epoch == 0 and (diversify_layer['mode'] == '1' or diversify_layer['mode'] == '4'):
            print("Pruning ...")
            pruning_model_path = student_model_path
            pruned_weights_info = load_obj(pruning_model_path, "pruned_info")
            pruning_propotion = diversify_layer['prune_propotion']
            
            if HP['Pruning_Strategy'] != HP['Pruning_Strategy'].split('Filter_Similar')[0]:
                info = list(HP['Pruning_Strategy'].split('Filter_Similar')[1])
                
                if 'C' in info:
                    print("Connection", end = " / ")
                if 'B' in info:
                    print("Beta", end = " / ")
                if 'F' in info:
                    print("Skip_Fisrt_Layer", end = " / ")
                if 'L' in info:
                    print("Skip_Last_Layer", end = " / ")
                if 'D' in info:
                    print("Discard_Feature", end = " / ")
                if 'A' in info:
                    print("All_Prune", end = " / ")
                if 'W' in info:
                    print("Cut_Connection_From_Less_2_More", end = " / ")
                print("")

                pruned_weights_info = filter_prune_by_similarity(
                    prune_info_dict             = prune_info_dict_s, 
                    pruning_propotion           = pruning_propotion, 
                    pruned_weights_info         = pruned_weights_info, 
                    sess                        = sess,
                    is_connection               = 'C' in info,
                    is_beta                     = 'B' in info,
                    skip_first_layer            = 'F' in info,
                    skip_last_layer             = 'L' in info,
                    discard_feature             = 'D' in info,
                    all_be_pruned               = 'A' in info,
                    cut_connection_less_to_more = 'W' in info,
                    beta_threshold              = pow(10, -int(info[info.index('B')-1])) if 'B' in info else 0.01)
            else:
                print("\033[1;31mError\033[0m : No such strategy!")
                exit()
            
            # Update Complexity Mode Mask
            if diversify_layer['mode'] == '4':
                print("Updating Complexity Mode Mask ...")
                for layer in diversify_layer['layer']:
                    if Model_dict[layer]['type'] == 'CONV':
                        # Pruned Mask
                        pruned_mask = tf.get_collection(
                            key = 'float32_weights_mask',
                            scope = "student/Model/" + layer + '/')
                        pruned_mask_value = sess.run(pruned_mask[0])
                        
                        # Complexity Mode Mask
                        complexity_mode_mask = tf.get_collection(
                            key = 'complexity_mode_mask',
                            scope = "student/Model/" + layer + '/')
                        complexity_mode_mask_value = sess.run(complexity_mode_mask[0])
                        
                        # New Complexity Mode Mask
                        new_complexity_mode_mask_value = complexity_mode_mask_value + pruned_mask_value
                        sess.run(tf.assign(complexity_mode_mask[0], new_complexity_mode_mask_value))
                        #print(pruned_mask)
                        #print(complexity_mode_mask)
        
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
            Global_Epoch = Global_Epoch + 1
            
            ## -- Learning Rate --
            if HP['LR_Strategy'] == 'Normal':
                if Global_Epoch <= HP['LR_Decade_1st_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 0)
                elif Global_Epoch <= HP['LR_Decade_2nd_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 1)
                elif Global_Epoch <= HP['LR_Decade_3rd_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 2)
                elif Global_Epoch <= HP['LR_Decade_4th_Epoch']:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 3)
                else:
                    lr = HP['LR'] / pow(HP['LR_Decade'], 4)
            
            ## -- Ternary Learning Rate --
            if IS_TERNARY and Global_Epoch >= HP['Ternary_Epoch']:
                if HP['Ternary_LR_Strategy'] == 'Normal':
                    if Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_1st_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 0)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_2nd_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 1)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_3rd_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 2)
                    elif Global_Epoch <= HP['Ternary_Epoch'] + HP['Ternary_LR_Decade_4th_Epoch']:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 3)
                    else:
                        lr = HP['Ternary_LR'] / pow(HP['Ternary_LR_Decade'], 4)

            ## -- Binary Learning Rate --
            if IS_BINARY and Global_Epoch >= HP['Binary_Epoch']:
                if HP['Binary_LR_Strategy'] == 'Normal':
                    if Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_1st_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 0)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_2nd_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 1)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_3rd_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 2)
                    elif Global_Epoch <= HP['Binary_Epoch'] + HP['Binary_LR_Decade_4th_Epoch']:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 3)
                    else:
                        lr = HP['Binary_LR'] / pow(HP['Binary_LR_Decade'], 4)
            
            ## -- Quantizad Activation --
            if IS_QUANTIZED_ACTIVATION and Global_Epoch == HP['Quantized_Activation_Epoch']:
                batch_xs = train_data[0:HP['Batch_Size']]
                # Calculate Each Activation's appropriate mantissa and fractional bit
                m, f = quantized_m_and_f(float32_net_collection, is_quantized_activation_collection, xs, Model_dict, batch_xs, sess)	
                # Assign mantissa and fractional bit to the tensor
                assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
                # Start Quantize Activation
                QUANTIZED_NOW = True
            
            ## -- Ternary --
            if IS_TERNARY and Global_Epoch >= HP['Ternary_Epoch']:
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
            
            ## -- Binary --
            if IS_BINARY and Global_Epoch >= HP['Binary_Epoch']:
                BINARY_NOW = True
                
            ## -- Set feed_dict --
            # Show Diversify Layer
            print('Diversifying Layer :', end=" ")
            for layer in diversify_layer['layer']:
                 print('\033[0;36m%s\033[0m,' %(layer), end = " ")
            print("")
            
            ## -- feed_dict_train --
            feed_dict_train = {}
            for layer in range(len(Model_dict_s)):
                feed_dict_train.update({is_training['layer'+str(layer)]: True}) #'layer'+str(layer) in diversify_layer['layer']
                feed_dict_train.update({is_ternary['layer'+str(layer)]: Model_dict_s['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_train.update({is_binary['layer'+str(layer)]: Model_dict_s['layer'+str(layer)]['IS_BINARY']=='TRUE' and BINARY_NOW})
                feed_dict_train.update({is_quantized_activation['layer'+str(layer)]: Model_dict_s['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION']=='TRUE' and QUANTIZED_NOW}) 
                # is_fast_mode
                if Model_dict_s['layer'+str(layer)]['IS_DIVERSITY']=='TRUE':
                    if 'layer'+str(layer) in diversify_layer['layer']:
                        feed_dict_train.update({is_fast_mode['layer'+str(layer)]: True})
                    else:
                        rand_mode = random.randint(0, 1) == 0
                        feed_dict_train.update({is_fast_mode['layer'+str(layer)]: rand_mode})
                        if rand_mode:
                            print("%s: \033[1;31m%d\033[0m" %('layer'+str(layer), 0), end = " ")
                        else:
                            print("%s: \033[1;34m%d\033[0m" %('layer'+str(layer), 1), end = " ")
                else:
                    feed_dict_train.update({is_fast_mode['layer'+str(layer)]: False})
                # Complexity Mode
                # (Training Layer)
                if diversify_layer['mode'] == '4' and 'layer'+str(layer) in diversify_layer['layer']:
                    feed_dict_train.update({complexity_mode['layer'+str(layer)]: float(PR_iter)})
                elif diversify_layer['mode'] == '7' and 'layer'+str(layer) in diversify_layer['layer']:
                    feed_dict_train.update({complexity_mode['layer'+str(layer)]: float(diversify_layer['times'])-float(PR_iter)})
                # (Trained Layer)
                for layer in range(len(Model_dict_s)):
                    if not 'layer'+str(layer) in diversify_layer['layer']:
                        # Pruning
                        if Model_dict_s['layer'+str(layer)]['DIVERSIFY_MODE'] == '4':
                            feed_dict_train.update({complexity_mode['layer'+str(layer)]: float(Model_dict_s['layer'+str(layer)]['REBUILING_MODE_MAX'])-1.})
                        elif Model_dict_s['layer'+str(layer)]['DIVERSIFY_MODE'] == '5':
                            feed_dict_train.update({complexity_mode['layer'+str(layer)]: float(Model_dict_s['layer'+str(layer)]['REBUILING_MODE_MAX'])-1.})
                        elif Model_dict_s['layer'+str(layer)]['DIVERSIFY_MODE'] == '6':
                            feed_dict_train.update({complexity_mode['layer'+str(layer)]: float(Model_dict_s['layer'+str(layer)]['REBUILING_MODE_MAX'])-1.})
                        # Rebuilding
                        elif Model_dict_s['layer'+str(layer)]['DIVERSIFY_MODE'] == '7':
                            rand_mode = float(random.randint(0, int(Model_dict_s['layer'+str(layer)]['REBUILING_MODE_MAX'])))
                            feed_dict_train.update({complexity_mode['layer'+str(layer)]: rand_mode})
                            feed_dict_assign.update({complexity_mode['layer'+str(layer)]: rand_mode})
                            layer_start = 'layer'+str(layer)
                        elif Model_dict_s['layer'+str(layer)]['DIVERSIFY_MODE'] == '8':
                            feed_dict_train.update({complexity_mode['layer'+str(layer)]: rand_mode})
                            feed_dict_assign.update({complexity_mode['layer'+str(layer)]: rand_mode})
                        elif Model_dict_s['layer'+str(layer)]['DIVERSIFY_MODE'] == '9':
                            feed_dict_train.update({complexity_mode['layer'+str(layer)]: rand_mode})
                            feed_dict_assign.update({complexity_mode['layer'+str(layer)]: rand_mode})
                            layer_end = 'layer'+str(layer)
                            print("%s_to_%s: \033[1;31m%d\033[0m" %(layer_start, layer_end, int(rand_mode)), end = " ")
                            
            if diversify_layer['mode'] == '4':
                print("Complexity_mode: \033[0;36m%d\033[0m" %(PR_iter), end = " ")
            elif diversify_layer['mode'] == '7':
                print("Complexity_mode: \033[0;36m%d\033[0m" %(int(diversify_layer['times'])-PR_iter), end = " ")
            print("")
            
            ## -- feed_dict_assign --
            feed_dict_assign = {}
            for layer in range(len(Model_dict_s)):
                feed_dict_assign.update({is_ternary['layer'+str(layer)]: Model_dict_s['layer'+str(layer)]['IS_TERNARY']=='TRUE' and TERNARY_NOW})
                feed_dict_assign.update({is_binary['layer'+str(layer)]: Model_dict_s['layer'+str(layer)]['IS_BINARY']=='TRUE' and BINARY_NOW})
                # Complexity Mode
                # (Training Layer)
                if diversify_layer['mode'] == '7' and 'layer'+str(layer) in diversify_layer['layer']:
                    feed_dict_assign.update({complexity_mode['layer'+str(layer)]: float(diversify_layer['times'])-float(PR_iter)})
                
            ## -- Training --
            total_correct_num = 0
            total_error_num = 0
            Train_loss = 0
            tStart = time.time()
            tStart_Batch = time.time()
            total_batch_iter = int(train_data_num / HP['Batch_Size'])
            for batch_iter in range(total_batch_iter):                
                # Assign
                for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                    sess.run(assign_var_list, feed_dict = feed_dict_assign)
                
                # Run Training Step
                feed_dict_train.update({learning_rate: lr})
                ####################################################
                _, Loss, Prediction, L2_norm, batch_ys, quantized_net, Gra_and_var = sess.run(
                    [train_step, loss, prediction_s, l2_norm, labels, quantized_net_collection, gra_and_var], 
                    feed_dict = feed_dict_train)
                #####################################################
                #Loss, Prediction, L2_norm, batch_ys, quantized_net = sess.run(
                #    [loss, prediction_s, l2_norm, labels, quantized_net_collection],
                #    feed_dict = feed_dict_train)
                #####################################################
                
                # Result
                y_pre             = np.argmax(Prediction, -1)
                correct_num       = np.sum(np.equal(np.argmax(batch_ys, -1), y_pre) == True , dtype = np.float32)
                error_num         = np.sum(np.equal(np.argmax(batch_ys, -1), y_pre) == False, dtype = np.float32)
                batch_accuracy    = correct_num / (correct_num + error_num)
                total_correct_num = total_correct_num + correct_num
                total_error_num   = total_error_num + error_num
                Train_loss        = Train_loss + np.mean(Loss)
                
                # Per Batch Info
                """
                tEnd_Batch = time.time()
                print("\033[1;34;40mEpoch\033[0m : %3d" %(Global_Epoch), end=" ")
                print("\033[1;34;40mData Iteration\033[0m : %7d" %(batch_iter*HP['Batch_Size']), end=" ")
                print("\033[1;32;40mBatch Accuracy\033[0m : %5f" %(batch_accuracy), end=" ")
                print("\033[1;32;40mLoss\033[0m : %5f" %(np.mean(Loss)), end=" ")
                print("\033[1;32;40mL2-norm\033[0m : %5f" %(L2_norm), end=" ")
                print("\033[1;32;40mLearning Rate\033[0m : %4f" %(lr), end=" ")
                print("(%2f sec)" %(tEnd_Batch-tStart_Batch))
                """
                
                # Per Class Accuracy
                """
                per_class_accuracy(Prediction, batch_ys)
                """
                
            # Assign
            for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
                sess.run(assign_var_list, feed_dict = feed_dict_assign)
            
            tEnd = time.time()            
            Train_acc  = total_correct_num  / (total_correct_num + total_error_num)
            Train_loss = Train_loss / total_batch_iter 
            print("\r\033[0;33mEpoch{}\033[0m" .format(Global_Epoch), end = "")
            print(" (Cost {TIME} sec)" .format(TIME = tEnd - tStart))
            print("\033[0;32mLearning Rate    \033[0m : {}".format(lr))
            print("\033[0;32mTraining Accuracy\033[0m : {}".format(Train_acc))
            print("\033[0;32mTraining Loss    \033[0m : {} (l2_norm: {})".format(Train_loss, L2_norm))
            
            # Training Failed
            if Train_acc < 0.15 or math.isnan(L2_norm):
                error = True
                break
                
            # train_info
            if epoch == 0:
                Train_acc_per_epoch = np.array([Train_acc])
                Train_loss_per_epoch = np.array([Train_loss])
            else:
                Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc])], axis=0)
                Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
            
            ## -- Update Constant Weights --
            if (epoch+1) == HP['Epoch']:
                if diversify_layer['mode'] == '4' :
                    print("Updating Constant Weights ...")
                    for layer in diversify_layer['layer']:
                        if Model_dict[layer]['type'] == 'CONV':
                            # Trainable Weights
                            trainable_weights = tf.get_collection(
                                key = 'float32_weights',
                                scope = "student/Model/" + layer + '/')
                            trainable_weights_value = sess.run(trainable_weights[0])
                            
                            # Pruned Mask
                            pruned_mask = tf.get_collection(
                                key = 'float32_weights_mask',
                                scope = "student/Model/" + layer + '/')
                            pruned_mask_value = sess.run(pruned_mask[0])
                            
                            # Constant Weights
                            constant_weights = tf.get_collection(
                                key = 'constant_float32_weights',
                                scope = "student/Model/" + layer + '/')
                            constant_weights_value = trainable_weights_value * pruned_mask_value
                            
                            sess.run(tf.assign(constant_weights[0], constant_weights_value))
                            #print(pruned_mask)
                            #print(trainable_weights)
                            #print(constant_weights)
                            
                if diversify_layer['mode'] == '7' :
                    print("Updating Constant Weights ...")
                    for layer in diversify_layer['layer']:
                        if Model_dict[layer]['type'] == 'CONV':
                            # Trainable Weights
                            trainable_weights = tf.get_collection(
                                key = 'float32_weights',
                                scope = "student/Model/" + layer + '/')
                            trainable_weights_value = sess.run(trainable_weights[0])
                            
                            # Pruned Mask
                            pruned_mask = tf.get_collection(
                                key = 'float32_weights_mask',
                                scope = "student/Model/" + layer + '/')
                            pruned_mask_value = sess.run(pruned_mask[0])
                            
                            # Constant Mask
                            constant_mask = tf.subtract(
                                x = tf.constant(1, tf.float32),
                                y = tf.get_collection(
                                       key = 'is_train_float32_weights_mask',
                                       scope = "student/Model/" + layer + '/')[0])
                            constant_mask_value = sess.run(constant_mask)
                            
                            # Constant Weights
                            constant_weights = tf.get_collection(
                                key = 'constant_float32_weights',
                                scope = "student/Model/" + layer + '/')
                            constant_weights_value = sess.run(constant_weights[0])
                            
                            old_constant_weights = constant_mask_value * constant_weights_value
                            new_constant_weights = (1. - constant_mask_value) * trainable_weights_value * pruned_mask_value
                            constant_weights_value = old_constant_weights + new_constant_weights
                            sess.run(tf.assign(constant_weights[0], constant_weights_value))
                            
                            #print(trainable_weights)
                            #print(pruned_mask)
                            #print(constant_mask)
                            #print(constant_weights)
                            
            ## -- Saving Model --
            if (epoch+1) == HP['Epoch']:
                # Dir
                if (not os.path.exists('Model/'+Model_first_name + '_Model/')) :
                    print("\033[0;35m%s\033[0m is not exist!" %'Model/'+Model_first_name)
                    os.mkdir(Model_first_name)
                    print("\033[0;35m%s\033[0m is created!" %'Model/'+Model_first_name)
                
                Dir = 'Model/' + Model_first_name + '_Model/'
                Dir = Dir + Model_Name + '_'
                Dir = Dir + diversify_layer['layer'][0] + '_to_' + diversify_layer['layer'][-1]
                if diversify_layer['mode'] == '1' or diversify_layer['mode'] == '4':
                    Pruned_Size = 0.
                    for layer in diversify_layer['layer']:
                        mask = tf.get_collection('float32_weights_mask', scope = 'student/Model/'+layer+'/')[0]
                        Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)
                        if np.sum(sess.run(mask) == 0) != 0:
                            print("{} -> {}" .format(mask, np.sum(sess.run(mask) == 0)))
                    Pruning_Propotion_Now = int(Pruned_Size / Model_Size * 100)
                    Dir = Dir + '_' + 'mode' + diversify_layer['mode']
                    Dir = Dir + '_' + HP['Pruning_Strategy'] + str(int(diversify_layer['prune_propotion']*100)) + '_' + str(Pruning_Propotion_Now)
                if diversify_layer['mode'] == '7':
                    Pruned_Size = 0.
                    for layer in diversify_layer['layer']:
                        if Model_dict_s[layer]['type'] == 'CONV':
                            mask = tf.get_collection('float32_weights_mask', scope = 'student/Model/'+layer+'/')[0]
                            Pruned_Size = Pruned_Size + np.sum(sess.run(mask) == 0)
                            if np.sum(sess.run(mask) == 0) != 0:
                                print("{} -> {}" .format(mask, np.sum(sess.run(mask) == 0)))
                    Pruning_Propotion_Now = int(Pruned_Size / Model_Size * 100)
                    Dir = Dir + '_' + 'mode' + diversify_layer['mode']
                    #Dir = Dir + '_' + str(Pruning_Propotion_Now)
                Dir = Dir + '/'
                
                if (not os.path.exists(Dir)):
                    print("\033[0;35m%s\033[0m is not exist!" %Dir)
                    print("\033[0;35m%s\033[0m is created!" %Dir)
                    os.makedirs(Dir)
                
                # Model.ckpt
                print("Saving Trained Weights ...")
                save_path = saver_s.save(sess, Dir + str(Global_Epoch) + ".ckpt")
                print("\033[0;35m{}\033[0m" .format(save_path))
                
                # Hyperparameter.csv
                np.savetxt(Dir + 'Hyperparameter.csv', HP_csv, delimiter=",", fmt="%s")
                
                # Model.csv
                Model_csv_Generator(Model_dict_s, Dir + 'model')
                
                # Analysis.csv
                #Save_Analyzsis_as_csv(Analysis, Dir + 'Analysis')
                
                # Computation.csv
                np.savetxt(Dir + 'computation.csv', np.array([computation]), delimiter=",", fmt="%d")
        
                # train_info.csv
                if Global_Epoch != HP['Epoch']:
                    with open(Dir + 'train_info.csv') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                        for i, row in enumerate(reader):
                            if i == 0:
                                Train_acc_per_epoch_ = np.array([float(row[0])])
                                Train_loss_per_epoch_ = np.array([float(row[1])])
                            elif i < Global_Epoch-HP['Epoch']:
                                Train_acc_per_epoch_ = np.concatenate([Train_acc_per_epoch_ , np.array([float(row[0])])], axis=0)
                                Train_loss_per_epoch_ = np.concatenate([Train_loss_per_epoch_, np.array([float(row[1])])], axis=0)
                                
                    Train_acc_per_epoch = np.concatenate([Train_acc_per_epoch_ , Train_acc_per_epoch], axis=0)
                    Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch_, Train_loss_per_epoch], axis=0)
                
                Train_acc_per_epoch = np.expand_dims(Train_acc_per_epoch, axis=1)
                Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
                Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
                np.savetxt(Dir + 'train_info.csv' , Train_info, delimiter=",", fmt="%f")
        
        # End Epoch
        tEnd_All = time.time()
        print("Total costs {TIME} sec\n" .format(TIME = tEnd_All - tStart_All))
    
    # Reset the tensorflow graph
    tf.reset_default_graph()
    
    if error:
        return student_model_path, student_model, error, Global_Epoch-epoch-1
    else:
        Model_Path = Dir
        Model = str(Global_Epoch) + '.ckpt'
        return Model_Path, Model, error, Global_Epoch

    
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
    layer_num = len(HP_tmp['layer'])
    
    #------------------------#
    #    Model Definition    #
    #------------------------#
    Model_dict = {}
    for layer in range(layer_num):
        HP = {}		
        for iter, key in enumerate(keys):
            if key == 'output_channel' and layer == (layer_num-1):
                HP.update({'output_channel': class_num})
            else:
                HP.update({key: HP_tmp[key][layer]})
        if not 'IS_BINARY' in keys:
            HP.update({'IS_BINARY': 'FALSE'})    
        if not 'IS_DIVERSITY' in keys:
            HP.update({'IS_DIVERSITY': 'FALSE'}) 
        if not 'DIVERSIFY_MODE' in keys:
            HP.update({'DIVERSIFY_MODE': None})
        if not 'REBUILING_MODE_MAX' in keys:
            HP.update({'REBUILING_MODE_MAX': None}) 
        Model_dict.update({'layer'+str(layer):HP})
    
    return Model_dict	

def Hyperparameter_Decoder( # No Use
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
    is_binary,
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
                                is_binary               = is_binary[shortcut_layer]                                ,
                                is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                                IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = shortcut_input_layer['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                padding                 = "SAME",			 
                                data_format             = data_format,
                                Analysis                = Analysis)
                    
                    # Add the shortcut and net
                    #with tf.variable_scope('layer%d' %(layer)):
                    if shortcut_input_layer['shortcut_connection'] == "ADD":
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> ADD")
                        net = tf.add(net, shortcut)
                        
                        # Activation Quantization
                        if layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
                            net = Model.quantize_Module(net, is_quantized_activation['layer'+str(layer)], data_format)
        
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
                                is_binary               = is_binary[shortcut_layer]                                ,
                                is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                                IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = shortcut_input_layer['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                padding                 = "SAME",			 
                                data_format             = data_format,
                                Analysis                = Analysis)            
                    
                    # Concatenate the shortcut and net
                    if shortcut_input_layer['shortcut_connection'] == "CONCAT":
                        with tf.variable_scope('layer%d' %(layer)):
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
                    elif layer_now['Activation'] == 'HardTanh':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> HardTanh")
                        net = tf.clip_by_value(net, -1, 1)
                    elif layer_now['Activation'] == 'PReLU':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> PReLU")
                        net = tf.keras.layers.PReLU()(net)
                    
                    # For Pruning
                    prune_info_dict['layer%d'%(layer-1)].update({
                        'beta': [beta for beta in tf.trainable_variables(scope="Model/layer%d/"%layer) 
                                 if 'beta' in beta.name][0]})
                    
                    # Activation Quantization
                    if layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Quantized Activation ")
                        net = Model.quantize_Module(net, is_quantized_activation['layer'+str(layer)], data_format)
                #------------------#
                #    Activation    #
                #------------------#
                if layer_now['type'] == 'ACT':
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
                    elif layer_now['Activation'] == 'HardTanh':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> HardTanh")
                        net = tf.clip_by_value(net, -1, 1)
                    elif layer_now['Activation'] == 'PReLU':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> PReLU")
                        net = tf.keras.layers.PReLU()(net) 
                    
                    # Activation Quantization
                    if layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Quantized Activation ")
                        net = Model.quantize_Module(net, is_quantized_activation['layer'+str(layer)], data_format)
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
                        is_binary               = is_binary['layer'+str(layer)]                 ,
                        is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                        IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                        IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
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
                        is_binary               = is_binary['layer'+str(layer)]                 ,
                        is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                        IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                        IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
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
                            {'layer%d'%layer: {'weights': tf.get_collection("weights", scope="Model/layer%d/"%layer)[0]}}
                        )
                    else:
                        prune_info_dict['layer%d'%layer].update(
                            {'weights': tf.get_collection("weights", scope="Model/layer%d/"%layer)[0],
                             
                            }
                        )
                    tf.trainable_variables(scope="Model/layer%d/"%layer)
                    prune_info_dict['layer%d'%layer].update({
                        'mask'        : tf.get_collection("float32_weights_mask", scope="Model/layer%d/"%layer)[0],
                        'outputs'     : tf.get_collection("conv_outputs"  , scope="Model/layer%d/"%layer)[0],
                        'stride'      : int(layer_now['stride']),
                        'is_shortcut' : is_shortcut_past_layer,
                        'is_depthwise': layer_now['is_depthwise'] == 'TRUE'
                    })
                    
                    if layer_now['is_batch_norm'] == 'TRUE':
                        prune_info_dict['layer%d'%layer].update({
                            'beta'        : [beta for beta in tf.trainable_variables(scope="Model/layer%d/"%layer) 
                                            if 'beta' in beta.name][0]
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
    
    Model = np.array(['layer'                   ,
                      'type'                    ,
                      'kernel_size'             ,
                      'stride'                  ,
                      'internal_channel'        ,
                      'output_channel'          ,
                      'rate'                    ,
                      'group'                   ,
                      'is_add_biases'           ,
                      'is_shortcut'             ,
                      'shortcut_destination'    ,
                      'is_projection_shortcut'  ,
                      'shortcut_type'           ,
                      'shortcut_connection'     ,
                      'is_batch_norm'           ,
                      'is_dilated'              ,
                      'is_depthwise'            ,
                      'IS_TERNARY'              ,
                      'IS_BINARY'               ,
                      'IS_QUANTIZED_ACTIVATION' ,
                      'IS_DIVERSITY'            ,
                      'DIVERSIFY_MODE'          ,
                      'REBUILING_MODE_MAX'      ,
                      'Activation'              ,
                      'indice'                  ,
                      'scope'                   ])     
                    
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

def Model_dict_Decoder_DiversifyVersion(
    net,
    diversify_layer,
    is_testing,
    Model_dict,
    is_student,
    is_restoring,
    complexity_mode,
    complexity_mode_now,
    is_training,
    is_ternary,
    is_binary,
    is_quantized_activation,
    is_fast_mode,
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
    fast_routine = 0
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
                            
                            ## -- Fast --
                            trainable = False
                            if not is_testing:
                                if shortcut_layer in diversify_layer['layer'] and is_student:
                                    trainable = True
                            
                            with tf.variable_scope('fast'):
                                if layer_now['DIVERSIFY_MODE'] == '2' and Model_dict[shortcut_layer]['DIVERSIFY_MODE'] == '2':
                                    shortcut_fast = layer_now['shortcut_input_fast'][str(shortcut_index)]
                                    shortcut_fast = Model.shortcut_Module(
                                        net                     = shortcut_fast,
                                        group                   = int(shortcut_input_layer['group']),
                                        destination             = net_fast,
                                        initializer             = tf.contrib.layers.variance_scaling_initializer(),
                                        is_training             = is_training[shortcut_layer],
                                        is_add_biases           = shortcut_input_layer['is_add_biases']           == 'TRUE',
                                        is_projection_shortcut  = shortcut_input_layer['is_projection_shortcut']  == 'TRUE',
                                        shortcut_type           = shortcut_input_layer['shortcut_type']                    ,
                                        shortcut_connection     = shortcut_input_layer['shortcut_connection']              ,
                                        is_batch_norm           = shortcut_input_layer['is_batch_norm']           == 'TRUE',
                                        is_ternary              = is_ternary[shortcut_layer]                               ,
                                        is_binary               = is_binary[shortcut_layer]                                ,
                                        is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                                        IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE',
                                        IS_BINARY               = shortcut_input_layer['IS_BINARY']               == 'TRUE',
                                        IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                        padding                 = "SAME",			 
                                        data_format             = data_format,
                                        Analysis                = Analysis,
                                        trainable               = trainable)
                                elif layer_now['DIVERSIFY_MODE'] == '3' and Model_dict[shortcut_layer]['DIVERSIFY_MODE'] == '2':
                                    shortcut_fast = layer_now['shortcut_input_fast'][str(shortcut_index)]
                                    shortcut_fast = Model.shortcut_Module(
                                        net                     = shortcut_fast,
                                        group                   = int(shortcut_input_layer['group']),
                                        destination             = net_fast,
                                        initializer             = tf.contrib.layers.variance_scaling_initializer(),
                                        is_training             = is_training[shortcut_layer],
                                        is_add_biases           = shortcut_input_layer['is_add_biases']           == 'TRUE',
                                        is_projection_shortcut  = shortcut_input_layer['is_projection_shortcut']  == 'TRUE',
                                        shortcut_type           = shortcut_input_layer['shortcut_type']                    ,
                                        shortcut_connection     = shortcut_input_layer['shortcut_connection']              ,
                                        is_batch_norm           = shortcut_input_layer['is_batch_norm']           == 'TRUE',
                                        is_ternary              = is_ternary[shortcut_layer]                               ,
                                        is_binary               = is_binary[shortcut_layer]                                ,
                                        is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                                        IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE',
                                        IS_BINARY               = shortcut_input_layer['IS_BINARY']               == 'TRUE',
                                        IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                        padding                 = "SAME",			 
                                        data_format             = data_format,
                                        Analysis                = Analysis,
                                        trainable               = trainable)
                            
                            ## -- Normal --
                            trainable = False
                            if not is_testing:
                                if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                                    if layer_now['DIVERSIFY_MODE'] == '4':
                                        trainable = True
                                    elif layer_now['DIVERSIFY_MODE'] == '5':
                                        trainable = True
                                    elif layer_now['DIVERSIFY_MODE'] == '6':
                                        trainable = True
                                    elif is_restoring:
                                        trainable = True
                                        
                            shortcut = Model.shortcut_Module(
                                net                     = shortcut,
                                group                   = int(shortcut_input_layer['group']),
                                destination             = net,
                                initializer             = tf.contrib.layers.variance_scaling_initializer(),
                                is_training             = is_training[shortcut_layer],
                                is_add_biases           = shortcut_input_layer['is_add_biases']           == 'TRUE',
                                is_projection_shortcut  = shortcut_input_layer['is_projection_shortcut']  == 'TRUE',
                                shortcut_type           = shortcut_input_layer['shortcut_type']                    ,
                                shortcut_connection     = shortcut_input_layer['shortcut_connection']              ,
                                is_batch_norm           = shortcut_input_layer['is_batch_norm']           == 'TRUE',
                                is_ternary              = is_ternary[shortcut_layer]                               ,
                                is_binary               = is_binary[shortcut_layer]                                ,
                                is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                                IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = shortcut_input_layer['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                padding                 = "SAME",			 
                                data_format             = data_format,
                                Analysis                = Analysis,
                                trainable               = trainable)
                    
                    # Add the shortcut and net
                    if shortcut_input_layer['shortcut_connection'] == "ADD":
                        ## -- Fast --
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.add(net, shortcut)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                if Model_dict[shortcut_layer]['DIVERSIFY_MODE'] == '2':
                                    net_fast = tf.add(net_fast, shortcut_fast)
                                else:
                                    net_fast = tf.add(net_fast, shortcut)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                if Model_dict[shortcut_layer]['DIVERSIFY_MODE'] == '2':
                                    net_fast = tf.add(net_fast, shortcut_fast)
                                else:
                                    net_fast = tf.add(net_fast, shortcut)
                        ## -- Normal --
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> ADD")
                        net = tf.add(net, shortcut)                        
                        
                        # Activation Quantization
                        if layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
                            net = Model.quantize_Module(net, is_quantized_activation['layer'+str(layer)], data_format)
        
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
                for shortcut_destination in layer_now['shortcut_destination'].split('/'):
                    if bool(Model_dict[shortcut_destination].get('shortcut_num')):
                        shortcut_num = Model_dict[shortcut_destination]['shortcut_num'] + 1
                        Model_dict[shortcut_destination].update({'shortcut_num':shortcut_num})
                        Model_dict[shortcut_destination]['shortcut_input'].update({str(shortcut_num-1):net})
                        Model_dict[shortcut_destination]['shortcut_input_layer'].update({str(shortcut_num-1):'layer%d' %(layer)})
                        
                        ## -- Fast --
                        #if layer_now['DIVERSIFY_MODE'] == '1':
                        #    Model_dict[shortcut_destination]['shortcut_input_fast'].update({str(shortcut_num-1):net})
                        if layer_now['DIVERSIFY_MODE'] == '2':
                            Model_dict[shortcut_destination]['shortcut_input_fast'].update({str(shortcut_num-1):net_fast})
                        #elif layer_now['DIVERSIFY_MODE'] == '3':
                        #    Model_dict[shortcut_destination]['shortcut_input_fast'].update({str(shortcut_num-1):net_fast})
                        #else:
                        #    Model_dict[shortcut_destination].update({'shortcut_input_fast': {str(shortcut_num-1):net}})
                    else:
                        shortcut_num = 1
                        Model_dict[shortcut_destination].update({'shortcut_num':shortcut_num})
                        Model_dict[shortcut_destination].update({'shortcut_input':{str(shortcut_num-1):net}})
                        Model_dict[shortcut_destination].update({'shortcut_input_layer':{str(shortcut_num-1):'layer%d' %(layer)}})
                        
                        ## -- Fast --
                        #if layer_now['DIVERSIFY_MODE'] == '1':
                        #    Model_dict[shortcut_destination].update({'shortcut_input_fast': {str(shortcut_num-1):net}})
                        if layer_now['DIVERSIFY_MODE'] == '2':
                            Model_dict[shortcut_destination].update({'shortcut_input_fast': {str(shortcut_num-1):net_fast}})
                        #elif layer_now['DIVERSIFY_MODE'] == '3':
                        #    Model_dict[shortcut_destination].update({'shortcut_input_fast': {str(shortcut_num-1):net_fast}})
                        #else:
                        #    Model_dict[shortcut_destination].update({'shortcut_input_fast': {str(shortcut_num-1):net}})
            
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
                                is_training             = is_training['layer'+str(layer)],
                                is_add_biases           = shortcut_input_layer['is_add_biases']           == 'TRUE',
                                is_projection_shortcut  = shortcut_input_layer['is_projection_shortcut']  == 'TRUE',
                                shortcut_type           = shortcut_input_layer['shortcut_type']                    ,
                                shortcut_connection     = shortcut_input_layer['shortcut_connection']              ,
                                is_batch_norm           = shortcut_input_layer['is_batch_norm']           == 'TRUE',
                                is_ternary              = is_ternary[shortcut_layer]                               ,
                                is_binary               = is_binary[shortcut_layer]                                ,
                                is_quantized_activation = is_quantized_activation[shortcut_layer]                  ,
                                IS_TERNARY              = shortcut_input_layer['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = shortcut_input_layer['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = shortcut_input_layer['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                padding                 = "SAME",			 
                                data_format             = data_format,
                                Analysis                = Analysis) 
                            
                    # Concatenate the shortcut and net
                    if shortcut_input_layer['shortcut_connection'] == "CONCAT":
                        with tf.variable_scope('layer%d' %(layer)):
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
                    
                    ## -- Fast --
                    with tf.variable_scope('fast'):
                        if layer_now['DIVERSIFY_MODE'] == '1':
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast if bool(layer_now.get('shortcut_num')) else net, [0, 2, 3, 1])
                            net_fast, indices_fast, output_shape_fast = Model.indice_pool( 
                                net_fast, 
                                kernel_size             = int(layer_now['kernel_size']), 
                                stride                  = int(layer_now['stride']), 
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 3, 1, 2])
                            if layer_now['indice'] != 'None' and layer_now['indice'] != 'FALSE':
                                Model_dict[layer_now['indice']].update({'indice_fast':indices_fast, 'output_shape_fat':output_shape_fast})
                        elif layer_now['DIVERSIFY_MODE'] == '2':
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 2, 3, 1])
                            net_fast, indices_fast, output_shape_fast = Model.indice_pool( 
                                net_fast, 
                                kernel_size             = int(layer_now['kernel_size']), 
                                stride                  = int(layer_now['stride']), 
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 3, 1, 2])
                            if layer_now['indice'] != 'None' and layer_now['indice'] != 'FALSE':
                                Model_dict[layer_now['indice']].update({'indice_fast':indices_fast, 'output_shape_fat':output_shape_fast})
                        elif layer_now['DIVERSIFY_MODE'] == '3':
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 2, 3, 1])
                            net_fast, indices_fast, output_shape_fast = Model.indice_pool( 
                                net_fast, 
                                kernel_size             = int(layer_now['kernel_size']), 
                                stride                  = int(layer_now['stride']), 
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 3, 1, 2])
                            if layer_now['indice'] != 'None' and layer_now['indice'] != 'FALSE':
                                Model_dict[layer_now['indice']].update({'indice_fast':indices_fast, 'output_shape_fat':output_shape_fast})
                    
                    ## -- Normal --
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
                    
                    # Diversify layer
                    if layer_now['DIVERSIFY_MODE'] == '3':
                        net = tf.cond(is_fast_mode['layer'+str(layer)], 
                                    lambda: net_fast,
                                    lambda: net)
                    
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
                    
                    ## -- Fast --
                    with tf.variable_scope('fast'):
                        if layer_now['DIVERSIFY_MODE'] == '1':
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast if bool(layer_now.get('shortcut_num')) else net, [0, 2, 3, 1])
                            net_fast = Model.indice_unpool( 
                                net_fast,  
                                output_shape = layer_now['output_shape_fast'], 
                                indices      = layer_now['indice_fast'], 
                                scope        = layer_now['scope'])
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 3, 1, 2])
                        elif layer_now['DIVERSIFY_MODE'] == '2':
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 2, 3, 1])
                            net_fast = Model.indice_unpool( 
                                net_fast,  
                                output_shape = layer_now['output_shape_fast'], 
                                indices      = layer_now['indice_fast'], 
                                scope        = layer_now['scope'])
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 3, 1, 2])
                        elif layer_now['DIVERSIFY_MODE'] == '3':
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 2, 3, 1])
                            net_fast = Model.indice_unpool( 
                                net_fast,  
                                output_shape = layer_now['output_shape_fast'], 
                                indices      = layer_now['indice_fast'], 
                                scope        = layer_now['scope'])
                            if data_format == "NCHW":
                                net_fast = tf.transpose(net_fast, [0, 3, 1, 2])
                    
                    ## -- Normal --
                    if data_format == "NCHW":
                        net = tf.transpose(net, [0, 2, 3, 1])
                    net = Model.indice_unpool( 
                        net,  
                        output_shape = layer_now['output_shape'], 
                        indices      = layer_now['indice'], 
                        scope        = layer_now['scope'])
                    if data_format == "NCHW":
                        net = tf.transpose(net, [0, 3, 1, 2])
                    
                    # Diversify layer
                    if layer_now['DIVERSIFY_MODE'] == '3':
                        net = tf.cond(is_fast_mode['layer'+str(layer)], 
                                    lambda: net_fast,
                                    lambda: net)
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
                    
                    ## -- Fast --
                    with tf.variable_scope('fast'):
                        if layer_now['DIVERSIFY_MODE'] == '1':
                            net_fast = tf.layers.average_pooling2d(
                                inputs      = net_fast if bool(layer_now.get('shortcut_num')) else net, 
                                pool_size   = int(layer_now['kernel_size']), 
                                strides     = int(layer_now['stride']), 
                                padding     = 'VALID',
                                data_format = data_format_)
                        elif layer_now['DIVERSIFY_MODE'] == '2':
                            net_fast = tf.layers.average_pooling2d(
                                inputs      = net_fast, 
                                pool_size   = int(layer_now['kernel_size']), 
                                strides     = int(layer_now['stride']), 
                                padding     = 'VALID',
                                data_format = data_format_)
                        elif layer_now['DIVERSIFY_MODE'] == '3':
                            net_fast = tf.layers.average_pooling2d(
                                inputs      = net_fast, 
                                pool_size   = int(layer_now['kernel_size']), 
                                strides     = int(layer_now['stride']), 
                                padding     = 'VALID',
                                data_format = data_format_)
                    ## -- Normal --
                    net = tf.layers.average_pooling2d(
                        inputs      = net, 
                        pool_size   = int(layer_now['kernel_size']), 
                        strides     = int(layer_now['stride']), 
                        padding     = 'VALID',
                        data_format = data_format_)
                        
                    # Diversify layer
                    if layer_now['DIVERSIFY_MODE'] == '3':
                        net = tf.cond(is_fast_mode['layer'+str(layer)], 
                                    lambda: net_fast,
                                    lambda: net)
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
                        is_training    = is_training['layer'+str(layer)],
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
                        is_training    = is_training['layer'+str(layer)],
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
                    ## -- Fast --
                    trainable = False
                    if not is_testing:
                        if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                            trainable = True
                        
                    with tf.variable_scope('fast'):
                        if layer_now['DIVERSIFY_MODE'] == '1':
                            net_fast = Model.batch_norm(
                                net         = net_fast if bool(layer_now.get('shortcut_num')) else net, 
                                is_training = is_training['layer'+str(layer)], 
                                data_format = data_format,
                                trainable   = trainable)
                        elif layer_now['DIVERSIFY_MODE'] == '2':
                            net_fast = Model.batch_norm(
                                net         = net_fast, 
                                is_training = is_training['layer'+str(layer)], 
                                data_format = data_format,
                                trainable   = trainable)
                        elif layer_now['DIVERSIFY_MODE'] == '3':
                            net_fast = Model.batch_norm(
                                net         = net_fast, 
                                is_training = is_training['layer'+str(layer)], 
                                data_format = data_format,
                                trainable   = trainable)
                    # Activation
                    if layer_now['Activation'] == 'ReLU':
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.nn.relu(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                net_fast = tf.nn.relu(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                net_fast = tf.nn.relu(net_fast)
                    elif layer_now['Activation'] == 'Sigmoid':
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.nn.sigmoid(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                net_fast = tf.nn.sigmoid(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                net_fast = tf.nn.sigmoid(net_fast)
                    elif layer_now['Activation'] == 'HardTanh':
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.clip_by_value(net_fast, -1, 1)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                net_fast = tf.clip_by_value(net_fast, -1, 1)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                net_fast = tf.clip_by_value(net_fast, -1, 1)
                    elif layer_now['Activation'] == 'PReLU':
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.keras.layers.PReLU()(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                net_fast = tf.keras.layers.PReLU()(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                net_fast = tf.keras.layers.PReLU()(net_fast)
                    
                    ## -- Normal --
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Batch_Norm")
                    # trainable
                    trainable = False
                    if not is_testing:
                        if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                            if layer_now['DIVERSIFY_MODE'] == '4':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '5':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '6':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '7':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '8':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '9':
                                trainable = True
                            elif is_restoring:
                                trainable = True
                    
                    # rebuilding_now
                    rebuilding_now = False
                    if not is_testing:
                        if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                            if layer_now['DIVERSIFY_MODE'] == '7':
                                rebuilding_now = True
                            elif layer_now['DIVERSIFY_MODE'] == '8':
                                rebuilding_now = True
                            elif layer_now['DIVERSIFY_MODE'] == '9':
                                rebuilding_now = True
                    
                    net_ = Model.batch_norm(
                        net         = net, 
                        is_training = is_training['layer'+str(layer)], #tf.logical_and(is_training['layer'+str(layer)], tf.constant(trainable, tf.bool)),
                        data_format = data_format,
                        trainable   = trainable)
                    
                    cond0 = layer_now['DIVERSIFY_MODE'] == '4'
                    cond1 = layer_now['DIVERSIFY_MODE'] == '5'
                    cond2 = layer_now['DIVERSIFY_MODE'] == '6'
                    cond3 = layer_now['DIVERSIFY_MODE'] == '7'
                    cond4 = layer_now['DIVERSIFY_MODE'] == '8'
                    cond5 = layer_now['DIVERSIFY_MODE'] == '9'
                    if cond0 or cond1 or cond2 or cond3 or cond4 or cond5:
                        with tf.variable_scope('complexity_1'):
                            # is_training
                            if cond0 or cond1 or cond2:
                                is_training_ = is_training['layer'+str(layer)] #tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(0, tf.float32)))
                                trainable_ = trainable
                            elif (cond3 or cond4 or cond5) and rebuilding_now:
                                is_training_ = tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(2, tf.float32)))
                                trainable_ = trainable and complexity_mode_now == 1
                            else:
                                is_training_ = is_training['layer'+str(layer)]
                                trainable_ = trainable
                            net_1 = Model.batch_norm(net, is_training_, data_format, trainable_)
                        with tf.variable_scope('complexity_2'):
                            # is_training
                            if cond0 or cond1 or cond2:
                                is_training_ = is_training['layer'+str(layer)] #tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(1, tf.float32)))
                                trainable_ = trainable
                            elif (cond3 or cond4 or cond5) and rebuilding_now:
                                is_training_ = tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(3, tf.float32)))
                                trainable_ = trainable and complexity_mode_now == 2
                            else:
                                is_training_ = is_training['layer'+str(layer)]
                                trainable_ = trainable
                            net_2 = Model.batch_norm(net, is_training_, data_format, trainable_)
                        with tf.variable_scope('complexity_3'):
                            # is_training
                            if cond0 or cond1 or cond2:
                                is_training_ = is_training['layer'+str(layer)] #tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(2, tf.float32)))
                                trainable_ = trainable
                            elif (cond3 or cond4 or cond5) and rebuilding_now:
                                is_training_ = tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(4, tf.float32)))
                                trainable_ = trainable and complexity_mode_now == 3
                            else:
                                is_training_ = is_training['layer'+str(layer)]
                                trainable_ = trainable
                            net_3 = Model.batch_norm(net, is_training_, data_format, trainable_)
                        with tf.variable_scope('complexity_4'):
                            # is_training
                            if cond0 or cond1 or cond2:
                                is_training_ = is_training['layer'+str(layer)] #tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(3, tf.float32)))
                                trainable_ = trainable
                            elif (cond3 or cond4 or cond5) and rebuilding_now:
                                is_training_ = tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(5, tf.float32)))
                                trainable_ = trainable and complexity_mode_now == 4
                            else:
                                is_training_ = is_training['layer'+str(layer)]
                                trainable_ = trainable
                            net_4 = Model.batch_norm(net, is_training_, data_format, trainable_)
                        with tf.variable_scope('complexity_5'):
                            # is_training
                            if cond0 or cond1 or cond2:
                                is_training_ = is_training['layer'+str(layer)] #tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(4, tf.float32)))
                                trainable_ = trainable
                            elif (cond3 or cond4 or cond5) and rebuilding_now:
                                is_training_ = tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(6, tf.float32)))
                                trainable_ = trainable and complexity_mode_now == 5
                            else:
                                is_training_ = is_training['layer'+str(layer)]
                                trainable_ = trainable
                            net_5 = Model.batch_norm(net, is_training_, data_format, trainable_)
                        with tf.variable_scope('complexity_6'):
                            # is_training
                            if cond0 or cond1 or cond2:
                                is_training_ = is_training['layer'+str(layer)] #tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(5, tf.float32)))
                                trainable_ = trainable
                            elif (cond3 or cond4 or cond5) and rebuilding_now:
                                is_training_ = tf.logical_and(is_training['layer'+str(layer)], tf.equal(complexity_mode['layer'+str(layer)], tf.constant(7, tf.float32)))
                                trainable_ = trainable and complexity_mode_now == 6
                            else:
                                is_training_ = is_training['layer'+str(layer)]
                                trainable_ = trainable
                            net_6 = Model.batch_norm(net, is_training_, data_format, trainable_)
                        
                        if cond0 or cond1 or cond2:
                            net = tf.case(
                                pred_fn_pairs = {
                                    tf.equal(complexity_mode['layer'+str(layer)], tf.constant(0, tf.float32)): lambda: net_1,
                                    tf.equal(complexity_mode['layer'+str(layer)], tf.constant(1, tf.float32)): lambda: net_2,
                                    tf.equal(complexity_mode['layer'+str(layer)], tf.constant(2, tf.float32)): lambda: net_3,
                                    tf.equal(complexity_mode['layer'+str(layer)], tf.constant(3, tf.float32)): lambda: net_4,
                                    tf.equal(complexity_mode['layer'+str(layer)], tf.constant(4, tf.float32)): lambda: net_5,
                                    tf.equal(complexity_mode['layer'+str(layer)], tf.constant(5, tf.float32)): lambda: net_6},
                                default = None, exclusive = True)
                        
                        if cond3 or cond4 or cond5:
                            if rebuilding_now:
                                net = tf.case(
                                    pred_fn_pairs = {
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(1, tf.float32)): lambda: net_,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(2, tf.float32)): lambda: net_1,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(3, tf.float32)): lambda: net_2,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(4, tf.float32)): lambda: net_3,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(5, tf.float32)): lambda: net_4,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(6, tf.float32)): lambda: net_5},
                                    default = None, exclusive = True)
                            else:
                                net = tf.case(
                                    pred_fn_pairs = {
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(0, tf.float32)): lambda: net_,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(1, tf.float32)): lambda: net_1,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(2, tf.float32)): lambda: net_2,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(3, tf.float32)): lambda: net_3,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(4, tf.float32)): lambda: net_4,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(5, tf.float32)): lambda: net_5,
                                        tf.equal(complexity_mode['layer'+str(layer)], tf.constant(6, tf.float32)): lambda: net_6},
                                    default = None, exclusive = True)
                    else:
                        net = net_
                    
                    # Activation
                    if layer_now['Activation'] == 'ReLU':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> ReLU")
                        net = tf.nn.relu(net)
                    elif layer_now['Activation'] == 'Sigmoid':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Sigmoid")
                        net = tf.nn.sigmoid(net)
                    elif layer_now['Activation'] == 'HardTanh':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> HardTanh")
                        net = tf.clip_by_value(net, -1, 1)
                    elif layer_now['Activation'] == 'PReLU':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> PReLU")
                        net = tf.keras.layers.PReLU()(net)
                    
                    # Diversify layer
                    if layer_now['DIVERSIFY_MODE'] == '3':
                        net = tf.cond(is_fast_mode['layer'+str(layer)], 
                                    lambda: net_fast,
                                    lambda: net)
                                    
                    # For Pruning
                    cond_1 = layer_now['DIVERSIFY_MODE'] == '2'
                    cond_2 = layer_now['DIVERSIFY_MODE'] == '3'
                    cond_3 = layer_now['DIVERSIFY_MODE'] == '4'
                    cond_4 = layer_now['DIVERSIFY_MODE'] == '5'
                    cond_5 = layer_now['DIVERSIFY_MODE'] == '6'
                    if not is_testing:
                        cond_6 = 'layer'+str(layer) in diversify_layer['layer']
                    else:
                        cond_6 = False
                        
                    if (cond_1 or cond_2 or cond_3 or cond_4 or cond_5) and cond_6:
                        prune_info_dict['layer%d'%(layer-1)].update({
                            'beta': [beta for beta in tf.trainable_variables(scope="student/Model/layer%d/"%layer) 
                                    if 'beta' in beta.name][0]})
                    
                    # Activation Quantization
                    if layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Quantized Activation ")
                        net = Model.quantize_Module(net, is_quantized_activation['layer'+str(layer)], data_format)
                #------------------#
                #    Activation    #
                #------------------#
                if layer_now['type'] == 'ACT':
                    ## -- Fast --
                    if layer_now['Activation'] == 'ReLU':
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.nn.relu(net_fast if bool(layer_now.get('shortcut_num')) else net)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                net_fast = tf.nn.relu(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                net_fast = tf.nn.relu(net_fast)
                    elif layer_now['Activation'] == 'Sigmoid':
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.nn.sigmoid(net_fast if bool(layer_now.get('shortcut_num')) else net)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                net_fast = tf.nn.sigmoid(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                net_fast = tf.nn.sigmoid(net_fast)
                    elif layer_now['Activation'] == 'HardTanh':
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.clip_by_value(net_fast if bool(layer_now.get('shortcut_num')) else net, -1, 1)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                net_fast = tf.clip_by_value(net_fast, -1, 1)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                net_fast = tf.clip_by_value(net_fast, -1, 1)
                    elif layer_now['Activation'] == 'PReLU':
                        with tf.variable_scope('fast'):
                            if layer_now['DIVERSIFY_MODE'] == '1':
                                net_fast = tf.keras.layers.PReLU()(net_fast if bool(layer_now.get('shortcut_num')) else net)
                            elif layer_now['DIVERSIFY_MODE'] == '2':
                                net_fast = tf.keras.layers.PReLU()(net_fast)
                            elif layer_now['DIVERSIFY_MODE'] == '3':
                                net_fast = tf.keras.layers.PReLU()(net_fast)
                    
                    ## -- Normal --
                    if layer_now['Activation'] == 'ReLU':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> ReLU")
                        net = tf.nn.relu(net)
                    elif layer_now['Activation'] == 'Sigmoid':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Sigmoid")
                        net = tf.nn.sigmoid(net)
                    elif layer_now['Activation'] == 'HardTanh':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> HardTanh")
                        net = tf.clip_by_value(net, -1, 1)
                    elif layer_now['Activation'] == 'PReLU':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> PReLU")
                        net = tf.keras.layers.PReLU()(net)
                    
                    # Diversify layer
                    if layer_now['DIVERSIFY_MODE'] == '3':
                        net = tf.cond(is_fast_mode['layer'+str(layer)], 
                                    lambda: net_fast,
                                    lambda: net)
                    
                    # Activation Quantization
                    if layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Quantized Activation ")
                        net = Model.quantize_Module(net, is_quantized_activation['layer'+str(layer)], data_format)
                #-----------------------#
                #    Fully Connected    #
                #-----------------------#
                if layer_now['type'] == 'FC':
                    
                    # Dropout
                    if layer==(len(Model_dict)-1):
                        net = tf.cond(is_training['layer'+str(layer)], lambda: tf.layers.dropout(net, DROPOUT_RATE), lambda: net)
                    
                    ## -- Fast --
                    trainable = False
                    if not is_testing:
                        if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                            trainable = True
                    
                    with tf.variable_scope('fast'):
                        if layer_now['DIVERSIFY_MODE'] == '1':
                            net_fast = Model.FC(
                                net                     = net_fast if bool(layer_now.get('shortcut_num')) else net, 
                                output_channel          = int(layer_now['output_channel']),
                                initializer             = tf.variance_scaling_initializer(),
                                trainable               = trainable,
                                is_training             = is_training['layer'+str(layer)],
                                is_add_biases           = layer_now['is_add_biases']           == 'TRUE',
                                is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',
                                is_dilated              = layer_now['is_dilated']              == 'TRUE',
                                is_ternary              = is_ternary['layer'+str(layer)]                ,
                                is_binary               = is_binary['layer'+str(layer)]                 ,
                                is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                                IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                IS_DIVERSITY            = False,
                                is_fast_mode            = is_fast_mode['layer'+str(layer)],
                                fast_mode               = layer_now['DIVERSIFY_MODE'] if layer_now['DIVERSIFY_MODE'] != None and layer_now['DIVERSIFY_MODE'] != 'None' else None,
                                Activation              = layer_now['Activation'],
                                data_format             = data_format,
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                        elif layer_now['DIVERSIFY_MODE'] == '2':
                            net_fast = Model.FC(
                                net                     = net_fast, 
                                output_channel          = int(layer_now['output_channel']),
                                initializer             = tf.variance_scaling_initializer(),
                                trainable               = trainable,
                                is_training             = is_training['layer'+str(layer)],
                                is_add_biases           = layer_now['is_add_biases']           == 'TRUE',
                                is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',
                                is_dilated              = layer_now['is_dilated']              == 'TRUE',
                                is_ternary              = is_ternary['layer'+str(layer)]                ,
                                is_binary               = is_binary['layer'+str(layer)]                 ,
                                is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                                IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                IS_DIVERSITY            = False,
                                is_fast_mode            = is_fast_mode['layer'+str(layer)],
                                fast_mode               = layer_now['DIVERSIFY_MODE'] if layer_now['DIVERSIFY_MODE'] != None and layer_now['DIVERSIFY_MODE'] != 'None' else None,
                                Activation              = layer_now['Activation'],
                                data_format             = data_format,
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                        elif layer_now['DIVERSIFY_MODE'] == '3':
                            net_fast = Model.FC(
                                net                     = net_fast, 
                                output_channel          = int(layer_now['output_channel']),
                                initializer             = tf.variance_scaling_initializer(),
                                trainable               = trainable,
                                is_training             = is_training['layer'+str(layer)],
                                is_add_biases           = layer_now['is_add_biases']           == 'TRUE',
                                is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',
                                is_dilated              = layer_now['is_dilated']              == 'TRUE',
                                is_ternary              = is_ternary['layer'+str(layer)]                ,
                                is_binary               = is_binary['layer'+str(layer)]                 ,
                                is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                                IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                IS_DIVERSITY            = False,
                                is_fast_mode            = is_fast_mode['layer'+str(layer)],
                                fast_mode               = layer_now['DIVERSIFY_MODE'] if layer_now['DIVERSIFY_MODE'] != None and layer_now['DIVERSIFY_MODE'] != 'None' else None,
                                Activation              = layer_now['Activation'],
                                data_format             = data_format,
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                    
                    ## -- Normal --
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Fully-Connected")
                    
                    trainable = False
                    if not is_testing:
                        if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                            if layer_now['DIVERSIFY_MODE'] == '0':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '4':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '5':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '6':
                                trainable = True
                            elif is_restoring:
                                trainable = True
                    
                    net = Model.FC(
                        net, 
                        output_channel          = int(layer_now['output_channel']),
                        initializer             = tf.variance_scaling_initializer(),
                        trainable               = trainable,
                        is_training             = is_training['layer'+str(layer)],
                        is_add_biases           = layer_now['is_add_biases']           == 'TRUE',
                        is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',
                        is_dilated              = layer_now['is_dilated']              == 'TRUE',
                        is_ternary              = is_ternary['layer'+str(layer)]                ,
                        is_binary               = is_binary['layer'+str(layer)]                 ,
                        is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                        IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                        IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
                        IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                        IS_DIVERSITY            = layer_now['IS_DIVERSITY'] == 'TRUE' and layer_now['DIVERSIFY_MODE'] == '0',
                        is_fast_mode            = is_fast_mode['layer'+str(layer)],
                        fast_mode               = layer_now['DIVERSIFY_MODE'] if layer_now['DIVERSIFY_MODE'] != None and layer_now['DIVERSIFY_MODE'] != 'None' else None,
                        Activation              = layer_now['Activation'],
                        data_format             = data_format,
                        Analysis                = Analysis,
                        scope                   = layer_now['scope'])
                    
                    # Diversify layer
                    if layer_now['DIVERSIFY_MODE'] == '3':
                        net = tf.cond(is_fast_mode['layer'+str(layer)], 
                                    lambda: net_fast,
                                    lambda: net)
                    
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
                        net = tf.cond(is_training['layer'+str(layer)], lambda: tf.layers.dropout(net, DROPOUT_RATE), lambda: net)

                    # For finding the similar weights
                    inputs_and_kernels.update({'layer%d' %(layer): {'inputs' : net}})
                    
                    ## -- Fast --
                    trainable = False
                    if not is_testing:
                        if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                            trainable = True
                        
                    # Fast Mode Conv (Pruned Conv)
                    with tf.variable_scope('fast'):
                        if layer_now['DIVERSIFY_MODE'] == '1':
                            net_fast = Model.conv2D( 
                                net                     = net_fast if bool(layer_now.get('shortcut_num')) else net, 
                                kernel_size             = int(layer_now['kernel_size']), 
                                stride                  = int(layer_now['stride']          ),
                                internal_channel        = int(layer_now['internal_channel']),
                                output_channel          = int(layer_now['output_channel']  ),
                                rate                    = int(layer_now['rate']            ),
                                group                   = int(layer_now['group']           ),
                                initializer             = tf.variance_scaling_initializer(),
                                trainable               = trainable,
                                is_training             = is_training['layer'+str(layer)],
                                is_add_biases           = layer_now['is_add_biases']           == 'TRUE', 
                                is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',      
                                is_dilated              = layer_now['is_dilated']              == 'TRUE',      
                                is_depthwise            = layer_now['is_depthwise']            == 'TRUE',  
                                is_ternary              = is_ternary['layer'+str(layer)]                ,
                                is_binary               = is_binary['layer'+str(layer)]                 ,
                                is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                                IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                IS_DIVERSITY            = False,
                                is_fast_mode            = is_fast_mode['layer'+str(layer)],
                                fast_mode               = layer_now['DIVERSIFY_MODE'] if layer_now['DIVERSIFY_MODE'] != None and layer_now['DIVERSIFY_MODE'] != 'None' else None,
                                Activation              = layer_now['Activation'],
                                padding                 = "SAME",
                                data_format             = data_format,
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                        elif layer_now['DIVERSIFY_MODE'] == '2':
                            net_fast = Model.conv2D( 
                                net_fast, 
                                kernel_size             = int(layer_now['kernel_size']), 
                                stride                  = int(layer_now['stride']          ),
                                internal_channel        = int(layer_now['internal_channel']),
                                output_channel          = int(layer_now['output_channel']  ),
                                rate                    = int(layer_now['rate']            ),
                                group                   = int(layer_now['group']           ),
                                initializer             = tf.variance_scaling_initializer(),
                                trainable               = trainable,
                                is_training             = is_training['layer'+str(layer)],
                                is_add_biases           = layer_now['is_add_biases']           == 'TRUE', 
                                is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',      
                                is_dilated              = layer_now['is_dilated']              == 'TRUE',      
                                is_depthwise            = layer_now['is_depthwise']            == 'TRUE',  
                                is_ternary              = is_ternary['layer'+str(layer)]                ,
                                is_binary               = is_binary['layer'+str(layer)]                 ,
                                is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                                IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                IS_DIVERSITY            = False,
                                is_fast_mode            = is_fast_mode['layer'+str(layer)],
                                fast_mode               = layer_now['DIVERSIFY_MODE'] if layer_now['DIVERSIFY_MODE'] != None and layer_now['DIVERSIFY_MODE'] != 'None' else None,
                                Activation              = layer_now['Activation'],
                                padding                 = "SAME",
                                data_format             = data_format,
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                        elif layer_now['DIVERSIFY_MODE'] == '3':
                            net_fast = Model.conv2D(
                                net_fast, 
                                kernel_size             = int(layer_now['kernel_size']), 
                                stride                  = int(layer_now['stride']          ),
                                internal_channel        = int(layer_now['internal_channel']),
                                output_channel          = int(layer_now['output_channel']  ),
                                rate                    = int(layer_now['rate']            ),
                                group                   = int(layer_now['group']           ),
                                initializer             = tf.variance_scaling_initializer(),
                                trainable               = trainable,
                                is_training             = is_training['layer'+str(layer)],
                                is_add_biases           = layer_now['is_add_biases']           == 'TRUE', 
                                is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',      
                                is_dilated              = layer_now['is_dilated']              == 'TRUE',      
                                is_depthwise            = layer_now['is_depthwise']            == 'TRUE',  
                                is_ternary              = is_ternary['layer'+str(layer)]                ,
                                is_binary               = is_binary['layer'+str(layer)]                 ,
                                is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                                IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                                IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
                                IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                                IS_DIVERSITY            = False,
                                is_fast_mode            = is_fast_mode['layer'+str(layer)],
                                fast_mode               = layer_now['DIVERSIFY_MODE'] if layer_now['DIVERSIFY_MODE'] != None and layer_now['DIVERSIFY_MODE'] != 'None' else None,
                                Activation              = layer_now['Activation'],
                                padding                 = "SAME",
                                data_format             = data_format,
                                Analysis                = Analysis,
                                scope                   = layer_now['scope'])
                    
                    ## -- Normal --
                    # trainable
                    trainable = False
                    if not is_testing:
                        if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                            if layer_now['DIVERSIFY_MODE'] == '0':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '4':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '5':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '6':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '7':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '8':
                                trainable = True
                            elif layer_now['DIVERSIFY_MODE'] == '9':
                                trainable = True
                            elif is_restoring:
                                trainable = True
                    
                    # rebuilding_now
                    rebuilding_now = False
                    if not is_testing:
                        if 'layer'+str(layer) in diversify_layer['layer'] and is_student:
                            if layer_now['DIVERSIFY_MODE'] == '7':
                                rebuilding_now = True
                            elif layer_now['DIVERSIFY_MODE'] == '8':
                                rebuilding_now = True
                            elif layer_now['DIVERSIFY_MODE'] == '9':
                                rebuilding_now = True
                                
                    net = Model.conv2D( 
                        net                     = net, 
                        kernel_size             = int(layer_now['kernel_size']), 
                        stride                  = int(layer_now['stride']          ),
                        internal_channel        = int(layer_now['internal_channel']),
                        output_channel          = int(layer_now['output_channel']  ),
                        rate                    = int(layer_now['rate']            ),
                        group                   = int(layer_now['group']           ),
                        initializer             = tf.variance_scaling_initializer(),
                        trainable               = trainable,
                        is_training             = is_training['layer'+str(layer)], #tf.logical_and(is_training['layer'+str(layer)], tf.constant(trainable, tf.bool)),
                        is_add_biases           = layer_now['is_add_biases']           == 'TRUE', 
                        is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',      
                        is_dilated              = layer_now['is_dilated']              == 'TRUE',      
                        is_depthwise            = layer_now['is_depthwise']            == 'TRUE',  
                        is_ternary              = is_ternary['layer'+str(layer)]                ,
                        is_binary               = is_binary['layer'+str(layer)]                 ,
                        is_quantized_activation = is_quantized_activation['layer'+str(layer)]   ,
                        IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',
                        IS_BINARY               = layer_now['IS_BINARY']               == 'TRUE',
                        IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
                        IS_DIVERSITY            = layer_now['IS_DIVERSITY'] == 'TRUE' and layer_now['DIVERSIFY_MODE'] == '0',
                        is_fast_mode            = is_fast_mode['layer'+str(layer)],
                        fast_mode               = layer_now['DIVERSIFY_MODE'] if layer_now['DIVERSIFY_MODE'] != None and layer_now['DIVERSIFY_MODE'] != 'None' else None,
                        complexity_mode         = complexity_mode['layer'+str(layer)],
                        rebuilding_now          = rebuilding_now,
                        complexity_mode_now     = complexity_mode_now,
                        Activation              = layer_now['Activation'],
                        padding                 = "SAME",
                        data_format             = data_format,
                        Analysis                = Analysis,
                        scope                   = layer_now['scope'])
                    
                    # Diversify layer
                    if layer_now['DIVERSIFY_MODE'] == '3':
                        net = tf.cond(is_fast_mode['layer'+str(layer)], 
                                    lambda: net_fast,
                                    lambda: net)
                    
                    if layer==(len(Model_dict)-1) and data_format == "NCHW":
                        net = tf.transpose(net, [0, 2, 3, 1])

                    # For Pruning
                    cond_0 = layer_now['DIVERSIFY_MODE'] == '1'
                    cond_1 = layer_now['DIVERSIFY_MODE'] == '2'
                    cond_2 = layer_now['DIVERSIFY_MODE'] == '3'
                    cond_3 = layer_now['DIVERSIFY_MODE'] == '4'
                    cond_4 = layer_now['DIVERSIFY_MODE'] == '5'
                    cond_5 = layer_now['DIVERSIFY_MODE'] == '6'
                    if not is_testing:
                        cond_6 = 'layer'+str(layer) in diversify_layer['layer']
                    else:
                        cond_6 = False
                        
                    if (cond_0 or cond_1 or cond_2 or cond_3 or cond_4 or cond_5) and cond_6:
                        # Scope
                        if cond_3 or cond_4 or cond_5:
                            scope = "student/Model/layer%d/"%layer
                        elif cond_0 or cond_1 or cond_2:
                            scope = "student/Model/layer%d/"%layer+'fast/'
                        
                        # prune_info_dict
                        if not bool(prune_info_dict.get('layer%d'%layer)):
                            prune_info_dict.update({
                                'layer%d'%layer: {'weights': tf.get_collection("weights", scope=scope)[0]}})
                        else:
                            prune_info_dict['layer%d'%layer].update({
                                'weights': tf.get_collection("weights", scope=scope)[0]})
                            
                        prune_info_dict['layer%d'%layer].update({
                            'mask'        : tf.get_collection("float32_weights_mask", scope=scope)[0],
                            'outputs'     : tf.get_collection("conv_outputs"  , scope=scope)[0],
                            'stride'      : int(layer_now['stride']),
                            'is_shortcut' : is_shortcut_past_layer,
                            'is_depthwise': layer_now['is_depthwise'] == 'TRUE'})
                        
                        try:
                            if layer_now['is_batch_norm'] == 'TRUE':
                                prune_info_dict['layer%d'%layer].update({
                                    'beta': [beta for beta in tf.trainable_variables(scope="student/Model/layer%d/"%layer) 
                                            if 'beta' in beta.name][0]})
                        except:
                            None
                        
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
                    
                # Diversify Trained Layer output
                if is_testing:
                    train_net = None
                elif 'layer'+str(layer) == diversify_layer['layer'][-1]:
                    train_net = net    
                    
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
            if is_testing or 'layer'+str(layer) in diversify_layer:
                break
        
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
            for v in tf.trainable_variables(): 
                print(v)
            exit()
    return net, train_net, Analysis, max_parameter, inputs_and_kernels, prune_info_dict

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

def read_csv_file( # Recall how to read csv file. Not for using.
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
    IS_BINARY               = False,
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
        elif IS_BINARY:
            weight_bits = 1
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
        
        """
        print("(%2d x %2d x %2d x %2d) x %2d x %2d = %2d" 
            %(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3], H, W, computation))
        """
        
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
    # Weight
    for i in range(len(weights_collection)):
        w = np.absolute(sess.run(weights_collection[i]))
        w_bd = 0.7 * np.mean(w)
        #w_bd = np.percentile(w, weights_bd_ratio) 
        if i==0:
            weights_bd = np.array([[-w_bd, w_bd]])
        else:
            weights_bd = np.concatenate([weights_bd, np.array([[-w_bd, w_bd]])], axis=0)    
    # Bias
    for i in range(len(biases_collection)):
        b = np.absolute(sess.run(biases_collection [i]))
        b_bd = 0.7 * np.mean(b)
        #b_bd = np.percentile(b, biases_bd_ratio )
        if i==0:
            biases_bd  = np.array([[-b_bd, b_bd]])
        else:
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
    # Weight
    for i in range(len(ternary_weights_bd_collection)):
        sess.run(ternary_weights_bd_collection[i].assign(ternary_weights_bd[i]))
    # Bias
    for i in range(len(ternary_biases_bd_collection)):
        sess.run(ternary_biases_bd_collection[i].assign(ternary_biases_bd [i]))
                
def binarize_bd(
    weights_collection,
    biases_collection,
    weights_bd_ratio,   # percentile. Ex 50=50%
    biases_bd_ratio,    # percentile. Ex 50=50%
    sess
    ):
    # Weight
    for i in range(len(weights_collection)):
        w = np.absolute(sess.run(weights_collection[i]))
        w_bd = np.mean(w)
        #w_bd = np.percentile(w, weights_bd_ratio) 
        if i==0:
            weights_bd = np.array([[w_bd]])
        else:
            weights_bd = np.concatenate([weights_bd, np.array([[w_bd]])], axis=0)    
    # Bias
    for i in range(len(biases_collection)):
        b = np.absolute(sess.run(biases_collection [i]))
        b_bd = np.mean(b)
        #b_bd = np.percentile(b, biases_bd_ratio )
        if i==0:
            biases_bd  = np.array([[b_bd]])
        else:
            biases_bd  = np.concatenate([biases_bd , np.array([[b_bd]])], axis=0)
            
    weights_table = [-1, 1]
    biases_table  = [-1, 1]
    
    return weights_bd, biases_bd, weights_table, biases_table

def assign_binary_boundary(
    binary_weights_bd_collection, 
    binary_biases_bd_collection, 
    binary_weights_bd,
    binary_biases_bd,
    sess
    ):
    # Weight
    for i in range(len(binary_weights_bd_collection)):
        sess.run(binary_weights_bd_collection[i].assign(binary_weights_bd[i]))
    # Bias
    for i in range(len(binary_biases_bd_collection)):
        sess.run(binary_biases_bd_collection[i].assign(binary_biases_bd [i]))
 
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

def average_gradients(
    tower_grads
    ):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
        
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
    
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
    
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

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

def filter_prune_by_similarity( # Integration Version of All the Following Filter Pruning Methods
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    sess,
    is_connection = False,
    is_beta = False,
    beta_threshold = 0.001,
    skip_first_layer = False,
    skip_last_layer = False,
    discard_feature = False,
    all_be_pruned = False,
    cut_connection_less_to_more = False
    ):
    #================#
    #  Sorted Layer  #
    #================#
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]
    print(sorted_layer)
    #===================#
    #  Load Model Info  #
    #===================#
    skip_layers = sorted_layer[0]
    if skip_first_layer:
        skip_layers.append(sorted_layer[1])
    if skip_last_layer:
        skip_layers.append(sorted_layer[-1])
    all_weight = {}
    all_mask = {}
    all_output_shape = {} # For Computing Computation
    all_stride = {} # For Computing Computation
    all_beta = {}
    for layer_iter, layer in enumerate(sorted_layer):
        all_weight.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_output_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})
        all_beta.update({layer: sess.run(prune_info_dict[layer]['beta'])})
    
    #=======================#
    #  Original Model Info  #
    #=======================#
    # Original Model Size
    original_size = 0
    for layer_iter, layer in enumerate(sorted_layer):
        original_size += reduce(lambda x, y: x*y, np.shape(all_mask[layer]))
    
    # before_prune_weights_size
    before_prune_weights_size = 0
    for layer_iter, layer in enumerate(sorted_layer):
        before_prune_weights_size += np.sum(all_mask[layer])
       
    #=====================#
    #  Build Pruned_dict  #
    #=====================#
    pruned_dict = {}
    for layer_iter, layer in enumerate(sorted_layer):
        mask = all_mask[layer]
        channel_num = np.shape(mask)[3]
        tmp_dict = {}
        for i in range(channel_num):
            tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        pruned_dict.update({layer: tmp_dict})
    
    #=========================#
    #  Build Similarity_dict  #
    #=========================#
    print('Calculating Cosine Similarities ...')
    similarity_dict = {}
    iter = 0
    for layer_iter, layer in enumerate(sorted_layer):
        # Skip Specific Layers
        if layer in skip_layers:
            continue
        else:
            print(layer, end=" ")
            tStart = time.time()
        # Current Layer
        weight = all_weight[layer]
        mask = all_mask[layer]
        beta = all_beta[layer]
        # past layer
        past_layer = sorted_layer[layer_iter-1]
        # Calculate Cosine Similarities
        depth = np.shape(weight)[2]
        depth_i_iter = 0
        for depth_i in range(depth):
            depth_j_iter = 0
            for depth_j in range(depth_i+1, depth):
                # If the depth_i or depth_i has been pruned, we don't calculate its angle to others
                # By doing so, we will not prune the kernel which has been pruned before.
                is_depth_i_pruned = np.sum(mask[:, :, depth_i, :]) == 0
                is_depth_j_pruned = np.sum(mask[:, :, depth_j, :]) == 0
                if not is_depth_i_pruned and not is_depth_j_pruned:
                    # Vector
                    x = weight[:, :, depth_i, :] * mask[:, :, depth_i, :]
                    y = weight[:, :, depth_j, :] * mask[:, :, depth_j, :]
                    x = np.reshape(x, [np.size(x)])
                    y = np.reshape(y, [np.size(y)])
                    
                    # Cosine Similarity
                    if np.sum(np.square(x))==0:
                        similarity = 0.0
                        print('\033[0;33mWarning\033[0m: Origin Point in {}, {}' .format(layer, 'depth'+str(i)))
                    elif np.sum(np.square(y))==0:
                        similarity = 0.0
                        print('\033[0;33mWarning\033[0m: Origin Point in {}, {}' .format(layer, 'depth'+str(i)))
                    else:
                        similarity = np.abs(np.sum(x*y)/(np.sqrt(np.sum(np.square(x)))*np.sqrt(np.sum(np.square(y)))))
                    
                    assert np.abs(similarity) <= 1.0, '|Similarity| <= 1.0'
                    
                    # (Option) Beta
                    if is_beta and (np.abs(beta[depth_i]) < beta_threshold or np.abs(beta[depth_j]) < beta_threshold):
                        similarity = similarity * 1.5
                    
                    # Fixed Point similarity (Something Wrong in Dictionary without This Step!)
                    similarity = int(similarity * 1e8)

                    # similarity_dict
                    if bool(similarity_dict.get(str(similarity))):
                        number = len(similarity_dict[str(similarity)])
                        similarity_dict[str(similarity)].update({
                            str(number):{
                                'past_layer': past_layer, 
                                'layer': layer, 
                                'depth_i': depth_i, 
                                'depth_j': depth_j, 
                                'pruned': False}})
                    else:
                        similarity_dict.update({str(similarity): {
                            '0': {
                                'past_layer': past_layer, 
                                'layer': layer, 
                                'depth_i': depth_i, 
                                'depth_j': depth_j, 
                                'pruned': False}}})
                                
                    # Total Similarities
                    if depth_j_iter == 0:
                        depth_i_similarities = np.array([similarity])
                    else:
                        depth_i_similarities = np.concatenate([depth_i_similarities, np.array([similarity])])
                    depth_j_iter = depth_j_iter + 1
            
            if depth_j_iter != 0:
                if depth_i_iter == 0:
                    layer_similarities = depth_i_similarities
                else:
                    layer_similarities = np.concatenate([layer_similarities, depth_i_similarities])
                depth_i_iter = depth_i_iter + 1
            
        if depth_i_iter != 0:
            if iter == 0:
                total_similarities = layer_similarities
            else:
                total_similarities = np.concatenate([total_similarities, layer_similarities])
            iter = iter + 1
        print(len(layer_similarities), end=" ")
        tEnd = time.time()
        print("(cost %f seconds)" %(tEnd - tStart))
    
    #=========================#
    #  Sort All Similarities  #
    #=========================#
    sorted_similarities = np.sort(total_similarities)[::-1]
    
    #========================================#
    #  Build Connection Tree & Similar Depth #
    #========================================#
    print('Building Connection Tree ...')
    similarity_dict_ = copy.deepcopy(similarity_dict)
    connection = {}
    similar_depth = {}
    pruned_size = 0
    pruned_similarity_num = 0
    for similarity_iter, similarity in enumerate(sorted_similarities):
        pruned_similarity_num = pruned_similarity_num + 1
        for _, index in enumerate(similarity_dict_[str(similarity)].keys()):
            if similarity_dict_[str(similarity)][index]['pruned']:
                continue
            similarity_dict_[str(similarity)][index].update({'pruned': True})
            # past layer
            past_layer = similarity_dict_[str(similarity)][index]['past_layer']
            past_mask = all_mask[past_layer]
            # current layer
            layer = similarity_dict_[str(similarity)][index]['layer']
            mask = all_mask[layer]
            depth_i = similarity_dict_[str(similarity)][index]['depth_i']
            depth_j = similarity_dict_[str(similarity)][index]['depth_j']
            
            if all_be_pruned:
                pruned_size = pruned_size + reduce(lambda x, y: x*y, np.shape(past_mask[:, :, :, depth_i]))
                pruned_size = pruned_size + reduce(lambda x, y: x*y, np.shape(mask[:, :, depth_i, :]))
                pruned_size = pruned_size + reduce(lambda x, y: x*y, np.shape(past_mask[:, :, :, depth_j]))
                pruned_size = pruned_size + reduce(lambda x, y: x*y, np.shape(mask[:, :, depth_j, :]))
            else:
                # Connection & Similar Depth
                if not bool(connection.get(layer)):
                    is_depth_i_in_connection = False
                    is_depth_j_in_connection = False
                    connection.update({layer: {str(depth_i): 1, str(depth_j): 1}})
                    similar_depth.update({layer: {str(depth_i): [str(depth_j)], str(depth_j): [str(depth_i)]}})
                else:
                    is_depth_i_in_connection = bool(connection[layer].get(str(depth_i)))
                    is_depth_j_in_connection = bool(connection[layer].get(str(depth_j)))
                    
                    # Connection
                    if not is_depth_i_in_connection:
                        connection[layer].update({str(depth_i): 1})
                    else:
                        connection[layer].update({str(depth_i): connection[layer][str(depth_i)]+1})
                    if not is_depth_j_in_connection:
                        connection[layer].update({str(depth_j): 1})
                    else:
                        connection[layer].update({str(depth_j): connection[layer][str(depth_j)]+1})
                    
                    # Similar Depth
                    if not is_depth_i_in_connection and not is_depth_j_in_connection:
                        similar_depth[layer].update({str(depth_i): [str(depth_j)], str(depth_j): [str(depth_i)]})
                    elif not is_depth_i_in_connection and is_depth_j_in_connection:
                        similar_depth[layer].update({str(depth_i): [str(depth_j)]})
                        similar_depth[layer][str(depth_j)].append(str(depth_i))
                    elif is_depth_i_in_connection and not is_depth_j_in_connection:
                        similar_depth[layer][str(depth_i)].append(str(depth_j))
                        similar_depth[layer].update({str(depth_j): [str(depth_i)]})
                    else:
                        similar_depth[layer][str(depth_i)].append(str(depth_j))
                        similar_depth[layer][str(depth_j)].append(str(depth_i))
                        
                # Prune Size
                if not is_depth_i_in_connection or not is_depth_j_in_connection:
                    pruned_size = pruned_size + reduce(lambda x, y: x*y, np.shape(past_mask[:, :, :, depth_i]))
                    pruned_size = pruned_size + reduce(lambda x, y: x*y, np.shape(mask[:, :, depth_i, :]))
        
        # More Precise Way to get pruned_size
        if pruned_size >= original_size * pruning_propotion * 0.8:
            # Build the connection_num directory
            if cut_connection_less_to_more:
                similarity_dict_ = copy.deepcopy(similarity_dict)
                pruned_dict_ = copy.deepcopy(pruned_dict)
                connection_similarity_dict = {}
                for pruned_similarity_iter in range(pruned_similarity_num):
                    similarity = sorted_similarities[pruned_similarity_iter]
                    for _, index in enumerate(similarity_dict_[str(similarity)].keys()):
                        if similarity_dict_[str(similarity)][index]['pruned']:
                            continue
                        similarity_dict_[str(similarity)][index].update({'pruned': True})
                        # current layer
                        layer = similarity_dict_[str(similarity)][index]['layer']
                        # depth
                        depth_i = similarity_dict_[str(similarity)][index]['depth_i']
                        depth_j = similarity_dict_[str(similarity)][index]['depth_j']
                        connection_i = connection[layer][str(depth_i)]
                        connection_j = connection[layer][str(depth_j)]
                        max_connection = max(connection_i, connection_j)
                        
                        if not bool(connection_similarity_dict.get(str(max_connection))):
                            connection_similarity_dict.update({str(max_connection): [similarity]})
                        else:
                            similarity_list = connection_similarity_dict[str(max_connection)]
                            similarity_list.append(similarity)
                            similarity_list = sorted(similarity_list)[::-1]
                            connection_similarity_dict.update({str(max_connection): similarity_list})
                            
                similarities_to_be_pruned = []
                keys = [int(c) for c in connection_similarity_dict.keys()]
                sorted_connection_num = sorted(keys)
                
                for c in sorted_connection_num:
                    for s in connection_similarity_dict[str(c)]:
                        similarities_to_be_pruned.append(s)
            
            # Normal Way Pruning
            similarity_dict_ = copy.deepcopy(similarity_dict)
            pruned_dict_ = copy.deepcopy(pruned_dict)
            all_mask_ = copy.deepcopy(all_mask)
            depth_can_not_be_pruned = {}
            for layer_iter, layer in enumerate(sorted_layer):
                depth_can_not_be_pruned.update({layer: []})
            for pruned_similarity_iter in range(pruned_similarity_num):
                # similarity
                if cut_connection_less_to_more:
                    similarity = similarities_to_be_pruned[pruned_similarity_iter]
                else:
                    similarity = sorted_similarities[pruned_similarity_iter]
                for _, index in enumerate(similarity_dict_[str(similarity)].keys()):
                    if similarity_dict_[str(similarity)][index]['pruned']:
                        continue
                    similarity_dict_[str(similarity)][index].update({'pruned': True})
                    # past layer
                    past_layer = similarity_dict_[str(similarity)][index]['past_layer']
                    past_mask  = all_mask_[past_layer]
                    past_weight = all_weight[past_layer]
                    past_output_shape = all_output_shape[past_layer]
                    # current layer
                    layer = similarity_dict_[str(similarity)][index]['layer']
                    mask  = all_mask_[layer]
                    weight = all_weight[layer]
                    output_shape = all_output_shape[layer]
                    beta = all_beta[layer]
                    # depth
                    depth_i = similarity_dict_[str(similarity)][index]['depth_i']
                    depth_j = similarity_dict_[str(similarity)][index]['depth_j']
                    is_depth_i_pruned = pruned_dict_[past_layer][str(depth_i)]
                    is_depth_j_pruned = pruned_dict_[past_layer][str(depth_j)]
                    
                    # Detemine which channel to be pruned
                    if all_be_pruned:
                        # Assign Zero to mask
                        pruned_dict_[past_layer].update({str(depth_i): True})
                        past_mask[:, :, :, depth_i] = 0
                        mask[:, :, depth_i, :] = 0
                        pruned_dict_[past_layer].update({str(depth_j): True})
                        past_mask[:, :, :, depth_j] = 0
                        mask[:, :, depth_j, :] = 0
                    else:
                        if not is_depth_i_pruned and not is_depth_j_pruned:
                            # -- Compare Connection --
                            if is_connection and connection[layer][str(depth_i)] < connection[layer][str(depth_j)]:
                                if not discard_feature and depth_j in depth_can_not_be_pruned[layer]:
                                    break
                                pruned_depth = depth_i
                            elif is_connection and connection[layer][str(depth_i)] > connection[layer][str(depth_j)]:
                                if not discard_feature and depth_i in depth_can_not_be_pruned[layer]:
                                    break
                                pruned_depth = depth_j
                            else:
                                # -- Can Not be Pruned --
                                if not discard_feature and depth_j in depth_can_not_be_pruned[layer]:
                                    pruned_depth = depth_i
                                elif not discard_feature and depth_i in depth_can_not_be_pruned[layer]:
                                    pruned_depth = depth_j
                                else:
                                    # -- Beta -- 
                                    if is_beta and np.abs(beta[depth_i]) < beta_threshold and np.abs(beta[depth_j]) >= beta_threshold:
                                        pruned_depth = depth_i
                                    elif is_beta and np.abs(beta[depth_i]) >= beta_threshold and np.abs(beta[depth_j]) < beta_threshold:
                                        pruned_depth = depth_j
                                    else:   
                                        # -- Magnitude --
                                        sum_of_past_layer_filter_i = np.sum(np.abs(past_weight[:, :, :, depth_i] * past_mask[:, :, :, depth_i]))
                                        sum_of_past_layer_filter_j = np.sum(np.abs(past_weight[:, :, :, depth_j] * past_mask[:, :, :, depth_j]))
                                        sum_of_depth_i = np.sum(np.abs(weight[:, :, depth_i, :] * mask[:, :, depth_i, :]))
                                        sum_of_depth_j = np.sum(np.abs(weight[:, :, depth_j, :] * mask[:, :, depth_j, :]))
                                        sum_of_pruned_depth_i = sum_of_past_layer_filter_i + sum_of_depth_i
                                        sum_of_pruned_depth_j = sum_of_past_layer_filter_j + sum_of_depth_j
                                        if sum_of_pruned_depth_i <= sum_of_pruned_depth_j:
                                            pruned_depth = depth_i
                                        else:
                                            pruned_depth = depth_j
                            
                            # Update Dict: depth_can_not_be_pruned
                            if not discard_feature:
                                unpruned_similar_depth = [] # Check the simlilar depths of the pruned_depth is unpruned or pruned
                                for depth in similar_depth[layer][str(pruned_depth)]:
                                    if not pruned_dict_[past_layer][depth]:
                                        unpruned_similar_depth.append(depth)
                                if len(unpruned_similar_depth) == 1: # If only one depth is unpruned, we set this depth can not be pruned.
                                    depth_can_not_be_pruned[layer].append(unpruned_similar_depth[0])
                                    #print("{} -> {}" .format(layer, depth_can_not_be_pruned[layer]))
                            
                            # Assign Zero to mask
                            pruned_dict_[past_layer].update({str(pruned_depth): True})
                            past_mask[:, :, :, pruned_depth] = 0
                            mask[:, :, pruned_depth, :] = 0
                        
            # Calculate pruned_size
            # after_prune_weights_size
            after_prune_weights_size = 0
            for layer_iter, layer in enumerate(sorted_layer):
                after_prune_weights_size += np.sum(all_mask_[layer])
            #print("%f, %f" %(before_prune_weights_size-after_prune_weights_size, original_size*pruning_propotion))
            if (before_prune_weights_size-after_prune_weights_size) >= original_size*pruning_propotion:
                break
    
    #==============================#
    #   Connection_num Directory   #
    #==============================#
    # Build the connection_num directory
    if cut_connection_less_to_more:
        similarity_dict_ = copy.deepcopy(similarity_dict)
        pruned_dict_ = copy.deepcopy(pruned_dict)
        connection_similarity_dict = {}
        for pruned_similarity_iter in range(pruned_similarity_num):
            similarity = sorted_similarities[pruned_similarity_iter]
            for _, index in enumerate(similarity_dict_[str(similarity)].keys()):
                if similarity_dict_[str(similarity)][index]['pruned']:
                    continue
                similarity_dict_[str(similarity)][index].update({'pruned': True})
                # current layer
                layer = similarity_dict_[str(similarity)][index]['layer']
                # depth
                depth_i = similarity_dict_[str(similarity)][index]['depth_i']
                depth_j = similarity_dict_[str(similarity)][index]['depth_j']
                connection_i = connection[layer][str(depth_i)]
                connection_j = connection[layer][str(depth_j)]
                max_connection = max(connection_i, connection_j)
                
                if not bool(connection_similarity_dict.get(str(max_connection))):
                    connection_similarity_dict.update({str(max_connection): [similarity]})
                else:
                    similarity_list = connection_similarity_dict[str(max_connection)]
                    similarity_list.append(similarity)
                    similarity_list = sorted(similarity_list)[::-1]
                    connection_similarity_dict.update({str(max_connection): similarity_list})
                    
        similarities_to_be_pruned = []
        keys = [int(c) for c in connection_similarity_dict.keys()]
        sorted_connection_num = sorted(keys)
        
        for c in sorted_connection_num:
            for s in connection_similarity_dict[str(c)]:
                similarities_to_be_pruned.append(s)
    
    #=================#
    #  Prune Weights  #
    #=================#
    t = PrettyTable(['Similarity', 'Layer', 'Channel', 'Computation'])
    t.align = 'l'
    pruned_num = 0
    depth_can_not_be_pruned = {}
    for layer_iter, layer in enumerate(sorted_layer):
        depth_can_not_be_pruned.update({layer: []})
    for pruned_similarity_iter in range(pruned_similarity_num):
        # similarity
        if cut_connection_less_to_more:
            similarity = similarities_to_be_pruned[pruned_similarity_iter]
        else:
            similarity = sorted_similarities[pruned_similarity_iter]
        for _, index in enumerate(similarity_dict[str(similarity)].keys()):
            if similarity_dict[str(similarity)][index]['pruned']:
                continue
            # past layer
            past_layer = similarity_dict[str(similarity)][index]['past_layer']
            past_mask = all_mask[past_layer]    
            past_weight = all_weight[past_layer]
            past_output_shape = all_output_shape[past_layer]
            # current layer
            layer = similarity_dict[str(similarity)][index]['layer']
            mask = all_mask[layer]
            weight = all_weight[layer]
            output_shape = all_output_shape[layer]
            beta = all_beta[layer]
            # depth
            depth_i = similarity_dict[str(similarity)][index]['depth_i']
            depth_j = similarity_dict[str(similarity)][index]['depth_j']
            is_depth_i_pruned = pruned_dict[past_layer][str(depth_i)]
            is_depth_j_pruned = pruned_dict[past_layer][str(depth_j)]
            
            # Detemine which channel to be pruned
            if all_be_pruned:
                for pruned_depth in [depth_i, depth_j]:
                    # Assign Zero to mask
                    pruned_dict[past_layer].update({str(pruned_depth): True})
                    past_mask[:, :, :, pruned_depth] = 0
                    mask[:, :, pruned_depth, :] = 0
                    
                    # Compute Reduced Computation
                    computation = 0
                    # Past Layer
                    # Remove the past pruned mask (for computing computation)
                    #--------------------------------------------------------
                    past_unpruned_mask = past_mask
                    past_channel_num = np.shape(past_mask)[3]
                    past_depth = np.shape(past_mask)[2]
                    past_pruned_channels = []
                    past_pruned_depth = []
                    # (channel)
                    for i in range(past_channel_num):
                        if np.sum(past_mask[:, :, :, i]) == 0:
                            past_pruned_channels = np.append(past_pruned_channels, [i])
                    past_unpruned_mask = np.delete(past_unpruned_mask, past_pruned_channels, axis = 3)
                    # (depth)
                    for i in range(past_depth):
                        if np.sum(past_mask[:, :, i, :]) == 0:
                            past_pruned_depth = np.append(past_pruned_depth, [i])
                    past_unpruned_mask = np.delete(past_unpruned_mask, past_pruned_depth, axis = 2)
                    #--------------------------------------------------------
                    past_weight_shape = np.shape(past_unpruned_mask)
                    computation = computation + past_weight_shape[0] * past_weight_shape[1] * past_weight_shape[2] * past_output_shape[2] * past_output_shape[3]
                    # Current Layer
                    # Remove the current pruned mask (for computing computation)
                    #-----------------------------------------------------------
                    current_unpruned_mask = mask
                    current_channel_num = np.shape(mask)[3]
                    current_depth = np.shape(mask)[2]
                    current_pruned_channels = []
                    current_pruned_depth = []
                    # (channel)
                    for i in range(current_channel_num):
                        if np.sum(mask[:, :, :, i]) == 0:
                            current_pruned_channels = np.append(current_pruned_channels, [i])
                    current_unpruned_mask = np.delete(current_unpruned_mask, current_pruned_channels, axis = 3)
                    # (depth)
                    for i in range(current_depth):
                        if np.sum(mask[:, :, i, :]) == 0:
                            current_pruned_depth = np.append(current_pruned_depth, [i])
                    current_unpruned_mask = np.delete(current_unpruned_mask, current_pruned_depth, axis = 2)
                    #-----------------------------------------------------------
                    weight_shape = np.shape(current_unpruned_mask)
                    computation = computation + weight_shape[0] * weight_shape[1] * weight_shape[3] * output_shape[2] * output_shape[3]
                    
                    # Update List: pruned_weights_info
                    info = {past_layer: {'channel': pruned_depth}, layer: {'depth': pruned_depth}, 'computation': computation}
                    pruned_weights_info.append(info)
                    t.add_row([similarity*1e-8, layer, 'depth' + str(pruned_depth), computation])
            else:
                if not is_depth_i_pruned and not is_depth_j_pruned:
                    similarity_dict[str(similarity)][index].update({'pruned': True})
                    # -- Compare Connection --
                    if is_connection and connection[layer][str(depth_i)] < connection[layer][str(depth_j)]:
                        if not discard_feature and depth_j in depth_can_not_be_pruned[layer]:
                            break
                        pruned_depth = depth_i
                    elif is_connection and connection[layer][str(depth_i)] > connection[layer][str(depth_j)]:
                        if not discard_feature and depth_i in depth_can_not_be_pruned[layer]:
                            break
                        pruned_depth = depth_j
                    else:
                        # -- Can Not be Pruned --
                        if not discard_feature and depth_j in depth_can_not_be_pruned[layer]:
                            pruned_depth = depth_i
                        elif not discard_feature and depth_i in depth_can_not_be_pruned[layer]:
                            pruned_depth = depth_j
                        else:
                            # -- Beta -- 
                            if is_beta and np.abs(beta[depth_i]) < beta_threshold and np.abs(beta[depth_j]) >= beta_threshold:
                                pruned_depth = depth_i
                            elif is_beta and np.abs(beta[depth_i]) >= beta_threshold and np.abs(beta[depth_j]) < beta_threshold:
                                pruned_depth = depth_j
                            else:   
                                # -- Magnitude --
                                sum_of_past_layer_filter_i = np.sum(np.abs(past_weight[:, :, :, depth_i] * past_mask[:, :, :, depth_i]))
                                sum_of_past_layer_filter_j = np.sum(np.abs(past_weight[:, :, :, depth_j] * past_mask[:, :, :, depth_j]))
                                sum_of_depth_i = np.sum(np.abs(weight[:, :, depth_i, :] * mask[:, :, depth_i, :]))
                                sum_of_depth_j = np.sum(np.abs(weight[:, :, depth_j, :] * mask[:, :, depth_j, :]))
                                sum_of_pruned_depth_i = sum_of_past_layer_filter_i + sum_of_depth_i
                                sum_of_pruned_depth_j = sum_of_past_layer_filter_j + sum_of_depth_j
                                
                                if sum_of_pruned_depth_i <= sum_of_pruned_depth_j:
                                    pruned_depth = depth_i
                                else:
                                    pruned_depth = depth_j
                    
                    # Update Dict: depth_can_not_be_pruned
                    if not discard_feature:
                        unpruned_similar_depth = []
                        for depth in similar_depth[layer][str(pruned_depth)]:
                            if not pruned_dict_[past_layer][depth]:
                                unpruned_similar_depth.append(depth)
                        if len(unpruned_similar_depth) == 1:
                            depth_can_not_be_pruned[layer].append(unpruned_similar_depth[0])
                            #print("{} -> {}" .format(layer, depth_can_not_be_pruned[layer]))
    
                    # Assign Zero to mask
                    pruned_dict[past_layer].update({str(pruned_depth): True})
                    past_mask[:, :, :, pruned_depth] = 0
                    mask[:, :, pruned_depth, :] = 0
                    
                    # Compute Reduced Computation
                    computation = 0
                    # Past Layer
                    # Remove the past pruned mask (for computing computation)
                    #--------------------------------------------------------
                    past_unpruned_mask = past_mask
                    past_channel_num = np.shape(past_mask)[3]
                    past_depth = np.shape(past_mask)[2]
                    past_pruned_channels = []
                    past_pruned_depth = []
                    # (channel)
                    for i in range(past_channel_num):
                        if np.sum(past_mask[:, :, :, i]) == 0:
                            past_pruned_channels = np.append(past_pruned_channels, [i])
                    past_unpruned_mask = np.delete(past_unpruned_mask, past_pruned_channels, axis = 3)
                    # (depth)
                    for i in range(past_depth):
                        if np.sum(past_mask[:, :, i, :]) == 0:
                            past_pruned_depth = np.append(past_pruned_depth, [i])
                    past_unpruned_mask = np.delete(past_unpruned_mask, past_pruned_depth, axis = 2)
                    #--------------------------------------------------------
                    past_weight_shape = np.shape(past_unpruned_mask)
                    computation = computation + past_weight_shape[0] * past_weight_shape[1] * past_weight_shape[2] * past_output_shape[2] * past_output_shape[3]
                    # Current Layer
                    # Remove the current pruned mask (for computing computation)
                    #-----------------------------------------------------------
                    current_unpruned_mask = mask
                    current_channel_num = np.shape(mask)[3]
                    current_depth = np.shape(mask)[2]
                    current_pruned_channels = []
                    current_pruned_depth = []
                    # (channel)
                    for i in range(current_channel_num):
                        if np.sum(mask[:, :, :, i]) == 0:
                            current_pruned_channels = np.append(current_pruned_channels, [i])
                    current_unpruned_mask = np.delete(current_unpruned_mask, current_pruned_channels, axis = 3)
                    # (depth)
                    for i in range(current_depth):
                        if np.sum(mask[:, :, i, :]) == 0:
                            current_pruned_depth = np.append(current_pruned_depth, [i])
                    current_unpruned_mask = np.delete(current_unpruned_mask, current_pruned_depth, axis = 2)
                    #-----------------------------------------------------------
                    weight_shape = np.shape(current_unpruned_mask)
                    computation = computation + weight_shape[0] * weight_shape[1] * weight_shape[3] * output_shape[2] * output_shape[3]
                    
                    # Update List: pruned_weights_info
                    info = {past_layer: {'channel': pruned_depth}, layer: {'depth': pruned_depth}, 'computation': computation}
                    pruned_weights_info.append(info)
                    t.add_row([similarity*1e-8, layer, 'depth' + str(pruned_depth), computation])
            break
    #print(t)
    #===============#
    #  Update Mask  #
    #===============#
    print('Updating Masks ... ')
    for layer_iter, layer in enumerate(sorted_layer):
        sess.run(tf.assign(prune_info_dict[layer]['mask'], all_mask[layer]))
        
    #===================#
    #  Show the Result  #
    #===================#
    for layer_iter, layer in enumerate(sorted_layer):
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
    
    
def filter_prune_by_magnitude( # not finished
    prune_info_dict,
    pruning_propotion,
    #pruning_layer,
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
            if angles_iter != 0:
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
    t = PrettyTable(['Pruned Number', 'Angle', 'layer', 'channel', 'Computation'])
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

def filter_prune_by_angleIII(
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
    can_be_pruned_dict = {}
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
                        
                        # Dictionary for each angle
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
                        
                        # Dictionary for checking its partner has been pruned or not
                        if not bool(can_be_pruned_dict.get(layer)):
                            can_be_pruned_dict.update({layer: {str(i): True, str(j): True}})
                        else:
                            if not bool(can_be_pruned_dict.get(str(i))):
                                can_be_pruned_dict[layer].update({str(i): True})
                            if not bool(can_be_pruned_dict.get(str(j))):
                                can_be_pruned_dict[layer].update({str(j): True})
                            
                        # Record all the angles for sorting
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
                is_depth_i_been_pruned = False # For checking depth_i has been pruned or not, True means been pruned
                is_depth_j_been_pruned = False # For checking depth_j has been pruned or not, True means been pruned
                weights_to_be_pruned.update({layer: {str(depth_i): 1, str(depth_j): 1}})
            else:
                is_depth_i_been_pruned = bool(weights_to_be_pruned[layer].get(str(depth_i)))
                is_depth_j_been_pruned = bool(weights_to_be_pruned[layer].get(str(depth_j)))
                if not is_depth_i_been_pruned:
                    weights_to_be_pruned[layer].update({str(depth_i): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_i): weights_to_be_pruned[layer][str(depth_i)]+1})
                if not is_depth_j_been_pruned:
                    weights_to_be_pruned[layer].update({str(depth_j): 1})
                else:
                    weights_to_be_pruned[layer].update({str(depth_j): weights_to_be_pruned[layer][str(depth_j)]+1})
            
            # If one of depth_i and depth_j is not been pruned, we consider one of them will be pruned then.
            if not is_depth_i_been_pruned or not is_depth_j_been_pruned:
                #if bool(dict_[str(angle)][index].get('past_layer')):
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, depth_i]))
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
            #print(pruned_num)
        if pruned_num >= total_num * pruning_propotion:
            break
            
    # Prune the corresponding weights
    t = PrettyTable(['Pruned Number', 'Angle', 'layer', 'channel', 'Computation'])
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
                # depth_i can be pruned and depth_j cannot be pruned, so prune the depth_i
                if can_be_pruned_dict[layer][str(depth_i)] and not can_be_pruned_dict[layer][str(depth_j)]:
                    if weights_to_be_pruned[layer][str(depth_i)] <= weights_to_be_pruned[layer][str(depth_j)]:
                        pruned_depth = depth_i
                    else:
                        break
                # depth_i cannot be pruned and depth_j can be pruned, so prune the depth_j
                elif not can_be_pruned_dict[layer][str(depth_i)] and can_be_pruned_dict[layer][str(depth_j)]:
                    if weights_to_be_pruned[layer][str(depth_i)] >= weights_to_be_pruned[layer][str(depth_j)]:
                        pruned_depth = depth_j
                    else:
                        break
                # Both of depth_i and depth_j can be pruned
                elif can_be_pruned_dict[layer][str(depth_i)] and can_be_pruned_dict[layer][str(depth_j)]:
                    # Compare the connection number first
                    if weights_to_be_pruned[layer][str(depth_i)] < weights_to_be_pruned[layer][str(depth_j)]:
                        pruned_depth = depth_i
                    elif weights_to_be_pruned[layer][str(depth_i)] > weights_to_be_pruned[layer][str(depth_j)]:
                        pruned_depth = depth_j
                    # If the connection number are the same, compare their sum of absolute weights value
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
                # Both of depth_i and depth_j cannot be pruned
                else:
                    break
                    
                # Update the can_be_prund_dict
                can_be_pruned_dict[layer].update({str(pruned_depth): False})
                
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
                # past layer
                past_weights_mask[:, :, :, pruned_depth] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(past_weights_mask[:, :, :, depth_i]))
                past_weights_shape = np.shape(unpruned_past_mask)
                computation = computation + past_weights_shape[0] * past_weights_shape[1] * past_weights_shape[2] * past_outputs_shape[2] * past_outputs_shape[3]
                tmp.update({past_layer: {'channel': pruned_depth}})
                # current layer
                current_weights_mask[:, :, pruned_depth, :] = 0
                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
                current_weights_shape = np.shape(unpruned_current_mask)
                computation = computation + current_weights_shape[0] * current_weights_shape[1] * current_weights_shape[3] * current_outputs_shape[2] * current_outputs_shape[3]
                tmp.update({layer: {'depth': pruned_depth}})
                
                # pruned_weights_info
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

def filter_prune_by_angleIV(
    prune_info_dict,
    pruning_propotion,
    pruned_weights_info,
    sess
    ):
    
    beta_threshold = 0.01
    
    key = prune_info_dict.keys()
    sorted_layer = np.sort(np.array([int(key[i].split('layer')[-1]) for i in range(len(key))]))
    sorted_layer = ['layer' + str(sorted_layer[i]) for i in range(len(key))]

    # Load all wegihts and masks
    all_weights = {}
    all_mask = {}
    all_outputs_shape = {}
    all_stride = {}
    all_beta = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})
        all_beta.update({layer: sess.run(prune_info_dict[layer]['beta'])})
    
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
            beta = all_beta[past_layer]
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
                        
                        if beta[i] < beta_threshold or beta[j] < beta_threshold:
                            angle = angle * 1.5
                            
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
            if angles_iter != 0:
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
    t = PrettyTable(['Pruned Number', 'Angle', 'layer', 'channel', 'Computation'])
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
            beta = all_beta[past_layer]
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
                    # Beta 
                    if beta[depth_i] < beta_threshold or beta[depth_j] < beta_threshold:
                        if beta[depth_i] < beta[depth_j]:
                            pruned_depth = depth_i
                        else:
                            pruned_depth = depth_j
                    # Magnitude
                    else:
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
                        if child != layer:
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
 
def mobileNet_filter_prune_by_angle(
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
    all_is_depthwise = {}
    all_parents = {}
    all_children = {}
    for layer_iter in range(len(sorted_layer)):
        layer = sorted_layer[layer_iter]
        all_weights.update({layer: sess.run(prune_info_dict[layer]['weights'])})
        all_mask.update({layer: sess.run(prune_info_dict[layer]['mask'])})
        all_outputs_shape.update({layer: prune_info_dict[layer]['outputs'].get_shape().as_list()})
        all_stride.update({layer: prune_info_dict[layer]['stride']})
        all_is_depthwise.update({layer: prune_info_dict[layer]['is_depthwise']})
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
        if not all_is_depthwise[layer]:
            channel_num = np.shape(mask)[3]
            tmp_dict = {}
            for i in range(channel_num):
                tmp_dict.update({str(i): np.mean(mask[:,:,:,i]) == 0})
        else:
            channel_num = np.shape(mask)[2]
            tmp_dict = {}
            for i in range(channel_num):
                tmp_dict.update({str(i): np.mean(mask[:,:,i,:]) == 0})
        pruned_dict.update({layer: tmp_dict})
        
    # Build the dictionary for angle
    dict = {}
    iter = 0
    for layer_iter in range(1, len(sorted_layer)):
        # current layer
        layer = sorted_layer[layer_iter]
        print(layer, end=" ")
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

        tStart = time.time()
        # calculate angle
        if weights.size != 0:
            angles_iter = 0
            depth = np.shape(weights)[-1]
            for i in range(depth):
                angles_per_i_iter = 0
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
            """            
                        if iter == 0:
                            total_angle = np.array([angle])
                        else:
                            total_angle = np.concatenate([total_angle, np.array([angle])])
                        iter = iter + 1    
            """ 
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
    all_mask_ = copy.deepcopy(all_mask)
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
            # Build the pruning tree
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
            # Prune the channel
            if not is_depth_i_appear or not is_depth_j_appear:
                # Check the parent layer channel should be pruned or not
                if bool(all_parents.get(layer)):
                    # random choose depth i or j to be pruned
                    rand_index = random.randrange(0, 2)
                    if rand_index == 0:
                        pruned_depth = depth_i
                    else:
                        pruned_depth = depth_j
                    interval = pruned_depth
                    
                    all_mask_[layer][:, :, pruned_depth, :] = 0
                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(current_weights_mask[:, :, depth_i, :]))
                    
                    # If current layer is depthwise, 
                    # then prune the next layer until the next layer is not depthwise
                    if all_is_depthwise[layer]:
                        layer_index = sorted_layer.index(layer)
                        while(1):
                            layer_index = layer_index + 1
                            next_layer = sorted_layer[layer_index]
                            if not all_is_depthwise[next_layer]:
                                all_mask_[next_layer][:, :, interval, :] = 0
                                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[next_layer][:, :, interval, :]))
                                break
                            else:
                                all_mask_[next_layer][:, :, interval, :] = 0
                                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[next_layer][:, :, interval, :]))

                    # Find the head for checking the head to be pruned or not
                    for _, parent in enumerate(all_parents[layer]):
                        if pruned_depth >= all_parents[layer][parent] and (pruned_depth - all_parents[layer][parent]) <= interval:
                            interval = pruned_depth - all_parents[layer][parent]
                            head = parent 
                    # Check through all children of the head
                    is_head_channel_to_be_pruned = True
                    for _, child in enumerate(all_children[head]):
                        start_point = all_children[head][child]
                        position = start_point + interval
                        if np.sum(all_mask_[child][:, :, position, :]) != 0:
                            is_head_channel_to_be_pruned = False
                            break
                    if is_head_channel_to_be_pruned:
                        if not all_is_depthwise[head]:
                            all_mask_[head][:, :, :, interval] = 0
                            pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head][:, :, :, interval]))
                        else: # The head is depthwise
                            all_mask_[head][:, :, interval, :] = 0
                            pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head][:, :, interval, :]))
                            # Prune the past layer until the past layer is not depthwise
                            layer_index = sorted_layer.index(head)
                            while(1):
                                layer_index = layer_index - 1
                                head_past_layer = sorted_layer[layer_index]
                                if not all_is_depthwise[head_past_layer]:
                                    all_mask_[head_past_layer][:, :, :, interval] = 0
                                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head_past_layer][:, :, :, interval]))
                                    break
                                else:
                                    all_mask_[head_past_layer][:, :, interval, :] = 0
                                    pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head_past_layer][:, :, interval, :]))
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
                
                # If current layer is depthwise, 
                # then prune the next layer until the next layer is not depthwise
                if all_is_depthwise[layer]:
                    layer_index = sorted_layer.index(layer)
                    while(1):
                        layer_index = layer_index + 1
                        next_layer = sorted_layer[layer_index]
                        if not all_is_depthwise[next_layer]:
                            all_mask[next_layer][:, :, pruned_depth, :] = 0
                            pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[next_layer][:, :, pruned_depth, :]))
                            break
                        else:
                            all_mask[next_layer][:, :, pruned_depth, :] = 0
                            pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[next_layer][:, :, pruned_depth, :]))
                # Past 
                # Check the parent layer channel should be pruned or not
                if bool(all_parents.get(layer)):
                    interval = pruned_depth             
                    for _, parent in enumerate(all_parents[layer]):
                        if pruned_depth >= all_parents[layer][parent] and (pruned_depth - all_parents[layer][parent]) <= interval:
                            interval = pruned_depth - all_parents[layer][parent]
                            head = parent
                            
                    # Check through all children
                    is_head_channel_to_be_pruned = True
                    for _, child in enumerate(all_children[head]):
                        start_point = all_children[head][child]
                        position = start_point + interval
                        if np.sum(all_mask[child][:, :, position, :]) != 0:
                            is_head_channel_to_be_pruned = False
                            break
                if is_head_channel_to_be_pruned:
                    pruned_dict[head].update({str(pruned_depth): True})
                    head_weights_mask  = all_mask[head]
                    if not all_is_depthwise[head]:
                        head_weights_mask[:, :, :, interval] = 0
                        pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head][:, :, :, interval]))
                    else:
                        head_weights_mask[:, :, interval, :] = 0
                        pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head][:, :, interval, :]))
                        # Prune the past layer until the past layer is not depthwise
                        layer_index = sorted_layer.index(head)
                        while(1):
                            layer_index = layer_index - 1
                            head_past_layer = sorted_layer[layer_index]
                            if not all_is_depthwise[head_past_layer]:
                                all_mask[head_past_layer][:, :, :, interval] = 0
                                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head_past_layer][:, :, :, interval]))
                                break
                            else:
                                all_mask[head_past_layer][:, :, interval, :] = 0
                                pruned_num = pruned_num + reduce(lambda x, y: x*y, np.shape(all_mask[head_past_layer][:, :, interval, :]))                           
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

#========================#
#   Testing Components   #
#========================#
def compute_accuracy(
    xs, 
    ys, 
    is_training,
    is_quantized_activation,
    is_fast_mode_dict,
    complexity_mode,
    complexity_mode_dict,
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
        #dict_test.update({is_fast_mode['layer'+str(layer)]: Model_dict['layer'+str(layer)]['IS_DIVERSITY']=='TRUE'}) 
        dict_test.update(is_fast_mode_dict)
    
    for i in range(batch_num): 
        dict = dict_test
        DIVERSIFY_MODE_list = [Model_dict[D]['DIVERSIFY_MODE'] for D in Model_dict.keys()]
        if not all(D=='None' for D in DIVERSIFY_MODE_list):
            for layer in range(len(Model_dict)):
                dict_test.update({is_training['layer'+str(layer)]: False})
        else:
            dict_test.update({is_training: False}) #xs: v_xs_part,
        
        # Complexity Mode
        dict_test.update(complexity_mode_dict)
        """
        for layer in range(len(Model_dict)):
            cond0 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '4'
            cond1 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '5'
            cond2 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '6'
            cond3 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '7'
            cond4 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '8'
            cond5 = Model_dict['layer'+str(layer)]['DIVERSIFY_MODE'] == '9'
            if cond0 or cond1 or cond2 or cond3 or cond4 or cond5:
                dict_test.update({complexity_mode['layer'+str(layer)]: complexity_mode_dict['layer'+str(layer)]})
        """ 
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
        #print(accuracy_top1)
        ##print("\r{} / {} -> \033[1;32mAccuracy\033[0m: {}" .format((i+1)*test_batch_size, data_num, accuracy_top1), end = "") 

    ##print("")    
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
    # Model_dict
    Model_dict = Model_dict_Generator(model_path + 'model.csv', class_num)
    
    # Placeholder
    ## -- is_training --
    is_training = tf.placeholder(tf.bool)
    
    ## -- learning Rate --
    learning_rate = tf.placeholder(tf.float32)

    ## -- is_quantized_activation --
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
        
    ## -- is_ternary --
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    ## -- is_binary --
    is_binary = {}
    for layer in range(len(Model_dict)):
        is_binary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    # net
    net = tf.ones([1, H_Resize, W_Resize, 3])
    
    # model
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_binary               = is_binary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = 0.0,
        data_format             = "NCHW",
        reuse                   = None)
    """
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict_, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_binary               = is_binary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = HP['Dropout_Rate'],
        data_format             = data_format,
        reuse                   = None)
    """
    
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
    # Model_dict
    Model_dict = Model_dict_Generator(model_path + 'model.csv', class_num)
    
    # Placeholder
    ## -- is_training --
    is_training = tf.placeholder(tf.bool)
    
    ## -- learning Rate --
    learning_rate = tf.placeholder(tf.float32)

    ## -- is_quantized_activation --
    is_quantized_activation = {}
    for layer in range(len(Model_dict)):
        is_quantized_activation.update({'layer%d'%layer : tf.placeholder(tf.bool)}) 
        
    ## -- is_ternary --
    is_ternary = {}
    for layer in range(len(Model_dict)):
        is_ternary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    ## -- is_binary --
    is_binary = {}
    for layer in range(len(Model_dict)):
        is_binary.update({'layer%d'%layer : tf.placeholder(tf.bool)})
    
    # net
    net = tf.ones([1, H_Resize, W_Resize, 3])
    
    # model
    prediction, Analysis, max_parameter, inputs_and_kernels, prune_info_dict = Model_dict_Decoder(
        net                     = net, 
        Model_dict              = Model_dict, 
        is_training             = is_training,
        is_ternary              = is_ternary,
        is_binary               = is_binary,
        is_quantized_activation = is_quantized_activation,
        DROPOUT_RATE            = 0.0,
        data_format             = "NCHW",
        reuse                   = None)
    # weights tensor
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
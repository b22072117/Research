import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import argparse
import utils
import Model
import sys
from sys import platform
import os
import time
#========================#
#    Global Parameter    #
#========================#
parser = argparse.ArgumentParser()

parser.add_argument('--Dataset'        , type = str, default = 'cifar10')
parser.add_argument('--Model_1st'      , type = str, default = 'ResNet')
parser.add_argument('--Model_2nd'      , type = str, default = '20_cifar10_2')
parser.add_argument('--BatchSize'      , type = int, default = 128)
parser.add_argument('--Epoch'          , type = int, default = 250)
parser.add_argument('--epochs_per_eval', type = int, default = 10)


FLAGs = parser.parse_args()

Model_Name = FLAGs.Model_1st + '_' + FLAGs.Model_2nd
IS_HYPERPARAMETER_OPT = False

print('\n\033[1;32;40mMODEL NAME\033[0m :\033[1;37;40m {MODEL_NAME}\033[0m' .format(MODEL_NAME = Model_Name))
#============#
#    Path    #
#============#
if platform == 'win32':
    # For Loading Dataset
    Dataset_Path = '..\\dataset\\' + FLAGs.Dataset
    
    if FLAGs.Dataset=='ade20k':
        Dataset_Path = Dataset_Path + '\\ADEChallengeData2016'
    Y_pre_Path   = FLAGs.Model_1st + '_Y_pre\\' + FLAGs.Dataset
    
    # For Saving Result Picture of Testing
    train_target_path = '..\\result\\' + FLAGs.Dataset + '\\' + FLAGs.Model_1st + '\\' 
    valid_target_path = '..\\result\\' + FLAGs.Dataset + '\\' + FLAGs.Model_1st + '\\' 
    test_target_path  = '..\\result\\' + FLAGs.Dataset + '\\' + FLAGs.Model_1st + '\\' 
    
    # For Saving Result in .npz file
    train_Y_pre_path  = 'Y_pre\\' + FLAGs.Model_1st + '\\' + FLAGs.Dataset + '\\'
    valid_Y_pre_path  = 'Y_pre\\' + FLAGs.Model_1st + '\\' + FLAGs.Dataset + '\\'
    test_Y_pre_path   = 'Y_pre\\' + FLAGs.Model_1st + '\\' + FLAGs.Dataset + '\\'
    
    # For Loading Trained Model
    Model_Path = None
    Model = None
else:
    # For Loading Dataset
    Dataset_Path = '../dataset/' + FLAGs.Dataset
    
    if FLAGs.Dataset=='ade20k':
        Dataset_Path = Dataset_Path + '/ADEChallengeData2016'
    Y_pre_Path   = FLAGs.Model_1st + '_Y_pre/' + FLAGs.Dataset
    
    # For Saving Result Picture of Testing
    train_target_path = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 
    valid_target_path = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 
    test_target_path  = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 
    
    # For Saving Result in .npz file
    train_Y_pre_path  = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'
    valid_Y_pre_path  = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'
    test_Y_pre_path   = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'
    
    # For Loading Trained Model
    Model_Path = None
    Model = None

#==============# 
#    Define    #
#==============#
def main(argv):
    print("start!")
    # -- Training --
    Model_Path = None
    Model = None
    Global_Epoch = 0
    for _ in range(FLAGs.Epoch//FLAGs.epochs_per_eval):
        Model_Path, Model = utils.run_training( 
            Hyperparameter        = None                 ,               
            FLAGs                 = FLAGs                ,
            Epoch                 = FLAGs.epochs_per_eval,
            Global_Epoch          = Global_Epoch         ,
            IS_HYPERPARAMETER_OPT = IS_HYPERPARAMETER_OPT,
            Dataset_Path          = Dataset_Path         ,
            Y_pre_Path            = Y_pre_Path           ,
            trained_model_path    = Model_Path           ,
            trained_model         = Model                )
    
        # -- Testing --
        test_accuracy = utils.run_testing( 
            Hyperparameter        = None                 ,               
            FLAGs                 = FLAGs                ,
            IS_HYPERPARAMETER_OPT = IS_HYPERPARAMETER_OPT,  
            Dataset_Path          = Dataset_Path         ,
            testing_model_path    = Model_Path           ,
            testing_model         = Model                ,
            train_target_path     = train_target_path    ,
            valid_target_path     = valid_target_path    ,
            test_target_path      = test_target_path     ,
            train_Y_pre_path      = train_Y_pre_path     ,
            valid_Y_pre_path      = valid_Y_pre_path     ,
            test_Y_pre_path       = test_Y_pre_path      )
            
        Global_Epoch = Global_Epoch + FLAGs.epochs_per_eval
        
if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)


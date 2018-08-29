import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import argparse
import utils_binary as utils
import Model_binary as Model
#import utils
#import Model
import sys
from sys import platform
import os
import time
#========================#
#    Global Parameter    #
#========================#
parser = argparse.ArgumentParser()

parser.add_argument('--Dataset'                     , type = str, default = 'cifar10')
parser.add_argument('--Model_1st'                   , type = str, default = 'ResNet')
parser.add_argument('--Model_2nd'                   , type = str, default = '20_cifar10_2')
parser.add_argument('--BatchSize'                   , type = int, default = 128)
parser.add_argument('--Epoch'                       , type = int, default = 160)
parser.add_argument('--epochs_per_eval'             , type = int, default = 1)
parser.add_argument('--Pruning_Strategy'            , type = str, default = 'Filter_Similar')
parser.add_argument('--Pruning_Propotion'           , type = int, default = 10)
parser.add_argument('--Pruning_Propotion_2'         , type = int, default = 10)
parser.add_argument('--Pruning_Threshold'           , type = int, default = 0)
parser.add_argument('--Pruning_Threshold_2'         , type = int, default = 0)
parser.add_argument('--Pruning_Times'               , type = int, default = 6)
parser.add_argument('--Pruning_Propotion_Per_Time'  , type = int, default = 10)
parser.add_argument('--Iterative_Pruning'           , type = str, default = 'False')

FLAGs = parser.parse_args()
IS_HYPERPARAMETER_OPT = False

assert FLAGs.epochs_per_eval <= FLAGs.Epoch, "epochs_per_eval must small than Epoch"

print('\n\033[1;32mMODEL NAME\033[0m :\033[1;37m {MODEL_NAME}\033[0m' 
    .format(MODEL_NAME = FLAGs.Model_1st + '_' + FLAGs.Model_2nd))
#============#
#    Path    #
#============#
# For Loading Dataset
Dataset_Path = '../dataset/' + FLAGs.Dataset
if FLAGs.Dataset=='ade20k':
    Dataset_Path = Dataset_Path + '/ADEChallengeData2016'
elif FLAGs.Dataset=='ILSVRC2012':
    Dataset_Path = Dataset_Path + '/imagenet-data' 
Y_pre_Path   = FLAGs.Model_1st + '_Y_pre/' + FLAGs.Dataset

# For Saving Result Picture of Testing
train_target_path = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 
valid_target_path = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 
test_target_path  = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 

# For Saving Result in .npz file
train_Y_pre_path  = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'
valid_Y_pre_path  = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'
test_Y_pre_path   = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'

#============# 
#    Main    #
#============#
def main(argv):
    Global_Epoch = 0
    # ResNet
    if FLAGs.Model_1st == 'ResNet':
        ## ResNet-110
        if FLAGs.Model_2nd == '110_cifar10_0':
            Model_Path = 'Model/ResNet_Model/ResNet_110_cifar10_0_99_2018.02.08/'
            Model = '10.ckpt'
            #Global_Epoch = 0
        ## ResNet-56
        if FLAGs.Model_2nd == '56_cifar10_0':
            Model_Path = 'Model/ResNet_Model/ResNet_56_cifar10_0_99_2018.02.09_Filter_Similar60_59/'
            Model = '100.ckpt' 
            #Global_Epoch = 70
        if FLAGs.Model_2nd == '56_Binary':
            Model_Path = 'Model/ResNet_Model/ResNet_56_Binary_98_cifar10_2018.04.25_Filter_Angle10_40/' 
            Model = '166.ckpt'
            Global_Epoch = 1400
        ## ResNet-32
        if FLAGs.Model_2nd == '32_cifar10_0':
            Model_Path = 'Model/ResNet_Model/ResNet_32_cifar10_0_99_2018.02.09/'
            Model = '10.ckpt'
        ## ResNet-20
        if FLAGs.Model_2nd == '20_cifar10_2':
            Model_Path = 'Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06/'
            Model = '10.ckpt'
        if FLAGs.Model_2nd == '20_Binary':
            Model_Path = 'Model/ResNet_Model/ResNet_20_Binary_cifar10_2018.06.07/'
            Model = '122.ckpt'
            Global_Epoch = 0
        ## ResNet-50
        if FLAGs.Model_2nd == '50':
            Model_Path = 'Model/ResNet_Model/ResNet_50_74_cifar10_2018.03.27/'
            Model = '1.ckpt'
            #Global_Epoch = 43
    # DenseNet
    if FLAGs.Model_1st == 'DenseNet':
        ## DenseNet_40_12
        if FLAGs.Model_2nd == '40_12':
            Model_Path = 'Model/DenseNet_Model/DenseNet_40_12_99_cifar10_2018.03.06_Filter_Similar_K10_9/'
            Model = '300.ckpt'
            #Global_Epoch = 220 
    # MobileNet
    if FLAGs.Model_1st == 'MobileNet':
        ## MobileNet_100_100
        if FLAGs.Model_2nd == '100_100':
            Model_Path = 'Model/MobileNet_Model/MobileNet_100_100_65_cifar10_2018.04.07_Filter_Similar20_13/'
            Model = '43.ckpt'
            Global_Epoch = 43
    
    Model_Path_Ori = Model_Path
    Model_Ori = Model
    
    for pruning_iter in range(FLAGs.Pruning_Times):
        if FLAGS.Iterative_Pruning == 'False':
            Model_Path = Model_Path_Ori
            Model = Model_Ori
        while(1):
            if FLAGs.Dataset=='ILSVRC2012':
                epochs_per_eval = 1
            else:
                if Global_Epoch < FLAGs.Epoch * 0.9 and FLAGs.Epoch >= 10:
                    epochs_per_eval = 10
                else:
                    epochs_per_eval = 10#FLAGs.epochs_per_eval
                
            Model_Path, Model, Global_Epoch, is_skip = utils.run_pruning(              
                FLAGs               = FLAGs             ,
                Epoch               = epochs_per_eval   ,
                Global_Epoch        = Global_Epoch      ,
                Dataset_Path        = Dataset_Path      ,
                Y_pre_Path          = Y_pre_Path        ,
                pruning_model_path  = Model_Path        ,
                pruning_model       = Model             ,
                pruning_iter        = pruning_iter      )
            
            if is_skip:
                Global_Epoch = 0
                break
                
            # -- Testing --
            test_accuracy = utils.run_testing( 
                Hyperparameter        = None                    ,               
                FLAGs                 = FLAGs                   ,
                IS_HYPERPARAMETER_OPT = IS_HYPERPARAMETER_OPT   ,  
                Dataset_Path          = Dataset_Path            ,
                testing_model_path    = Model_Path              ,
                testing_model         = Model                   ,
                train_target_path     = train_target_path       ,
                valid_target_path     = valid_target_path       , 
                test_target_path      = test_target_path        ,
                train_Y_pre_path      = train_Y_pre_path        ,
                valid_Y_pre_path      = valid_Y_pre_path        ,
                test_Y_pre_path       = test_Y_pre_path         ,
                training_type         = 'prune'                 ,
                diversify_layers      = None                    ,
                is_find_best_model    = True                    )
                
            print("\033[0;33mGlobal Epoch{}\033[0m" .format(Global_Epoch))
            if Global_Epoch >= FLAGs.Epoch:
                Global_Epoch = 0
                break
        
if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)


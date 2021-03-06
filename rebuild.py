import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import argparse
import csv
import random

import utils_binary as utils
import Model_binary as Model

import sys
from sys import platform
import os
import time
#========================#
#    Global Parameter    #
#========================#
parser = argparse.ArgumentParser()

parser.add_argument('--Dataset'         , type = str, default = 'cifar10')
parser.add_argument('--Model_1st'       , type = str, default = 'ResNet')
parser.add_argument('--Model_2nd'       , type = str, default = '20_cifar10_2')
parser.add_argument('--BatchSize'       , type = int, default = 128)
parser.add_argument('--Epoch'           , type = int, default = 160)
parser.add_argument('--epochs_per_eval' , type = int, default = 1)
parser.add_argument('--mode'            , type = int, default = 1)
parser.add_argument('--Computation_ori' , type = int, default = 0)

FLAGs = parser.parse_args()

Model_Name = FLAGs.Model_1st + '_' + FLAGs.Model_2nd
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
    
#==============# 
#    Define    #
#==============#
def main(argv):
    # -- Training --
    ## ResNet-110
    if FLAGs.Model_2nd == '110_cifar10_0':
        Model_Path_base = 'Model/ResNet_Model/ResNet_110_cifar10_0_99_2018.02.08_Filter_Similar90_90/'
        Model_Paths     = []
    ## ResNet-56
    if FLAGs.Model_2nd == '56_cifar10_0':
        Model_Path_base = 'Model/ResNet_Model/ResNet_56_cifar10_0_99_2018.02.09_Filter_Similar60_59/'
        Model_Paths     = []
    ## ResNet-32
    if FLAGs.Model_2nd == '32_cifar10_0':
        Model_Path_base = None
        Model_Paths     = []
    ## ResNet-20
    if FLAGs.Model_2nd == '20_cifar10_2':
        Model_Path_base = 'Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06_Filter_Similar90_88/'
        Model_Paths     = ['Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06_Filter_Similar10_50/',
                           'Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06_Filter_Similar10_40/',
                           'Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06_Filter_Similar10_30/',
                           'Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06_Filter_Similar10_20/',
                           'Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06_Filter_Similar10_10/',
                           'Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06/']
    
    # Model_base
    Model_base = "100.ckpt"
    """
    with open(Model_Path_base + 'info.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if i == 1:
                Model_base = row[0].split('/')[-1]
    """
    # Models
    if FLAGs.mode == 0:
        Models = []
        for Model_Path in Model_Paths:
            with open(Model_Path + 'info.csv') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for i, row in enumerate(reader):
                    if i == 1:
                        Models.append(row[0].split('/')[-1])
        Models.append('100.ckpt')
    #--------------------------------#
    #    Decision of Middle Floor    #
    #--------------------------------#
    if FLAGs.mode == 0:
        None
    if FLAGs.mode == 1:
        pruned_weights_info = utils.load_obj(Model_Path_base, "pruned_info")[::-1]
        random.shuffle(pruned_weights_info)
        model_info = utils.load_obj(Model_Path_base, "model_info")
        computation_ori = FLAGs.Computation_ori
        selected_index = utils.middle_floor_decision(pruned_weights_info, model_info, Model_Path_base, computation_ori)
        Model_Path = Model_Path_base
        Model = Model_base
    
    if FLAGs.mode == 2:
        pruned_weights_info = utils.load_obj(Model_Path_base, "pruned_info")
        with open(Model_Path_base + 'selected_index.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for _, row in enumerate(reader):
                selected_index = row
        Model_Path = Model_Path_base
        Model = Model_base
        
    #------------------#
    #    Rebuilding    #
    #------------------#
    # rebuild_times
    if FLAGs.mode == 0:
        rebuild_times = len(Model_Paths)
    elif FLAGs.mode == 1:
        rebuild_times = len(selected_index)
    elif FLAGs.mode == 2:
        rebuild_times = len(selected_index)-1
        
    # Rebuild
    for iter in range(0, rebuild_times):
        Global_Epoch = 0
        # Middle Floor
        if FLAGs.mode == 0:
            Model_Path = Model_Paths[iter]
            Model = Models[iter]
            index_now = None
        elif FLAGs.mode == 1:
            index_now = []
            # Start Index
            if iter == 0:
                if not selected_index[iter] == len(pruned_weights_info)-1:
                    index_now.append(len(pruned_weights_info)-1)
            else:
                if not selected_index[iter-1]-1 == selected_index[iter]:
                    index_now.append(int(selected_index[iter-1]-1))
            # End Index
            index_now.append(int(selected_index[iter]))
            print("\033[1;32mReuilding Range\033[0m : {}" .format(index_now))
        elif FLAGs.mode == 2:
            index_now = []
            # Start Index
            index_now.append(int(selected_index[iter])-1)
            # End Index
            index_now.append(int(selected_index[iter+1]))
            print("\033[1;32mReuilding Range\033[0m : {}" .format(index_now))
            
        # Training
        while(1):
            if Global_Epoch < FLAGs.Epoch * 0.9 and FLAGs.Epoch >= 10:
                epochs_per_eval = 10
            else:
                epochs_per_eval = 10 #FLAGs.epochs_per_eval
                
            Model_Path, Model, Global_Epoch = utils.run_rebuilding(
                FLAGs                      = FLAGs              ,
                Epoch                      = epochs_per_eval    ,
                Global_Epoch               = Global_Epoch       ,
                Dataset_Path               = Dataset_Path       ,
                Y_pre_Path                 = Y_pre_Path         ,
                rebuilding_model_path_base = Model_Path_base    ,
                rebuilding_model_base      = Model_base         ,
                rebuilding_model_path      = Model_Path         ,
                rebuilding_model           = Model              ,
                index_now                  = index_now          )
        
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
                training_type         = 'rebuild'               ,
                diversify_layers      = None                    ,
                is_find_best_model    = True                    )
            
            print("\033[0;33mGlobal Epoch{}\033[0m" .format(Global_Epoch))
            if Global_Epoch >= FLAGs.Epoch:
                Global_Epoch = 0
                break
        
        Model_Path_base = Model_Path
        Model_base = Model
        
        # Save the selected index
        if FLAGs.mode == 1 or FLAGs.mode == 2:
            np.savetxt(Model_Path + 'selected_index.csv', np.array([selected_index[iter+1:]]), delimiter=",", fmt="%d")
        
if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)


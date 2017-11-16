import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import utils
import Model
import sys
import os
import time


#=======================#
#	Global Parameter	#
#=======================#
Dataset = 'CamVid' 
Model_first_name  = 'SegNet' #sys.argv[1] # e.g. : SegNet
Model_second_name = 'VGG_10' #sys.argv[2] # e.g. : VGG_16
Model_Name = Model_first_name + '_' + Model_second_name
Model_Call = getattr(Model, Model_Name)
print('\n\033[1;32;40mMODEL NAME\033[0m =\033[1;37;40m {MODEL_NAME}\033[0m' .format(MODEL_NAME=Model_Name))

IS_HYPERPARAMETER_OPT = False
IS_TRAINING 		  = True
IS_TESTING  		  = True

#==========#
#   Path   #
#==========#
# For Loading Dataset
Dataset_Path = '/home/2016/b22072117/ObjectSegmentation/codes/dataset/' + Dataset
if Dataset=='ade20k':
	Dataset_Path = Dataset_Path + '/ADEChallengeData2016'
Y_pre_Path   = '/home/2016/b22072117/ObjectSegmentation/codes/nets/' + Model_first_name + '_Y_pre/' + Dataset

# For Saving Result Picture of Testing
train_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/' + Dataset + '/' + Model_first_name + '/' 
valid_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/' + Dataset + '/' + Model_first_name + '/' 
test_target_path  = '/home/2016/b22072117/ObjectSegmentation/codes/result/' + Dataset + '/' + Model_first_name + '/' 

# For Saving Result in .npz file
train_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/' + Dataset + '/' + Model_first_name + '_Y_pre/'
valid_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/' + Dataset + '/' + Model_first_name + '_Y_pre/'
test_Y_pre_path   = '/home/2016/b22072117/ObjectSegmentation/codes/nets/' + Dataset + '/' + Model_first_name + '_Y_pre/'

# For Loading Trained Model
TESTING_WEIGHT_PATH   = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Model/' + 'SegNet_VGG_16_2017.10.31_09:15/'
TESTINGN_WEIGHT_MODEL = 'SegNet_VGG_16_200'
#============# 
#	Define   #
#============#


def main(argv=None):
	train_accuracy, valid_accuracy, test_accuracy = utils.run(
		Hyperparameter			= None,
		# Info
		Dataset 				= Dataset,
		Model_first_name 		= Model_first_name,
		Model_second_name 		= Model_second_name,
		Model_Name 				= Model_Name,
		Model_Call 				= Model_Call,
		IS_HYPERPARAMETER_OPT 	= IS_HYPERPARAMETER_OPT,
		IS_TRAINING 			= IS_TRAINING,
		IS_TESTING 				= IS_TESTING,
		# Path
		Dataset_Path			= Dataset_Path,
		Y_pre_Path				= Y_pre_Path,
		train_target_path		= train_target_path,
		valid_target_path		= valid_target_path,
		test_target_path		= test_target_path,
		train_Y_pre_path		= train_Y_pre_path,
		valid_Y_pre_path		= valid_Y_pre_path,
		test_Y_pre_path			= test_Y_pre_path,
		TESTING_WEIGHT_PATH     = TESTING_WEIGHT_PATH,
		TESTINGN_WEIGHT_MODEL   = TESTINGN_WEIGHT_MODEL
	)
	
if __name__ == "__main__":
	tf.app.run()


#General Imports
import sys
import numpy as np
import pandas as pd
import argparse
import copy
import random
import json
import pickle

#PyTorch
import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils

#Tensorflow
from absl import flags
import tensorflow as tf
from tensorflow.keras import layers

#Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

from .base_eval import BaseEval
from utils.privacy_attack import to_onehot, mia



class PrivacyAttack(BaseEval):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda)
        
    
    def get_metric_eval(self):
        
        '''
          Train Size: 2*sample_size
          Test Size: 2*sample_size

        '''
        final_res={}
        acc_train=[]
        acc_test=[]
        precision=[]
        recall=[]
        
        sample_size= self.args.mia_sample_size
        
        # Save the logits in .pkl file
        self.get_logits()
        
        # load the logits from .pkl file
        train_data= pickle.load(open(self.save_path + "_train.pkl", 'rb'))
        test_data= pickle.load(open(self.save_path + "_test.pkl", 'rb'))
        
        print(self.save_path)

        train_df = pd.DataFrame(train_data[0].cpu().detach().numpy())
        test_df = pd.DataFrame(test_data[0].cpu().detach().numpy())
        print('MIA Dataset', train_df.shape, test_df.shape, train_df)

        # ***********************Create data for probabilities members and non-members **********************
        X_dnn_train = train_df[:sample_size]

        Y_dnn_train = [1,0]
        Y_dnn_train = np.pad(Y_dnn_train, (len(X_dnn_train)-1,len(X_dnn_train)-1),'edge')
        Y_dnn_train = to_onehot(Y_dnn_train)

        X_dnn_train = X_dnn_train.append(test_df[:sample_size], ignore_index=True)
        X_dnn_train.rename(columns = {0:'value_0', 1:'value_1', 2:'value_2', 3:'value_3', 4:'value_4', 5:'value_5', 6:'value_6', 7:'value_7', 8:'value_8', 9:'value_9'}, inplace = True)

        ran_idx = np.random.permutation(X_dnn_train.index)
        X_dnn_train = X_dnn_train.reindex(ran_idx)
        Y_dnn_train = Y_dnn_train.reindex(ran_idx)

        X_dnn_test = train_df[-1-sample_size:-1]

        Y_dnn_test = [1,0]
        Y_dnn_test = np.pad(Y_dnn_test, (len(X_dnn_test)-1,len(X_dnn_test)-1),'edge')
        Y_dnn_test = to_onehot(Y_dnn_test)

        X_dnn_test = X_dnn_test.append(test_df[-1-sample_size:-1], ignore_index=True)
        X_dnn_test.rename(columns = {0:'value_0', 1:'value_1', 2:'value_2', 3:'value_3', 4:'value_4', 5:'value_5', 6:'value_6', 7:'value_7', 8:'value_8', 9:'value_9'}, inplace = True)    

        ran_idx = np.random.permutation(X_dnn_test.index)
        X_dnn_test = X_dnn_test.reindex(ran_idx)
        Y_dnn_test = Y_dnn_test.reindex(ran_idx)
        
        print('MIA Final Dataset: ', X_dnn_train.shape, X_dnn_test.shape)
        # Features for the attack dnn model
        attack_features = []
        
        for attack_idx in range(self.args.out_classes):
            attack_features.append( tf.feature_column.numeric_column(key="value_"+str(attack_idx)) )
        #print('Attack Features: ', attack_features)

        output_dnn = mia(X_dnn_train, Y_dnn_train, X_dnn_test, Y_dnn_test, attack_features, self.args.mia_batch_size, self.args.mia_dnn_steps, self.save_path)
        acc_train.append( 100*output_dnn['tr_attack']['accuracy'] )
        acc_test.append( 100*output_dnn['te_attack']['accuracy'] )
#         precision.append( output_dnn['precision'] )
#         recall.append( output_dnn['recall'] )

        acc_train= np.array(acc_train)
        acc_test= np.array(acc_test)
#         precision= np.array(precision)
#         recall= np.array(recall)

        final_res['tr_attack_mean']= np.round( np.mean(acc_train), 4 )
        final_res['te_attack_mean']= np.round( np.mean(acc_test), 4 )
#         final_res['precision_mean']= np.round( np.mean(precision) )
#         final_res['recall_mean']= np.round( np.mean(recall) )

        print('\nTrain Attack accuracy: ', final_res['tr_attack_mean'])
        print('\nTest Attack accuracy: ', final_res['te_attack_mean'])
#         print('\nPrecision: ', precision/args.total_seed )
#         print('\nRecall: ', recall/args.total_seed )

        self.metric_score['train_acc']= final_res['tr_attack_mean']
        self.metric_score['test_acc']= final_res['te_attack_mean']

        return         
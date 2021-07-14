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
from utils.attribute_attack import to_onehot, mia

class SpurCorrDataLoader(data_utils.Dataset):
    def __init__(self, dataloader):
        super(SpurCorrDataLoader, self).__init__()
        
        self.x= dataloader.data
        self.y= dataloader.labels
        self.d= dataloader.domains
        self.indices= dataloader.indices
        self.objects= dataloader.objects
        self.spur_corr= dataloader.spur
        

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        batch_x = self.x[index]
        batch_y = self.y[index]
        batch_d = self.d[index]
        batch_idx = self.indices[index]
        batch_obj= self.objects[index]
        batch_spur= self.spur_corr[index]
            
        return batch_x, batch_y, batch_d, batch_idx, batch_obj, batch_spur        

class AttributeAttack(BaseEval):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda)
        
    
    def get_spur_logits(self):

        #Train Environment Data
        train_data={}
        train_data['logits']=[]
        train_data['labels']=[]
        
        dataset= SpurCorrDataLoader(self.train_dataset['data_obj'])            
        dataset= data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs )        
        
        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e, spur_e) in enumerate(dataset):
            #Random Shuffling along the batch axis
            rand_indices= torch.randperm(x_e.size()[0])
            x_e= x_e[rand_indices]
            spur_e= spur_e[rand_indices]
            
            with torch.no_grad():
                x_e= x_e.to(self.cuda)                
                spur_e= spur_e.to(self.cuda)
                
                if self.args.mia_logit:
                    out= self.forward(x_e)
                else:
                    out= F.softmax(self.forward(x_e), dim=1)
                
                train_data['logits'].append(out)
                train_data['labels'].append(spur_e)
        
        train_data['logits']= torch.cat(train_data['logits'], dim=0)
        train_data['labels']= torch.cat(train_data['labels'], dim=0).cpu().numpy()

        #Test Environment Data
        test_data={}
        test_data['logits']=[]
        test_data['labels']=[]
        
        dataset= SpurCorrDataLoader(self.test_dataset['data_obj'])            
        dataset= data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs )        
        
        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e, spur_e) in enumerate(dataset):
            #Random Shuffling along the batch axis
            rand_indices= torch.randperm(x_e.size()[0])
            x_e= x_e[rand_indices]
            spur_e= spur_e[rand_indices]

            with torch.no_grad():
                x_e= x_e.to(self.cuda)                
                spur_e= spur_e.to(self.cuda)
                
                if self.args.mia_logit:
                    out= self.forward(x_e)
                else:
                    out= F.softmax(self.forward(x_e), dim=1)
                test_data['logits'].append(out)
                test_data['labels'].append(spur_e)
        
        test_data['logits']= torch.cat(test_data['logits'], dim=0)
        test_data['labels']= torch.cat(test_data['labels'], dim=0).cpu().numpy()
        
        print('Train Logits: ', train_data['logits'].shape, 'Train Labels: ', train_data['labels'].shape )
        print('Test Logits: ', test_data['logits'].shape, 'Test Labels: ', test_data['labels'].shape )
    
        return train_data, test_data
        
    
    def get_logits(self):

        #Train Environment Data
        train_data={}
        train_data['logits']=[]
        train_data['labels']=[]
        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(self.train_dataset['data_loader']):
            #Random Shuffling along the batch axis
            rand_indices= torch.randperm(x_e.size()[0])
            x_e= x_e[rand_indices]
            d_e= d_e[rand_indices]
            
            with torch.no_grad():
                x_e= x_e.to(self.cuda)                
                d_e= d_e.to(self.cuda)
                
                if self.args.mia_logit:
                    out= self.forward(x_e)
                else:
                    out= F.softmax(self.forward(x_e), dim=1)
                train_data['logits'].append(out)
                train_data['labels'].append(d_e)
        
        train_data['logits']= torch.cat(train_data['logits'], dim=0)
        train_data['labels']= torch.argmax( torch.cat(train_data['labels'], dim=0), dim=1 ).cpu().numpy()

        #Test Environment Data
        test_data={}
        test_data['logits']=[]
        test_data['labels']=[]
        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(self.test_dataset['data_loader']):
            #Random Shuffling along the batch axis
            rand_indices= torch.randperm(x_e.size()[0])
            x_e= x_e[rand_indices]
            d_e= d_e[rand_indices]

            with torch.no_grad():
                x_e= x_e.to(self.cuda)                
                d_e= d_e.to(self.cuda)
                
                if self.args.mia_logit:
                    out= self.forward(x_e)
                else:
                    out= F.softmax(self.forward(x_e), dim=1)
                test_data['logits'].append(out)
                test_data['labels'].append(d_e)
        
        test_data['logits']= torch.cat(test_data['logits'], dim=0)
        test_data['labels']= torch.argmax( torch.cat(test_data['labels'], dim=0), dim=1 ).cpu().numpy()
        
        print('Train Logits: ', train_data['logits'].shape, 'Train Labels: ', train_data['labels'].shape )
        print('Test Logits: ', test_data['logits'].shape, 'Test Labels: ', test_data['labels'].shape )
    
        return train_data, test_data

    
    
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
        if self.args.attribute_domain:
            train_data, test_data= self.get_logits()
        else:
            train_data, test_data= self.get_spur_logits()
        
        train_df = pd.DataFrame(train_data['logits'].cpu().detach().numpy())
        test_df = pd.DataFrame(test_data['logits'].cpu().detach().numpy())        
        train_labels= train_data['labels']
        test_labels= test_data['labels']
        print('MIA Dataset', train_df.shape, test_df.shape, train_labels.shape, test_labels.shape)

        # ***********************Create data for probabilities members and non-members **********************
        X_dnn_train = train_df
        Y_dnn_train = train_labels       
        print(np.unique(Y_dnn_train))
        Y_dnn_train = to_onehot(Y_dnn_train)

        X_dnn_train.rename(columns = {0:'value_0', 1:'value_1', 2:'value_2', 3:'value_3', 4:'value_4', 5:'value_5', 6:'value_6', 7:'value_7', 8:'value_8', 9:'value_9'}, inplace = True)

        ran_idx = np.random.permutation(X_dnn_train.index)
        X_dnn_train = X_dnn_train.reindex(ran_idx)
        Y_dnn_train = Y_dnn_train.reindex(ran_idx)


        X_dnn_test = test_df
        Y_dnn_test = test_labels       
        Y_dnn_test = to_onehot(Y_dnn_test)

        X_dnn_test.rename(columns = {0:'value_0', 1:'value_1', 2:'value_2', 3:'value_3', 4:'value_4', 5:'value_5', 6:'value_6', 7:'value_7', 8:'value_8', 9:'value_9'}, inplace = True)    

        ran_idx = np.random.permutation(X_dnn_test.index)
        X_dnn_test = X_dnn_test.reindex(ran_idx)
        Y_dnn_test = Y_dnn_test.reindex(ran_idx)
        
        print('MIA Final Dataset: ', X_dnn_train.shape, X_dnn_test.shape, Y_dnn_train.shape, Y_dnn_test.shape)
        # Features for the attack dnn model
        attack_features = []
        
        for attack_idx in range(self.args.out_classes):
            attack_features.append( tf.feature_column.numeric_column(key="value_"+str(attack_idx)) )
        #print('Attack Features: ', attack_features)
        
        num_classes= Y_dnn_train.shape[1]
        output_dnn = mia(X_dnn_train, Y_dnn_train, X_dnn_test, Y_dnn_test, attack_features, Y_dnn_train.shape[1], self.args.mia_batch_size, self.args.mia_dnn_steps, self.save_path)
        acc_train.append( output_dnn['tr_attack']['accuracy'] )
        acc_test.append( output_dnn['te_attack']['accuracy'] )
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

        self.metric_score['train_acc']= 100*final_res['tr_attack_mean']
        self.metric_score['test_acc']= 100*final_res['te_attack_mean']

        return         
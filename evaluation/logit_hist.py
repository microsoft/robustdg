#General Imports
import sys
import numpy as np
import pandas as pd
import argparse
import copy
import random
import json
import pickle
import matplotlib.pyplot as plt
import os

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



class LogitHist(BaseEval):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda)
        
    def get_loss(self):

        #Train Environment Logits
        final_loss=[]
        for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(self.train_dataset):
            #Random Shuffling along the batch axis
            x_e= x_e[ torch.randperm(x_e.size()[0]) ]
            y_e= torch.argmax(y_e, dim=1).to(self.cuda)

            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                out= self.forward(x_e)
                erm_loss= F.cross_entropy(out, y_e.long()).to(self.cuda)                
                final_loss.append(erm_loss)
        
        final_loss= torch.stack(final_loss)
        print('Train Loss: ', final_loss.shape, self.save_path)
        pickle.dump([final_loss], open( self.save_path + "_train_loss.pkl", 'wb'))

        #Test Environment Logits
        final_loss=[]
        for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(self.test_dataset):
            #Random Shuffling along the batch axis
            x_e= x_e[ torch.randperm(x_e.size()[0]) ]
            y_e= torch.argmax(y_e, dim=1).to(self.cuda)

            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                out= self.forward(x_e)
                erm_loss= F.cross_entropy(out, y_e.long()).to(self.cuda)                
                final_loss.append(erm_loss)
            
        final_loss= torch.stack(final_loss)
        print('Test Loss: ', final_loss.shape, self.save_path)
        pickle.dump([final_loss], open( self.save_path + "_test_loss.pkl", 'wb'))
    
        return
    
    def get_metric_eval(self):
        
        '''
          Train Size: 2*sample_size
          Test Size: 2*sample_size

        '''
        
        # Boolean Variable to check if loss for train/test examples are already saved
        compute_loss=1
        if os.path.exists(self.save_path + "_train_loss.pkl") and os.path.exists(self.save_path + "_test_loss.pkl"):
                compute_loss=0
        
        # Obtain Loss for each train/test example
        if compute_loss:
            self.get_loss()
        
        # load the logits from .pkl file
        train_data= pickle.load(open(self.save_path + "_train_loss.pkl", 'rb'))
        test_data= pickle.load(open(self.save_path + "_test_loss.pkl", 'rb'))
        
        train_df = pd.DataFrame(train_data[0].cpu().detach().numpy())
        test_df = pd.DataFrame(test_data[0].cpu().detach().numpy())
        print('MIA Dataset', train_df.shape, test_df.shape, train_df)

        meta_df= pd.concat([train_df, test_df], axis=1)
        meta_df.columns=['train', 'test']
        fig, ax = plt.subplots()
        meta_df.hist()
        print(self.args.logit_plot_path.split('/')[-1])
        ax.set_yscale('log')
        ax.set_title(' Dataset: ' + self.args.dataset_name + ' Method_Name ' + self.args.logit_plot_path.split('/')[-1])
        fig.savefig(self.args.logit_plot_path +  '_' +str(self.run) + '.jpg')

        return         
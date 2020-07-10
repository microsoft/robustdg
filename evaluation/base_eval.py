import sys
import numpy as np
import argparse
import copy
import random
import json

import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils


class BaseEval():
    def __init__(self, args, train_dataset, test_dataset, train_domains, total_domains, domain_size, training_list_size, save_path, cuda):
        self.args= args
        self.train_dataset= train_dataset
        self.test_dataset= test_dataset
        self.train_domains= train_domains
        self.total_domains= total_domains
        self.domain_size= domain_size 
        self.training_list_size= training_list_size
        self.save_path= save_path
        self.cuda= cuda
        self.phi= self.get_model()        
        self.metric_score={}                
    
    def get_model(self):
        
        if self.args.model_name == 'lenet':
            from models.LeNet import LeNet5
            phi= LeNet5().to(self.cuda)
        if self.args.model_name == 'alexnet':
            from models.AlexNet import alexnet
            phi= alexnet(self.args.out_classes, self.args.pre_trained, self.args.method_name).to(self.cuda)
        if self.args.model_name == 'resnet18':
            from models.ResNet import get_resnet
            phi= get_resnet('resnet18', self.args.out_classes, self.args.method_name, self.args.img_c, self.args.pre_trained).to(self.cuda)
        
        print('Model Architecture: ', self.args.model_name)
        return phi
    
    def load_model(self):
        
        self.phi.load_state_dict( torch.load(self.save_path) )
        self.phi.eval()        
    
    def get_metric_eval(self):
        
        #Test Env Code
        test_acc= 0.0
        test_size=0

        for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(self.test_dataset):
            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                d_e = torch.argmax(d_e, dim=1).numpy()       

                #Forward Pass
                out= self.phi(x_e)                
                loss_e= torch.mean(F.cross_entropy(out, y_e.long()).to(self.cuda))
                
                test_acc+= torch.sum( torch.argmax(out, dim=1) == y_e ).item()
                test_size+= y_e.shape[0]

        print(' Accuracy: ', 100*test_acc/test_size ) 
        self.metric_score['Test Accuracy']= 100*test_acc/test_size  
        return 

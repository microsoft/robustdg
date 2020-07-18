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

#Avertorch
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow
from advertorch.attacks import LinfPGDAttack

from .base_eval import BaseEval


class AdvAttack(BaseEval):
    
    def __init__(self, args, train_dataset, test_dataset, train_domains, total_domains, domain_size, training_list_size, base_res_dir, run, cuda):
        
        super().__init__(args, train_dataset, test_dataset, train_domains, total_domains, domain_size, training_list_size, base_res_dir, run, cuda)
        
        
    def get_metric_eval(self):        

        utr_score=[]
        tr_score=[]
        for i in range(1):
            
            ##TODO: Customise input parameters to methods like LinfPGDAttack
            adversary = LinfPGDAttack(
                self.phi, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.10,
                nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=False)    

            adv_untargeted = adversary.perturb(x_e, y_e)

            target = torch.ones_like(y_e)*3
            adversary.targeted = True
            adv_targeted = adversary.perturb(x_e, target)

            pred_cln = predict_from_logits(self.phi(x_e))
            pred_untargeted_adv= predict_from_logits(self.phi(adv_untargeted))
            pred_targeted_adv= predict_from_logits(self.phi(adv_targeted)) 
            utr_score.append( torch.sum( pred_cln != pred_untargeted_adv) )
            tr_score.append( torch.sum(pred_cln!= pred_targeted_adv) )

            batch_size=5
            plt.figure(figsize=(10, 8))
            for ii in range(batch_size):
                plt.subplot(3, batch_size, ii + 1)
                _imshow(x_e[ii])
                plt.title("clean \n pred: {}".format(pred_cln[ii]))
                plt.subplot(3, batch_size, ii + 1 + batch_size)
                _imshow(adv_untargeted[ii])
                plt.title("untargeted \n adv \n pred: {}".format(
                    pred_untargeted_adv[ii]))
                plt.subplot(3, batch_size, ii + 1 + batch_size * 2)
                _imshow(adv_targeted[ii])
                plt.title("targeted to 3 \n adv \n pred: {}".format(
                    pred_targeted_adv[ii]))

            plt.tight_layout()
            plt.savefig( self.save_path + '.png' )


        utr_score= np.array(utr_score)
        tr_score= np.array(tr_score)
        print('MisClassifcation on Untargetted Attack ', np.mean(utr_score), np.std(utr_score)  ) 
        print('MisClassifcation on Targeted Atttack', np.mean(tr_score), np.std(tr_score) )
    
        self.metric_score['Untargetted Method']= np.mean( utr_score ) 
        self.metric_score['Targetted Method']= np.mean( tr_score )
        
        return


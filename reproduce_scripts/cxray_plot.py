import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_base_dir(test_domain, dataset, metric):
    
    if metric == 'mia':
        res_dir= 'results/'+str(dataset)+'/privacy/' + str(test_domain) + '/'

    elif metric == 'privacy_entropy':
        res_dir= 'results/'+str(dataset)+'/privacy_entropy/' + str(test_domain) + '/'

    elif metric == 'privacy_loss_attack':
        res_dir= 'results/'+str(dataset)+'/privacy_loss/' + str(test_domain) + '/' 

    elif metric == 'attribute_attack':
        res_dir= 'results/'+str(dataset)+'/attribute_attack_' + data_case + '/'  + str(test_domain) + '/'  

    elif metric  == 'acc:train':
        res_dir= 'results/' + str(dataset) + '/acc_' + 'train' + '/' + str(test_domain) + '/'

    elif metric  == 'acc:test':
        res_dir= 'results/' + str(dataset) + '/acc_' + 'test' + '/' + str(test_domain) + '/'
        
    elif metric  == 'match_score:train':
        res_dir= 'results/' + str(dataset) + '/match_score_' + 'train' + '/' + str(test_domain) + '/'    

    elif metric  == 'match_score:test':
        res_dir= 'results/' + str(dataset) + '/match_score_' + 'test' + '/' + str(test_domain) + '/'    
        
    return res_dir

x=['ERM', 'Rand', 'MatchDG', 'CSD', 'IRM', 'Hybrid']
methods=['erm', 'rand', 'matchdg_erm', 'csd', 'irm', 'hybrid']
dataset= 'chestxray'

matplotlib.rcParams.update({'errorbar.capsize': 2})
fig, ax = plt.subplots(1, 3, figsize=(33, 8))
fontsize=35
fontsize_lgd= fontsize/1.2

# kaggle, nih, chex
for idx in range(3):
        
    marker_list = ['o', '^', '*']
    legend_list = ['RSNA', 'NIH', 'Chex']
    legend_count = 0
    for test_domain in ['kaggle', 'nih', 'chex']:

        metrics= ['acc:train', 'acc:test', 'privacy_loss_attack']

        acc_train=[]
        acc_train_err=[]

        acc_test=[]
        acc_test_err=[]

        mia=[]
        mia_err=[]

        entropy=[]
        entropy_err=[]

        loss=[]
        loss_err=[]

        rank_train=[]
        rank_train_err=[]

        rank_test=[]
        rank_test_err=[]

        for metric in metrics:
            for method in methods:

                res_dir= get_base_dir(test_domain, dataset, metric)

                f= open(res_dir+method+'.txt')
                data= f.readlines()
        #         print(data[-3].replace('\n', '').split(':')[-1].split(' '))
                mean= float(data[-3].replace('\n', '').split(':')[-1].split(' ')[-2])
                sd= float(data[-3].replace('\n', '').split(':')[-1].split(' ')[-1])

                if metric == 'acc:train':
                    acc_train.append(mean)
                    acc_train_err.append(sd)
                elif metric == 'acc:test':
                    acc_test.append(mean)
                    acc_test_err.append(sd)
                elif metric == 'mia':
                    mia.append(mean)
                    mia_err.append(sd)
                elif metric == 'privacy_entropy':
                    entropy.append(mean)
                    entropy_err.append(sd)
                elif metric == 'privacy_loss_attack':
                    loss.append(mean)
                    loss_err.append(sd)
                elif metric == 'match_score:train':
                    rank_train.append(mean)
                    rank_train_err.append(sd)
                elif metric == 'match_score:test':
                    rank_test.append(mean)
                    rank_test_err.append(sd)

        ax[idx].tick_params(labelsize=fontsize)
        ax[idx].set_xticklabels(x, rotation=25)
    
        if idx == 0:
            ax[idx].errorbar(x, acc_test, yerr=acc_test_err, label= legend_list[legend_count], marker= marker_list[legend_count], markersize= fontsize_lgd, linewidth=4, fmt='o--')
            ax[idx].set_ylabel('OOD Accuracy', fontsize=fontsize)

        if idx == 1:
            ax[idx].errorbar(x, loss, yerr=loss_err, label= legend_list[legend_count], marker= marker_list[legend_count], markersize= fontsize_lgd, linewidth=4, fmt='o--')
            ax[idx].set_ylabel('MI Attack Accuracy', fontsize=fontsize)

#         if idx == 2:
#     #         ax.errorbar(x, rank_train, yerr=rank_train_err, label='Train', fmt='o--', color='brown')
#             ax[idx].errorbar(x, rank_test, yerr=rank_test_err, label='Test', fmt='o--', color='green')
#     #         ax.set_xlabel('Models', fontsize=fontsize)
#             ax[idx].set_ylabel('Mean Rank of Perfect Match', fontsize=fontsize)
#             ax[idx].legend(fontsize=fontsize_lgd)

        if idx == 2:
            ax[idx].errorbar(x, np.array(acc_train) - np.array(acc_test), yerr=acc_train_err, label= legend_list[legend_count], marker= marker_list[legend_count], markersize= fontsize_lgd, linewidth=4, fmt='o--')
            ax[idx].set_ylabel('Train-Test Accuracy Gap ', fontsize=fontsize)
    
    
        legend_count+= 1 
    
save_dir= 'results/' + dataset+ '/plots/'    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)        

lines, labels = fig.axes[-1].get_legend_handles_labels()    
lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=fontsize, ncol=3)
    
plt.tight_layout()
plt.savefig(save_dir + 'privacy_' + str(dataset) + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_base_dir(train_case, test_case, dataset, metric):
    
    if metric == 'mia':
        res_dir= 'results/'+str(dataset)+'/privacy/'

    elif metric == 'privacy_entropy':
        res_dir= 'results/'+str(dataset)+'/privacy_entropy/'

    elif metric == 'privacy_loss_attack':
        res_dir= 'results/'+str(dataset)+'/privacy_loss/'

    elif metric == 'attribute_attack':
        res_dir= 'results/'+str(dataset)+'/attribute_attack_' + data_case + '/'  

    elif metric  == 'acc:train':
        res_dir= 'results/' + str(dataset) + '/acc_' + 'train' + '/'

    elif metric  == 'acc:test':
        res_dir= 'results/' + str(dataset) + '/acc_' + 'test' + '/'
        
    elif metric  == 'match_score:train':
        res_dir= 'results/' + str(dataset) + '/match_score_' + 'train' + '/'

    elif metric  == 'match_score:test':
        res_dir= 'results/' + str(dataset) + '/match_score_' + 'test' + '/'

    elif metric  == 'feat_eval:train':
        res_dir= 'results/' + str(dataset) + '/feat_eval_' + 'train' + '/'

    elif metric  == 'feat_eval:test':
        res_dir= 'results/' + str(dataset) + '/feat_eval_' + 'test' + '/'
        
    #Train Domains 30, 45 case
    if train_case == 'train_abl_2':
        res_dir= res_dir[:-1] +'_30_45/'

    #Train Domains 30, 45, 60 case
    if train_case == 'train_abl_3':
        res_dir= res_dir[:-1] +'_30_45_60/'            

    #Test on 30, 45 angles instead of the standard 0, 90
    if test_case  == 'test_common':
        res_dir+= 'test_common_domains/'
        
    return res_dir

    
#rot_mnist, fashion_mnist, rot_mnist_spur
dataset=sys.argv[1]

# train_all, train_abl_3, train_abl_2
# train_case= sys.argv[2]

# test_diff, test_common
test_case=['test_diff']

matplotlib.rcParams.update({'errorbar.capsize': 2})
fig, ax = plt.subplots(1, 3, figsize=(33, 8))
fontsize=35
fontsize_lgd= fontsize/1.2
x=['ERM', 'Rand', 'MatchDG', 'CSD', 'IRM', 'Perf']
methods=['erm', 'rand', 'matchdg', 'csd', 'irm', 'perf']
# metrics= ['acc:train', 'acc:test', 'mia', 'privacy_entropy', 'privacy_loss_attack', 'match_score:train', 'match_score:test', 'feat_eval:train', 'feat_eval:test']

metrics= ['acc:train', 'acc:test', 'privacy_loss_attack', 'match_score:test']

for idx in range(3):
    
    marker_list = ['o', '^', '*']
    legend_count = 0
    for train_case in ['train_all', 'train_abl_3', 'train_abl_2']:

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

        feat_eval_train=[]
        feat_eval_train_err=[]

        feat_eval_test=[]
        feat_eval_test_err=[]

        for metric in metrics:
            for method in methods:

                res_dir= get_base_dir(train_case, test_case, dataset, metric)

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
                elif metric == 'feat_eval:train':
                    feat_eval_train.append(mean)
                    feat_eval_train_err.append(sd)
                elif metric == 'feat_eval:test':
                    feat_eval_test.append(mean)
                    feat_eval_test_err.append(sd)

        if train_case == 'train_all':
            legend_label= '5 domains'
        elif train_case == 'train_abl_2':
            legend_label= '3 domains'
        elif train_case == 'train_abl_3':
            legend_label= '2 domains'
        
        ax[idx].tick_params(labelsize=fontsize)
        ax[idx].set_xticklabels(x, rotation=25) 
    
        if idx == 0:
            ax[idx].errorbar(x, acc_test, yerr=acc_test_err, label=legend_label, marker= marker_list[legend_count], markersize= fontsize_lgd, linewidth=4, fmt='o--')
            ax[idx].set_ylabel('OOD Accuracy', fontsize=fontsize)            
            
            
        if idx == 1:
            ax[idx].errorbar(x, loss, yerr=loss_err, label=legend_label, marker= marker_list[legend_count], markersize= fontsize_lgd, linewidth=4, fmt='o--')
            ax[idx].set_ylabel('MI Attack Accuracy', fontsize=fontsize)

        if idx == 2:
            ax[idx].errorbar(x, rank_test, yerr=rank_test_err, label=legend_label, marker= marker_list[legend_count], markersize= fontsize_lgd, linewidth=4, fmt='o--')
            ax[idx].set_ylabel('Mean Rank', fontsize=fontsize)

        if idx == 3:
            ax[idx].errorbar(x, np.array(acc_train) - np.array(acc_test), yerr=acc_train_err, marker= 's', markersize= fontsize_lgd, linewidth=4, fmt='o--')
            ax[idx].set_ylabel('Generalization Gap', fontsize=fontsize)
        
        legend_count+= 1 
    
save_dir= 'results/' + dataset+ '/plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

lines, labels = fig.axes[-1].get_legend_handles_labels()    
lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=fontsize, ncol=3)

plt.tight_layout()
plt.savefig(save_dir + 'privacy_' + str(dataset) + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

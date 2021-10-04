import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

slab_noise= float(sys.argv[1])
base_dir= 'results/slab/slab_noise_' + str(slab_noise) + '/'
methods=['erm', 'irm', 'csd', 'rand', 'perf', 'mask_linear']

# x=['ERM', 'IRM', 'CSD', 'Rand', 'MatchDG', 'Perf', 'Mask']
x=['ERM', 'IRM', 'CSD', 'Rand', 'Perf', 'Oracle']
matplotlib.rcParams.update({'errorbar.capsize': 2})
fig, ax = plt.subplots(1, 2, figsize=(24, 8))
fontsize=40
fontsize_lgd= fontsize/1.2
marker_list = ['o', '^', '*']
count= 0

for test_domain in [0.2, 0.9]:
        
    acc=[]
    acc_err=[]
    train_acc =[]
    train_acc_err =[]
    auc=[]
    auc_err=[]
    s_auc=[]
    s_auc_err=[]
    sc_auc=[]
    sc_auc_err=[]

    for method in methods:

        f= open(base_dir + method + '-auc-' + str(test_domain) + '.txt')
        data= f.readlines()
        acc.append( float( data[-4].replace('\n', '').split(' ')[-2] ))
        acc_err.append( float( data[-4].replace('\n', '').split(' ')[-1] ))
        auc.append( float( data[-3].replace('\n', '').split(' ')[-2] ))
        auc_err.append( float( data[-3].replace('\n', '').split(' ')[-1] ))
        s_auc.append( float( data[-2].replace('\n', '').split(' ')[-2] ))
        s_auc_err.append( float( data[-2].replace('\n', '').split(' ')[-1] ))
        sc_auc.append( float(data[-1].replace('\n', '').split(' ')[-2] ))
        sc_auc_err.append( float( data[-1].replace('\n', '').split(' ')[-1] ) )

        f= open(base_dir + method + '-train-auc-' + str(test_domain) + '.txt')
        data= f.readlines()
        train_acc.append( float( data[-4].replace('\n', '').split(' ')[-2] ))
        train_acc_err.append( float( data[-4].replace('\n', '').split(' ')[-1] ))
        
        
    #Privacy Metrics
    mia=[]
    mia_err=[]
    entropy=[]
    entropy_err=[]
    loss=[]
    loss_err=[]
    attribute=[]
    attribute_err=[]

        
    # eval_metrics= ['mi', 'entropy', 'loss', 'attribute']
    eval_metrics= ['loss']
    for metric in eval_metrics:
        for method in methods:
            f= open(base_dir+method+'-'+metric+ '-' + str(test_domain)+'.txt')
            data= f.readlines()

            mean= float(data[-3].replace('\n', '').split(' ')[-2])
            sd= float(data[-3].replace('\n', '').split(' ')[-1])

            if metric == 'mi':
                mia.append(mean)
                mia_err.append(sd)
            elif metric == 'entropy':
                entropy.append(mean)
                entropy_err.append(sd)
            elif metric == 'loss':
                loss.append(mean)
                loss_err.append(sd)
            elif metric == 'attribute':
                attribute.append(mean)
                attribute_err.append(sd)
    
    ax[count].tick_params(labelsize=fontsize)
    ax[count].set_xticklabels(x, rotation=25)
    
    ax[count].errorbar(x, acc, yerr=acc_err, marker= marker_list[0], markersize= fontsize_lgd, linewidth=4, fmt='o--', label='OOD Acc')
    ax[count].errorbar(x, s_auc, yerr=s_auc_err, marker= marker_list[1], markersize= fontsize_lgd, linewidth=4, fmt='o--', label='Stable Features (Linear-RAUC)')
    ax[count].errorbar(x, loss, yerr=loss_err, marker= marker_list[2], markersize= fontsize_lgd, linewidth=4, label='MI Attack Acc', fmt='o--')
    
#     gen_gap= np.array(train_acc) - np.array(acc)
#     ax[count].errorbar(x, gen_gap, yerr=0*gen_gap, marker= 's', markersize= fontsize_lgd, linewidth=4, fmt='o--', label='Generalization Gap')
    
    ax[count].set_ylabel('Metric Score', fontsize=fontsize)
    ax[count].set_title('Test Domain: ' + str(test_domain), fontsize=fontsize)
    
    count+=1
    
lines, labels = fig.axes[-1].get_legend_handles_labels()    
lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=fontsize, ncol=4)

save_dir= 'results/slab/plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.tight_layout()
plt.savefig( save_dir + 'privacy_slab_' + str(slab_noise) + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight',  dpi=600)


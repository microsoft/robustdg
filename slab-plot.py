import matplotlib
import matplotlib.pyplot as plt
import sys

test_domain= float(sys.argv[1])
slab_noise= float(sys.argv[2])
base_dir= 'slab_res/slab_noise_' + str(slab_noise) + '/'
methods=['erm', 'irm', 'csd', 'rand', 'matchdg', 'perf', 'mask_linear']

acc=[]
acc_err=[]
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
eval_metrics= ['mi', 'entropy', 'loss']
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

x=['ERM', 'IRM', 'CSD', 'Rand', 'MatchDG', 'Perf', 'Mask']
for idx in range(0, 2):
    
    matplotlib.rcParams.update({'errorbar.capsize': 2})
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fontsize=20
    ax.tick_params(labelsize=fontsize)
    ax.set_xticklabels(x, rotation=25)
    
    if idx == 0:
#         ax.errorbar(x, train_acc, yerr=train_acc_err, fmt='o--', label='Train-Acc')
        ax.errorbar(x, acc, yerr=acc_err, fmt='o--', label='Acc')
        ax.errorbar(x, auc, yerr=auc_err, fmt='o--', label='AUC')
        ax.errorbar(x, s_auc, yerr=s_auc_err, fmt='o--', label='Linear-RAUC')
        ax.errorbar(x, sc_auc, yerr=sc_auc_err, fmt='o--', label='Slab-RAUC')
#         ax.set_xlabel('Models', fontsize=fontsize)
        ax.set_ylabel('ML Model Acc/ AUC', fontsize=fontsize)
        ax.set_title('OOD Evaluation', fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        
    if idx == 1:
        ax.errorbar(x, mia, yerr=mia_err, label='Classifier Attack', color='blue', fmt='o--')        
        ax.errorbar(x, entropy, yerr=entropy_err, label='Entropy Attack', color='red', fmt='o--')
        ax.errorbar(x, loss, yerr=loss_err, label='Loss Attack', color='orange', fmt='o--')
        ax.set_ylabel('Attack Model Accuracy', fontsize=fontsize)
        ax.set_title('Privacy Attack Evaluation', fontsize=fontsize)
        ax.legend(fontsize=fontsize)

    if idx == 2:
        ax.errorbar(x, attribute, yerr=attribute_err, label='Classifier Attack', color='blue', fmt='o--')        
        ax.set_ylabel('Attack Model Accuracy', fontsize=fontsize)
        ax.set_title('Attribute Attack Evaluation', fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        
    
    plt.savefig('results/privacy_slab_' + str(test_domain) + str(idx) + '.pdf', dpi=600)
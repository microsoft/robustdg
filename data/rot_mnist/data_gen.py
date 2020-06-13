import numpy as np
import sys

if sys.argv[1] == 'resnet':

    # Generate 10 random subsets of size 2,000 each for Rotated MNIST 
    data_size=60000
    subset_size=2000
    val_size= 100
    data_dir='rot_mnist_indices/'
    total_subset=10

    for idx in range(total_subset):
        # Train, Test indices
        res=np.random.choice(data_size, subset_size)
        np.save( data_dir + 'supervised_inds_' + str(idx) +'.npy', res)

        # Val indices
        res=np.random.choice(data_size, val_size)
        np.save( data_dir + 'val/' + 'supervised_inds_' + str(idx) +'.npy', res)


    # Genetate 10 random subsets of size 10,000 each for Fashion MNIST
    data_size=60000
    subset_size=10000
    val_size= 500
    data_dir='fashion_mnist_indices/'
    total_subset=10

    for idx in range(total_subset):
        # Train, Test indices
        res=np.random.choice(data_size, subset_size)
        np.save( data_dir + 'supervised_inds_' + str(idx) +'.npy', res)

        # Val indices
        res=np.random.choice(data_size, val_size)
        np.save( data_dir + 'val/' + 'supervised_inds_' + str(idx) +'.npy', res)
        
elif sys.argv[1] == 'lenet':
    # Generate 10 random subsets of size 1,000 each for Rotated MNIST 
    data_size=60000
    subset_size=1000
    val_size= 100
    data_dir='rot_mnist_lenet_indices/'
    total_subset=10

    for idx in range(total_subset):
        # Train, Test indices
        res=np.random.choice(data_size, subset_size)
        np.save( data_dir + 'supervised_inds_' + str(idx) +'.npy', res)

        # Val indices
        res=np.random.choice(data_size, val_size)
        np.save( data_dir + 'val/' + 'supervised_inds_' + str(idx) +'.npy', res)

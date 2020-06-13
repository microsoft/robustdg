The commands below would train and evaluate models corresponding to the Table 1, Table 2 and Table 3 in the paper

The commands below would generate results for dataset Rot MNIST and source domains [15, 30, 45, 60, 75]

* Change the input to --dataset from rot_mnist to fashion_mnist in the commands below to get results on Fashion MNIST

* Change the input to --domain_abl from 0 to 2 to get results on source domains [30, 45]

* Change the input to --domain_abl from 0 to 3 to get results on source domains [30, 45, 60]




Table 1:

ERM: 

python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.01 --penalty_ws 0.0

ERM_RandomMatch:

python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.01 --penalty_ws 0.1

ERM_PerfectMatch:

python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 1.0 --penalty_ws 0.1

MatchDG:

python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 2 --match_case 0.01




Table 2:

ERM: 

python3 eval.py --dataset rot_mnist --domain_abl 0 --test_metric other --match_dg 0 --match_case 0.01 --penalty_ws 0.0

MatchDG (Default):

python3 eval.py --dataset rot_mnist --domain_abl 0 --test_metric other --match_dg 1 --match_case 0.01

MatchDG (PerfMatch):

python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 2 --match_case 1.0

python3 eval.py --dataset rot_mnist --domain_abl 0 --test_metric other --match_dg 1 --match_case 1.0




Table 3:

Approx 25:

python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.25 --penalty_ws 0.1

Approx 50:

python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.50 --penalty_ws 0.1

Approx 75:

python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.75 --penalty_ws 0.1

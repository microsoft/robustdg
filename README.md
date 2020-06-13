
# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# MNIST and Fashion MNIST

The commands below would train and evaluate models corresponding to the Table 1, Table 2 and Table 3 in the paper

The commands below would generate results for dataset Rot MNIST and source domains [15, 30, 45, 60, 75]

* Change the input to --dataset from rot_mnist to fashion_mnist in the commands below to get results on Fashion MNIST

* Change the input to --domain_abl from 0 to 2 to get results on source domains [30, 45]

* Change the input to --domain_abl from 0 to 3 to get results on source domains [30, 45, 60]


## Prepare Data

Move to the directory: data/rot_mnist
Execute: python3 data_gen.py resnet

## Table 1

* ERM: 

  - python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.01 --penalty_ws 0.0

* ERM_RandomMatch:

  - python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.01 --penalty_ws 0.1

* ERM_PerfectMatch:

  - python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 1.0 --penalty_ws 0.1

* MatchDG:

  - python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 2 --match_case 0.01


## Table 2

* ERM: 

  - python3 eval.py --dataset rot_mnist --domain_abl 0 --test_metric other --match_dg 0 --match_case 0.01 --penalty_ws 0.0

* MatchDG (Default):

  - python3 eval.py --dataset rot_mnist --domain_abl 0 --test_metric other --match_dg 1 --match_case 0.01

* MatchDG (PerfMatch):

  - python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 2 --match_case 1.0

  - python3 eval.py --dataset rot_mnist --domain_abl 0 --test_metric other --match_dg 1 --match_case 1.0


## Table 3

* Approx 25:

  - python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.25 --penalty_ws 0.1

* Approx 50:

  - python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.50 --penalty_ws 0.1

* Approx 75:

  - python3 train.py --dataset rot_mnist --domain_abl 0 --match_dg 0 --match_case 0.75 --penalty_ws 0.1

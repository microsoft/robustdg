# Toolkit for Building Robust ML models that generalize to unseen domains (RobustDG)

For machine learning models to be reliable, they need to generalize to data
beyond the train distribution. In addition, ML models should be robust to
privacy attacks like membership inference and domain knowledge-based attacks like adversarial attacks.

To advance research in building robust and generalizable models, we are
releasing a toolkit for building and evaluating ML models, *RobustDG*. RobustDG contains implementations of a few domain
generalization algorithms and includes variety of evaluation benchmarks based
on out-of-distribution accuracy, robustness to privacy and adversarial attacks. 

It is easily extendable. Add your own DG algorithms and evaluate them on the
different benchmarks.

# Demo

# Installation
To use the command-line interface of RobustDG, clone this repo and add the
folder to your system's PATH (or alternatively, run the commands from the
RobustDG root directory)

```
train ---<more params>
```

# Demo

A quick introduction on how to use our repository can be accessed here in the [notebook](https://github.com/microsoft/robustdg/blob/master/docs/notebook/robustdg_getting_started.ipynb)

# Reproducing results from the paper

## MNIST and Fashion MNIST

The commands below would train and evaluate models corresponding to the Table 1, Table 2 and Table 3 in the paper

The commands below would generate results for dataset Rot MNIST and source domains [15, 30, 45, 60, 75]

* Change the input to --dataset from rot_mnist to fashion_mnist in the commands below to get results on Fashion MNIST

* Change the input to --domain_abl from 0 to 2 to get results on source domains [30, 45]

* Change the input to --domain_abl from 0 to 3 to get results on source domains [30, 45, 60]


## Prepare Data

  - Move to the directory: data/rot_mnist
  - Execute: python3 data_gen.py resnet

## Table 1

* ERM: 

  - python3 train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0

* ERM_RandomMatch:

  - python3 train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1

* ERM_PerfectMatch:

  - python3 train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1

* MatchDG:

  - python3 train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01
  - python3 train.py --dataset rot_mnist --method_name matchdg_erm --penalty_ws 0.1


## Table 2

* ERM: 

  - python3 eval.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --test_metric match_score 

* MatchDG (Default):

  - python3 eval.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --test_metric match_score

* MatchDG (PerfMatch):

  - python3 eval.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --test_metric match_score


## Table 3

* Approx 25:

  - python3 train.py --dataset rot_mnist --method_name erm_match --match_case 0.25 --penalty_ws 0.1

* Approx 50:

  - python3 train.py --dataset rot_mnist --method_name erm_match --match_case 0.50 --penalty_ws 0.1

* Approx 75:

  - python3 train.py --dataset rot_mnist --method_name erm_match --match_case 0.75 --penalty_ws 0.1


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

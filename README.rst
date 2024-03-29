Toolkit for Building Robust ML models that generalize to unseen domains (RobustDG)
==================================================================================
`Divyat Mahajan <https://divyat09.github.io/>`_, 
`Shruti Tople <https://www.microsoft.com/en-us/research/people/shtople/>`_, 
`Amit Sharma <http://www.amitsharma.in>`_

`Privacy & Causal Learning (ICML 2020) <https://arxiv.org/abs/1909.12732>`_ | `MatchDG: Causal View of DG (ICML 2021) <http://proceedings.mlr.press/v139/mahajan21b.html>`_ | `Privacy & DG Connection paper <https://arxiv.org/abs/2110.03369>`_

For machine learning models to be reliable, they need to generalize to data
beyond the train distribution. In addition, ML models should be robust to
privacy attacks like membership inference and domain knowledge-based attacks like adversarial attacks.

To advance research in building robust and generalizable models, we are
releasing a toolkit for building and evaluating ML models, *RobustDG*. RobustDG contains implementations of domain
generalization algorithms and includes evaluation benchmarks based
on out-of-distribution accuracy and robustness to membership privacy attacks. We will be adding evaluation for adversarial attacks and more privacy attacks soon. 

It is easily extendable. Add your own DG algorithms and evaluate them on different benchmarks.


Installation
------------
To use the command-line interface of RobustDG, clone this repo and add the folder to your system's PATH (or alternatively, run the commands from the RobustDG root directory). 

**Load dataset**

Let's first load the rotatedMNIST dataset in a suitable format for the resnet18 architecture.

.. code:: shell

    python data/data_gen_mnist.py --dataset rot_mnist --model resnet18 --img_h 224 --img_w 224 --subset_size 2000

**Train and evaluate ML model**

The following commands would train and evalute the MatchDG method on the Rotated MNIST dataset.

.. code:: shell


    python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.0 --match_flag 1 --epochs 50 --batch_size 64 --pos_metric cos --match_func_aug_case 1
    
    python train.py --dataset rot_mnist --method_name matchdg_erm --penalty_ws 0.1 --match_case -1 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18 --epochs 25
    
    python test.py --dataset rot_mnist --method_name matchdg_erm --penalty_ws 0.1 --match_case -1 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18 --epochs 25 --test_metric acc
    
    python test.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.0 --match_flag 1 --pos_metric cos --test_metric match_score    


Demo
----

A quick introduction on how to use our repository can be accessed here in the `Getting Started notebook <https://github.com/microsoft/robustdg/blob/master/docs/notebooks/robustdg_getting_started.ipynb>`_.

If you are interested in reproducing results from the MatchDG paper, check out the `Reproducing results notebook <https://github.com/microsoft/robustdg/blob/master/docs/notebooks/reproduce_results.ipynb>`_. 

Roadmap
-------

* Support for more domain generalization algorithms like CSD and IRM. If you are an author of a DG algorithm and would like to contribute, please raise a  pull request `here <https://github.com/microsoft/robustdg/pulls>`_ or get in touch.

* More evaluation metrics based on adversarial attacks, privacy attacks like model inversion. If you'd like to see an evaluation metric implemented, please raise an issue `here <https://github.com/microsoft/robustdg/issues>`_.

Contributing
--------------

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_.
For more information see the `Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_ or
contact `opencode@microsoft.com <mailto:opencode@microsoft.com>`_ with any additional questions or comments.

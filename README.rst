Toolkit for Building Robust ML models that generalize to unseen domains (RobustDG)
==================================================================================
`Divyat Mahajan <https://divyat09.github.io/>`_, 
`Shruti Tople <https://www.microsoft.com/en-us/research/people/shtople/>`_, 
`Amit Sharma <http://www.amitsharma.in>`_

`ICML 2020 Paper <https://arxiv.org/abs/1909.12732>`_ | `MatchDG paper <https://arxiv.org/abs/2006.07500>`_ | `Privacy & DG Connection paper <http://divy.at/privacy_dg.pdf>`_

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

    cd data/
    python data_gen.py resnet18

**Train and evaluate ML model**

The following commands would train and evalute the MatchDG method on the Rotated MNIST dataset.

.. code:: shell

    python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 100 --batch_size 256 --pos_metric cos 
    
    python train.py --dataset rot_mnist --method_name matchdg_erm --match_case -1 --penalty_ws 0.1 --epochs 25 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18
      
    python test.py --dataset rot_mnist --method_name matchdg_erm --match_case -1 --penalty_ws 0.1 --epochs 25 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18 --test_metric acc
    
    python test.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --pos_metric cos --test_metric match_score


Demo
----

A quick introduction on how to use our repository can be accessed here in the `Getting Started notebook <https://github.com/microsoft/robustdg/blob/master/docs/notebooks/robustdg_getting_started.ipynb>`_.

If you are interested in reproducing results from the MatchDG paper, check out the `Reproducing results notebook <https://github.com/microsoft/robustdg/blob/master/docs/notebooks/reproducing_results_matchdg_paper.ipynb>`_. 

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

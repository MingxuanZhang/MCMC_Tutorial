***************
Getting Started
***************

.. toctree::
   :maxdepth: 2

   Installing Jupyter
   Cloning Repo + Starting Jupyter Server

==================
Installing Jupyter
==================

Before you play around with our tutorial, first install Jupyter Notebook:

.. code:: bash

    pip install jupyter

.. note::

    If ``pip`` is connected to Python 2.7 on your computer, then you can try ``pip3``.

    You can also install `Anaconda Python 3.6 <https://www.anaconda.com/download>`__, which downloads both Python 3.6 and Jupyter.

======================================
Cloning Repo + Starting Jupyter Server
======================================

After you have installed Jupyter, run the following commands:

.. code:: bash

    # Clone the tutorial repo
    git clone https://github.com/MingxuanZhang/MCMC_Tutorial
    # Go to the repo directory
    cd MCMC_Tutorial
    # Start the Jupyter server
    jupyter notebook

After executing the last command, you should be redirected to a localhost server in your browser. Click on ``mcmc_tutorial.ipynb`` in the list of files.
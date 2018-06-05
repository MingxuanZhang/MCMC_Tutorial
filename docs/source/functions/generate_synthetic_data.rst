***********************
generate_synthetic_data
***********************

.. code:: python

    # Function to generate synthetic data (and plot it)
    def generate_synthetic_data(astar, bstar, sigma, random_seed=1234, plot=False)

===========
Explanation
===========

.. todo::

    Fix explanation of what function does, below is just a copy from tutorial, which is not what we need for this section

Let's say we have a linear model that models the amount of ice cream you will consume (where :math:`y` has units of pints) after going on a bike ride of length :math:`x` (where :math:`x` has units of miles). So the true process model, based on the unknown true values of the slope and intercept parameters, which we can call :math:`\alpha^*` and :math:`\beta^*`, is:

.. math::

    y_{true} = \alpha^*x ~ + ~ \beta^*

We want to make this more realistic by incorporating some normally-distributed uncertainty, for instance, with mean :math:`0` and standard deviation :math:`\sigma`.  Call these uncertain measurements :math:`y_{meas}`:

.. math::

    y_{meas} = y_{true} ~ + ~ \epsilon = \alpha^*x ~ + ~ \beta^* ~ + ~ \epsilon

where :math:`\epsilon \sim N(0, 1)` is our normally-distributed measurement error. We now want to estimate :math:`\alpha` and :math:`\beta`. Let us call :math:`\hat{\alpha}` and :math:`\hat{\beta}` our estimates of these parameters. They lead to a set of estimates of the response, :math:`\hat{y}`:

.. math::
    \hat{y} = \hat{\alpha}x ~ + ~ \hat{\beta}

This technique is also known as the parameter estimation problem.

=============
Example Usage
=============

.. code:: python

    [(x_true, y_true), (x_meas, y_meas)] = generate_synthetic_data(astar=0.5, bstar=8, sigma=1, plot=True)
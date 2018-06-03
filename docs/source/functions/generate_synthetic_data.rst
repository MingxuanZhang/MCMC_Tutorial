***********************
generate_synthetic_data
***********************

.. toctree::
   :maxdepth: 2

   Explanation
   Example Usage

.. code:: python

    # Function to generate synthetic data (and plot it)
    def generate_synthetic_data(astar, bstar, sigma, random_seed=1234, plot=False)

===========
Explanation
===========

Let's say we have a linear model that models the amount of ice cream you will consume (where `y` has units of pints) after going on a bike ride of length `x` (where `x` has units of miles). So the true process model, based on the unknown true values of the slope and intercept parameters, which we can call `\alpha^*` and `\beta^*`, is:

.. math::

    y_{true} = \alpha^*x ~ + ~ \beta^*

We want to make this more realistic by incorporating some normally-distributed uncertainty, for instance, with mean `0` and standard deviation `\sigma`.  Call these uncertain measurements `y_{meas}`:

.. math::

    y_{meas} = y_{true} ~ + ~ \epsilon = \alpha^*x ~ + ~ \beta^* ~ + ~ \epsilon

where `\epsilon \sim N(0, 1)` is our normally-distributed measurement error. We now want to estimate `\alpha` and `\beta`. Let us call `\hat{\alpha}` and `\hat{\beta}` our estimates of these parameters. They lead to a set of estimates of the response, `\hat{y}`:

.. math::
    \hat{y} = \hat{\alpha}x ~ + ~ \hat{\beta}

This technique is also known as the parameter estimation problem.

=============
Example Usage
=============

.. code:: python

    [(x_true, y_true), (x_meas, y_meas)] = generate_synthetic_data(astar=0.5, bstar=8, sigma=1, plot=True)
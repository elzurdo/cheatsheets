# -*- coding: utf-8 -*-
# Copied and modified from `statistics_python.ipynb`

# # Entropy



# ## Mutual Information

#    
# For two discrete random variables  $𝑋$  and  $𝑌$ , the mutual information between  $𝑋$  and  $𝑌$ , denoted as  $𝐼(𝑋;𝑌)$ , measures how much information they share.  
#
# $$
# I(X;Y) \equiv D_{\text{KL}}(p_{X,Y}||p_Xp_Y) = \sum_x\sum_y p_{X,Y}(x,y) log_2\left(\frac{p_{X,Y}(x,y)}{p_X(x)p_Y(y)} \right)
# $$
#
# The mutual information could be thought of as how far  $𝑋$  and  $𝑌$  are from being independent.  
#
# * $I(X;Y)=0$: when $X$ and $Y$ are independent.  
# * $I(X;Y)=H(X)$: when $X$ and $Y$ are equal.  
#
# This second point means that $I(X;Y)$ may be larger than 1 ($I(X;Y)=1$ just means that they share one bit of information).  
# In general:  
# $I(X;Y) \le \text{min}[H(X),H(Y)]$
#
# **Usage**   
# * Probabilistic models: mutual information helps figure out which random variables we should directly model pairwise interactions with.
#
# **Useful Resources**
# * [`sklearn.metrics.mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html)
# * [`stackoverflow.com/questions/20491028`](https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy)
# * [`sklearn.feature_selection.mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)

# +
# Mutual information is about comparing the joint distribution of 𝑋 and 𝑌 
# with what the joint distribution would be if 𝑋 and 𝑌 were actually independent.

import numpy as np
import pandas as pd
from IPython.display import display

from scipy.special import kl_div

# To Do: fix edge cases where if p has any null values it returns nan
# Possible solution: replace all 0 values with epsilon=1.e-7
def information_divergence(p, q, base=2):    
    return kl_div(p, q).sum() / np.log(base)
    #return np.sum(p * np.log2(p / q)) # not using becuase of mishandling of zeros (but otherwise this is the basic idea)

def mutual_information(joint_prob_XY, verbose=True):
    # marginal distributions
    prob_X = joint_prob_XY.sum(axis=1)
    prob_Y = joint_prob_XY.sum(axis=0)
 
    # joint distribution if X and Y were actually independent
    joint_prob_XY_indep = np.outer(prob_X, prob_Y)
    
    if len(joint_prob_XY.shape) != 2:
        verbose = False
    
    if verbose:
        print('marginals')
        display(pd.DataFrame(prob_X, columns=['p_X']))
        display(pd.DataFrame(prob_Y, columns=['p_Y']).T)
        
        print('joint probability')
        df_joint_prob_XY = pd.DataFrame(joint_prob_XY)
        df_joint_prob_XY.columns.name = 'Y'
        df_joint_prob_XY.index.name = 'X'
        display(df_joint_prob_XY)
        
        print('joint probability if independent')
        df_joint_prob_XY_indep = pd.DataFrame(joint_prob_XY_indep)
        df_joint_prob_XY_indep.columns.name = 'Y'
        df_joint_prob_XY_indep.index.name = 'X'
        display(df_joint_prob_XY_indep)
        
    return information_divergence(joint_prob_XY, joint_prob_XY_indep)



# -

# ### Joint Distributions

# +
#epsilon = 1.e-7

n_vars = 2  # e.g, number of rolls of dice (currently works only for 2D ...)
n_dims = 6  # e.g, the number of options on each die
verbose = True

#distribution_type = 'zero MI'
distribution_type = 'max MI'
#distribution_type = 'zero entropy'

expected_MI = None
if 'zero MI' == distribution_type:
    joint_prob_XY = np.ones([n_dims, n_dims]) / (n_dims*n_dims)
    expected_MI = 0.
    explanation = f'information about one variable tells us nothing about the other\n(e.g, {n_vars} rolls of a fair {n_dims} sided die)'
elif 'max MI'  == distribution_type:
    joint_prob_XY = np.eye(n_dims) / n_dims
    expected_MI = np.log2(n_dims)
    explanation = 'information about one variable tells us everything about the other'
elif 'zero entropy' == distribution_type:
    joint_prob_XY = np.zeros([n_dims, n_dims])
    joint_prob_XY[0,0] = 1
    expected_MI = 0
    explanation = "if one of the variables doesn't have entropy, neither will the MI"
    

mi_outcome = mutual_information(joint_prob_XY, verbose=verbose)

is_close = None
if expected_MI is not None:
    is_close = np.isclose(mi_outcome, expected_MI)
    
print(f"{distribution_type} of n_dims={n_dims} results in MI={mi_outcome:0.3f} bits")
if is_close is not None:
    print(f"as expected because {explanation}.")
elif False == is_close:
    print(f"instead of the expected {is_close}")
# -

# ### Random Variables

# +
from itertools import product


# TODO: need to verify that joint_prob is normalised
# TODO: generalise from n_vars=2
# TODO: consider using A = np.identity(3); np.dstack([A]*3); but not sure ... examine: https://stackoverflow.com/questions/46029017
def generate_random_variables(joint_prob, sample_size, n_vars=2):
    n_dims = joint_prob.shape[0]
    
    choices = list(product(np.arange(n_dims), repeat=n_vars))
    
    def choice_to_values(idx):
        return choices[idx]
    
    choices_bool = np.arange(n_dims ** 2)
    
    choice_indexes = np.random.choice(choices_bool, sample_size, p=joint_prob.flatten())
    
    return np.array(list(zip(*(list(map(choice_to_values, choice_indexes))))))


def random_vars_to_joint_prob(rand_vars, n_dims):
    #n_vars = rand_vars.shape[0]
    joint_p = np.zeros([n_dims,n_dims])
    for idx in range(rand_vars.shape[-1]):
        joint_p[rand_vars[0][idx], rand_vars[1][idx]] += 1
        
    joint_p /= joint_p.sum()
    
    return joint_p


# +
# Graph Idea:
# horizontal axis = sample_size = 10, 30, 100 (in log space)
# vertical axis = mi_obsr - mi_true
# different lines for n_dims = 2, 3, 4, 6, 10, 20, 50

# each line should have credible region

n_dims = 6        # e.g, the number of options on each die
# n_vars = 2  # e.g, number of rolls of dice (currently works only for 2D ...)

sample_size = 10000
joint_prob_true = np.eye(n_dims) / n_dims  

rand_vars = generate_random_variables(joint_prob_true, sample_size, n_vars=2)
joint_prob_obsr = random_vars_to_joint_prob(rand_vars, n_dims=n_dims)

mi_true = mutual_information(joint_prob_true, verbose=False)
mi_obsr = mutual_information(joint_prob_obsr, verbose=False)

print(f"Sample size {sample_size:,}")
print(f"{mi_true:0.3f}: True MI")
print(f"{mi_obsr:0.3f}: Observed MI")
print(f"{mi_obsr - mi_true:0.3f} MI difference Observed - True ({100.*(mi_obsr - mi_true) / mi_true:0.3f}%)")

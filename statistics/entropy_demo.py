# -*- coding: utf-8 -*-
# Copied and modified from `statistics_python.ipynb`

# +
import numpy as np
import pandas as pd

from IPython.display import display
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2

# troubleshooting
# when autocompletion does not work
# %config Completer.use_jedi= False # stackoverflow

# +
# Visualising
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

FIG_WIDTH, FIG_HEIGHT = 8, 6

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["figure.figsize"] = FIG_WIDTH, FIG_HEIGHT
# plt.rcParams["hatch.linewidth"] = 0.2

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
# -

# # Entropy
# TBD

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
# * $I(X;Y)=H(X)$: when $X$ and $Y$ are equal. $H(X)$ is the marginal entropy.$ 
#
# This second point means that $I(X;Y)$ may be larger than 1 ($I(X;Y)=1$ just means that they share one bit of information).  
# In general:  
# $I(X;Y) \le \text{min}[H(X),H(Y)]$
#
# **Usage**   
# * Probabilistic models: mutual information helps figure out which random variables we should directly model pairwise interactions with.
#
# **Useful Equations**
#
# $$I(X;Y) \\ \equiv H(X) - H(X|Y) \\ \equiv H(Y) - H(Y|X) \\ \equiv H(X) + H(Y) - H(X,Y) \\ \equiv H(X,Y)-H(X|Y) - H(Y|X)$$, where $H(X|Y)$ and $H(Y|X)$ are the marginal entropies and $H(X,Y)$ is the joint entropy.  
#
# **Useful Resources**
# * [`sklearn.metrics.mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html)
# * [`stackoverflow.com/questions/20491028`](https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy)
# * [`sklearn.feature_selection.mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)

# +
# Mutual information is about comparing the joint distribution of 𝑋 and 𝑌 
# with what the joint distribution would be if 𝑋 and 𝑌 were actually independent.

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
        print('joint probability p(X,Y)')
        df_joint_prob_XY = pd.DataFrame(joint_prob_XY)
        df_joint_prob_XY.columns.name = 'Y'
        df_joint_prob_XY.index.name = 'X'
        display(df_joint_prob_XY)
        
        print('marginals')
        display(pd.DataFrame(prob_X, columns=['p_X']))
        display(pd.DataFrame(prob_Y, columns=['p_Y']).T)
        
        print(r'the marginals are used to build the joint probability if independent: p(X,Y) if X⊥Y')
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
def generate_random_variables(joint_prob, sample_size, n_vars=2, seed=None):
    n_dims = joint_prob.shape[0]
    
    choices = list(product(np.arange(n_dims), repeat=n_vars))
    
    def choice_to_values(idx):
        return choices[idx]
    
    choices_bool = np.arange(n_dims ** 2)
    
    if seed is not None:
        np.random.seed(seed)
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
# each line should have credible region

# TODO add more mi_type options, like zero mi or something in between
def stochastic_mi_estimate_accuracy(n_dims=6, n_vars=2, sample_size=100, seed=None, verbose=0):
    # n_dims = int        # e.g, the number of options on each die
    # n_vars = int        # e.g, number of rolls of dice (currently works only for 2D ...)
    
    mi_type = 'max MI' #'zero entropy' #'max MI'
    
    assert n_vars == 2   
    

    if 'max MI' == mi_type:
        joint_prob_true = np.eye(n_dims) / n_dims 
    elif 'zero MI' == mi_type:
        joint_prob_true= np.ones([n_dims, n_dims]) / (n_dims*n_dims)
    elif 'zero entropy':
        joint_prob_true = np.zeros([n_dims, n_dims])
        joint_prob_true[0,0] = 1

    rand_vars = generate_random_variables(joint_prob_true, sample_size, n_vars=2, seed=seed)
    joint_prob_obsr = random_vars_to_joint_prob(rand_vars, n_dims=n_dims)

    mi_true = mutual_information(joint_prob_true, verbose=False)
    mi_obsr = mutual_information(joint_prob_obsr, verbose=False)

    if verbose:
        print(f"Sample size {sample_size:,}")
        print(f"{mi_true:0.3f}: True MI")
        print(f"{mi_obsr:0.3f}: Observed MI")
        print(f"{mi_obsr - mi_true:0.3f} MI difference Observed - True ({100.*(mi_obsr - mi_true) / mi_true:0.3f}%)")
        
    return {'mi true': mi_true, 'mi observed': mi_obsr, 'mi difference': mi_obsr - mi_true, 'mi accuracy': (mi_obsr - mi_true) / mi_true}
    
stochastic_mi_estimate_accuracy(seed=1)

# +
# Graph Idea:
# horizontal axis = sample_size = 10, 30, 100 (in log space)
# vertical axis = mi_obsr - mi_true
# different lines for n_dims = 2, 3, 4, 6, 10, 20, 50

sample_sizes = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300]
l_n_dims = [2, 4, 6, 10] # 3, 4, 6, 10][::-1] #[2, 3, 4, 6, 10]
n_seeds = 5
credible_interval_width = 0.955

seeds = np.arange(n_seeds)
epsilon = (1. - credible_interval_width)/2

all_stats = {}
summary_stats = {}
for n_dims in l_n_dims:
    all_stats[n_dims], summary_stats[n_dims] = {}, {}
    for sample_size in sample_sizes:
        summary_stats[n_dims][sample_size] = {}
        aux_stats = {'mi true': [], 'mi observed': [], 'mi difference': [], 'mi accuracy': []}
        for seed in seeds:
            these_stats = stochastic_mi_estimate_accuracy(n_dims=n_dims, n_vars=2, sample_size=sample_size, seed=seed)
            
            for k, v in these_stats.items():
                aux_stats[k].append(v)
                
        for k in aux_stats:
            aux_stats[k] = np.array(aux_stats[k])
            
            summary_stats[n_dims][sample_size][f"{k} mean"] = np.mean(aux_stats[k])
            summary_stats[n_dims][sample_size][f"{k} std"] = np.std(aux_stats[k])
            
            summary_stats[n_dims][sample_size][f"{k} low ci"] = np.percentile(aux_stats[k], epsilon * 100)
            summary_stats[n_dims][sample_size][f"{k} high ci"] = np.percentile(aux_stats[k], (1. - epsilon) * 100)

            
        all_stats[n_dims][sample_size] = aux_stats
        


# +
# Graph Idea:
# horizontal axis = sample_size = 10, 30, 100 (in log space)
# vertical axis = mi_obsr - mi_true
# different lines for n_dims = 2, 3, 4, 6, 10, 20, 50

colors = ["purple", "orange", "red", "blue", "green"]

assert len(colors) >= len(summary_stats)

#plt.figure(figsize=(16,5))
fig, axes = plt.subplots(1, 2, figsize=(16,5))

for iplot, n_dims in enumerate(summary_stats):
    plot_stats = {'mi observed mean': [], 'mi observed low ci': [], 'mi observed high ci': [], 'mi true mean': [],
                  'mi accuracy mean': [], 'mi accuracy low ci': [], 'mi accuracy high ci': []
                 }
    for sample_size in summary_stats[n_dims]:
        for stat in plot_stats:
            plot_stats[stat].append(summary_stats[n_dims][sample_size][stat])
    
    for stat in plot_stats:
        plot_stats[stat] = np.array(plot_stats[stat])
    
    #plt.subplot(1, 2, 1)
    axes[0].plot(sample_sizes, plot_stats['mi true mean'], '--', color=colors[iplot], alpha=0.4)
    axes[0].plot(sample_sizes, plot_stats['mi observed mean'], '-o', color=colors[iplot], alpha=0.7)
    label = f"{n_dims:,}: {plot_stats['mi true mean'][0]:0.1f}"
    axes[0].fill_between(sample_sizes, 
                     plot_stats['mi observed low ci'],
                     plot_stats['mi observed high ci'],
                     color=colors[iplot], alpha=0.1,
                     label=label
                    )
    # ------
    
    #plt.plot(sample_sizes, plot_stats['mi true mean'] * , '--', color=colors[iplot], alpha=0.4)
    axes[1].plot(sample_sizes, plot_stats['mi accuracy mean'] * 100., '-o', color=colors[iplot], alpha=0.7)
    label = f"{n_dims:,}: {plot_stats['mi accuracy mean'][0] * 100.:0.1f}%"
    axes[1].fill_between(sample_sizes, 
                     plot_stats['mi accuracy low ci'] * 100.,
                     plot_stats['mi accuracy high ci'] * 100.,
                     color=colors[iplot], alpha=0.1,
                     label=label
                    )


axes[0].set_xlabel("sample size")
axes[0].set_ylabel("mutual information")
axes[0].legend(title="n dim: MI")
axes[0].set_xscale("log")

axes[1].plot(sample_sizes, [0] * len(sample_sizes), '--', color="gray", alpha=0.4)
axes[1].set_xlabel("sample size")
axes[1].set_ylabel("mi accuracy (%)")
axes[1].legend(title=f"n dim: % at n={sample_sizes[0]}")
axes[1].set_xscale("log")
# -

# ### Comparisons with Covariance

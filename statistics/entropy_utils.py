# Mutual information is about comparing the joint distribution of 𝑋 and 𝑌
# with what the joint distribution would be if 𝑋 and 𝑌 were actually independent.

import numpy as np
import pandas as pd
from scipy.special import kl_div
from IPython.display import display

# To Do: fix edge cases where if p has any null values it returns nan
# Possible solution: replace all 0 values with epsilon=1.e-7
def information_divergence(p, q, base=2):
    return kl_div(p, q).sum() / np.log(base)
    # return np.sum(p * np.log2(p / q)) # not using because of mishandling of zeros (but otherwise this is the basic idea)

def joint_p_mutual_information(joint_prob_XY, verbose=True):
    """

    :param joint_prob_XY: list of lists or np.array([N, 2])
    :param verbose: bool
    :return: float

    Example Usage:
    > joint_p_mutual_information([[0.5, 0], [0., 0.5]], verbose=False)
    > 1  # max value for 2 dimensions. information about one variable tells us everything about the other.
    > joint_p_mutual_information([[0.25, 0.25], [0.25, 0.25]], verbose=False)
    > 0  # the variables are independent
    > joint_p_mutual_information([[1, 0.], [0., 0.]], verbose=False)
    > 0 # the variables are independent
    > joint_p_mutual_information([[1./3, 0.,0.], [0., 1./3,0], [0., 0., 1./3]], verbose=False)
    > 1.584962500721156  # max value for 3 dimensions: np.log2(3) as in 1.58 bits of information
    > joint_p_mutual_information([[1./3, 0.,0.], [1/3., 0,0], [1./3, 0., 0.]], verbose=False)
    > 0  # i.e, the 2 variables are are independent
    """
    if isinstance(joint_prob_XY, list):
        joint_prob_XY = np.array(joint_prob_XY)

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

        print(
            r'the marginals are used to build the joint probability if independent: p(X,Y) if X⊥Y')
        df_joint_prob_XY_indep = pd.DataFrame(joint_prob_XY_indep)
        df_joint_prob_XY_indep.columns.name = 'Y'
        df_joint_prob_XY_indep.index.name = 'X'
        display(df_joint_prob_XY_indep)

    return information_divergence(joint_prob_XY, joint_prob_XY_indep)


# Todo: better annotatoins
#  n_dims is indicates the number of values possible in rand_vars
#  rand_vars may only get integer values from 0 to n_dims-1
def random_vars_to_joint_prob(rand_vars, n_dims):
    n_samples = rand_vars.shape[0]

    joint_p = np.zeros([n_dims, n_dims])
    for idx in range(n_samples):
        joint_p[rand_vars[idx][0], rand_vars[idx][1]] += 1

    joint_p /= joint_p.sum()

    return joint_p


def random_vars_to_mutual_information(rand_vars, n_dims, verbose=False):
    joint_p = random_vars_to_joint_prob(rand_vars, n_dims)
    return joint_p_mutual_information(joint_p, verbose=verbose)


# Todo: understand why np.cov(rand_vars).sum() does something else ...
def random_vars_to_covariance(rand_vars, verbose=False):
    cov_ = ((rand_vars[:, 0] - rand_vars[:, 0].mean()) * (
                rand_vars[:, 1] - rand_vars[:, 1].mean())).mean()

    if verbose:
        pearson = cov_ / np.std(rand_vars[:, 0]) / np.std(rand_vars[:, 1])
        print(f"{cov_:0.3f} - COV(X,Y)\n{pearson:0.3f} - Pearson R(X,Y)")

    return cov_





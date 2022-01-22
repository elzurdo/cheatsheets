from itertools import product
import numpy as np


# TODO: need to verify that joint_prob is normalised
# TODO: consider using A = np.identity(3); np.dstack([A]*3); but not sure ... examine: https://stackoverflow.com/questions/46029017
def joint_prob_to_random_variables(joint_prob, sample_size, seed=None):
    """
    :param joint_prob: np.array([N,N]), i.e, 2 variables
    :param sample_size: int
    :param n_vars:
    :param seed:
    :return:
    """
    n_dims = joint_prob.shape[0]

    n_vars = 2  # joint probability for 2 variables
    choices = list(product(np.arange(n_dims), repeat=n_vars))

    def choice_to_values(idx):
        return choices[idx]

    choices_bool = np.arange(n_dims ** 2)

    if seed is not None:
        np.random.seed(seed)
    choice_indexes = np.random.choice(choices_bool, sample_size, p=joint_prob.flatten())

    return np.array(list(zip(*(list(map(choice_to_values, choice_indexes)))))).T


def generate_joint_probability(n_dims, mi_type='max MI', flip=False):
    if 'max MI' == mi_type:
        joint_prob = np.eye(n_dims) / n_dims
    elif 'zero MI' == mi_type:
        joint_prob = np.ones([n_dims, n_dims]) / (n_dims * n_dims)
    elif 'zero entropy':
        joint_prob = np.zeros([n_dims, n_dims])
        joint_prob[0, 0] = 1

    # flipping has no effect on Mutual Information, but adding for completeness...
    if flip:
        joint_prob = np.flip(joint_prob, 0)

    return joint_prob

# mutual information
import numpy as np
import scipy
from tqdm.autonotebook import tqdm
import scipy.special
from sklearn.metrics.cluster.supervised import contingency_matrix
from scipy import sparse as sp

# modelling
from sklearn.externals.joblib import Parallel, delayed

from sklearn.metrics.cluster.expected_mutual_info_fast import (
    expected_mutual_information,
)


######## Mutual Information ############
def entropy(X):
    dist = np.array(X)
    N = float(len(dist))
    Nall = np.array([np.sum(dist == c) for c in set(dist)])
    pAll = Nall / N
    S = -np.sum(pAll * np.log2(pAll))
    var = np.var(-np.log2(pAll))
    return S, var


def joint_entropy(X, Y):
    N = float(len(X))
    contingency = contingency_matrix(X, Y, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)
    pAll = Nall / N
    S = -np.sum(pAll * np.log2(pAll))
    var = np.var(-np.log2(pAll))
    return S, var


def mutual_info(a, b):
    e_a, var_a = entropy(a)
    e_b, var_b = entropy(b)
    e_ab, var_ab = joint_entropy(a, b)
    return e_a + e_b - e_ab, np.sqrt((var_a + var_b + var_ab) / len(a))


######## Estimate Mutual Information ############
def est_entropy(X):

    dist = np.array(X)
    N = float(len(dist))
    Nall = [np.sum(dist == c) for c in set(dist)]
    pAll = np.array([float(Ni) * scipy.special.psi(float(Ni)) for Ni in Nall])
    S_hat = np.log2(N) - 1.0 / N * np.sum(pAll)
    var = np.var(scipy.special.psi(np.array(Nall, dtype="float32")))
    return S_hat, var


def est_joint_entropy(X, Y):

    N = float(len(X))
    contingency = contingency_matrix(X, Y, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)

    pAll = np.array([Ni * scipy.special.psi(Ni) for Ni in Nall if Ni > 0])
    S_hat = np.log2(N) - 1 / N * np.sum(pAll)
    var = np.var(scipy.special.psi(np.array(Nall, dtype="float32")))
    return S_hat, var


def est_mutual_info(a, b):
    e_a, var_a = est_entropy(a)
    e_b, var_b = est_entropy(b)
    e_ab, var_ab = est_joint_entropy(a, b)
    return e_a + e_b - e_ab, (var_a + var_b + var_ab) / len(a)


######## Estimate Mutual Information Faster ############


def est_mutual_info_p(a, b):
    # contingency matrix of a * b
    contingency = contingency_matrix(a, b, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)

    # entropy of a
    Na = np.ravel(contingency.sum(axis=0))  # number of A
    S_a, var_a = entropyp(Na)  # entropy with P(A) as input

    # entropy of b
    Nb = np.ravel(contingency.sum(axis=1))
    S_b, var_b = entropyp(Nb)
    S_ab, var_ab = entropyp(Nall)

    # mutual information
    MI = S_a + S_b - S_ab

    # uncertainty and variance of MI
    MI_var = var_a + var_b + var_ab

    uncertainty = np.sqrt((MI_var) / len(a))

    return MI, uncertainty


######## Mutual Information Faster ############
def entropyp(Nall):
    N = np.sum(Nall)
    pAll = np.array([float(Ni) * scipy.special.psi(float(Ni)) for Ni in Nall])
    S_hat = np.log2(N) - 1.0 / N * np.sum(pAll)
    var = np.var(scipy.special.psi(np.array(Nall, dtype="float32")))

    return S_hat, var


def mutual_info_p(a, b, average_method="arithmetic", normalize=False):
    """ Fast mutual information calculation based upon sklearn,
    but with estimation of uncertainty from Lin & Tegmark 2016
    """
    # contingency matrix of a * b
    contingency = contingency_matrix(a, b, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)
    N = len(a)
    # entropy of a
    Na = np.ravel(contingency.sum(axis=0))
    S_a, var_a = entropyp(Na / np.sum(Na))

    # entropy of b
    Nb = np.ravel(contingency.sum(axis=1))
    S_b, var_b = entropyp(Nb / np.sum(Nb))

    # joint entropy
    S_ab, var_ab = entropyp(Nall / N)

    # mutual information
    MI = S_a + S_b - S_ab

    # uncertainty and variance of MI
    MI_var = var_a + var_b + var_ab

    # normalization
    if normalize:
        # expected mutual information
        emi = expected_mutual_information(contingency, N)
        # normalization
        normalizer = _generalized_average(S_a, S_b, average_method)
        denominator = normalizer - emi
        if denominator < 0:
            denominator = min(denominator, -np.finfo("float64").eps)
        else:
            denominator = max(denominator, np.finfo("float64").eps)

        MI = (MI - emi) / denominator
        # this breaks MI_var - we would need to account for EMI
        MI_var = MI_var / denominator

    uncertainty = np.sqrt((MI_var) / len(a))

    return MI, uncertainty


def _generalized_average(U, V, average_method):
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError(
            "'average_method' must be 'min', 'geometric', " "'arithmetic', or 'max'"
        )


######## Mutual Information From distributions ############
def MI_from_distributions(sequences, dist, estimate=False, unclustered_element=None):
    np.random.seed()  # set seed
    # create distributions
    if np.sum([len(seq) > dist for seq in sequences]) == 0:
        return (np.nan, np.nan)

    distribution_a = np.concatenate(
        [seq[dist:] for seq in sequences if len(seq) > dist]
    )

    distribution_b = np.concatenate(
        [seq[:-dist] for seq in sequences if len(seq) > dist]
    )

    # mask unclustered so they are not considered in MI
    if unclustered_element is not None:
        mask = (distribution_a == unclustered_element) | (
            distribution_b == unclustered_element
        )
        distribution_a = distribution_a[mask == False]
        distribution_b = distribution_b[mask == False]

    # calculate MI

    if estimate:
        return est_mutual_info_p(distribution_a, distribution_b)
    else:
        return mutual_info_p(distribution_a, distribution_b, normalize=True)


def sequential_mutual_information(
    sequences,
    distances,
    n_jobs=1,
    verbosity=5,
    n_shuff_repeats=1,
    estimate=True,
    disable_tqdm=False,
    prefer="threads",
    unclustered_element=None,
):
    """
    Compute mutual information as a function of distance between sequences
    if n_jobs > 1,  will run in parallel
    """
    # convert to numeric for faster computation
    unique_elements = np.unique(np.concatenate(sequences))
    n_unique = len(unique_elements)
    seq_dict = {j: i for i, j in enumerate(unique_elements)}
    if n_unique < 256:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint8") for seq in sequences
        ]
    elif n_unique < 65535:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint16") for seq in sequences
        ]
    else:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint32") for seq in sequences
        ]

    if unclustered_element is not None:
        unclustered_element = seq_dict[unclustered_element]
        print(unclustered_element)
    else:
        unclustered_element = None

    # compute MI
    if n_jobs == 1:
        MI = [
            MI_from_distributions(
                sequences,
                dist,
                estimate=estimate,
                unclustered_element=unclustered_element,
            )
            for dist_i, dist in enumerate(
                tqdm(distances, leave=False, disable=disable_tqdm)
            )
        ]
        distances_rep = np.repeat(distances, n_shuff_repeats)
        shuff_MI = [
            MI_from_distributions(
                [np.random.permutation(i) for i in sequences], dist, estimate=estimate
            )
            for dist_i, dist in enumerate(
                tqdm(distances_rep, leave=False, disable=disable_tqdm)
            )
        ]

        shuff_MI = np.reshape(shuff_MI, (len(distances), n_shuff_repeats, 2))
        shuff_MI = np.mean(shuff_MI, axis=1)

    else:
        with Parallel(n_jobs=n_jobs, verbose=verbosity, prefer=prefer) as parallel:
            MI = parallel(
                delayed(MI_from_distributions)(
                    sequences,
                    dist,
                    estimate=estimate,
                    unclustered_element=unclustered_element,
                )
                for dist_i, dist in enumerate(
                    tqdm(distances, leave=False, disable=disable_tqdm)
                )
            )

        with Parallel(n_jobs=n_jobs, verbose=verbosity, prefer=prefer) as parallel:
            distances_rep = np.repeat(distances, n_shuff_repeats)
            shuff_MI = parallel(
                delayed(MI_from_distributions)(
                    [np.random.permutation(i) for i in sequences],
                    dist,
                    estimate=estimate,
                    unclustered_element=unclustered_element,
                )
                for dist_i, dist in enumerate(
                    tqdm(distances_rep, leave=False, disable=disable_tqdm)
                )
            )
            shuff_MI = np.reshape(shuff_MI, (len(distances), n_shuff_repeats, 2))
            shuff_MI = np.mean(shuff_MI, axis=1)

    return np.array(MI).T, np.array(shuff_MI).T

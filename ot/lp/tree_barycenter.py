from ..backend import get_backend
import numpy as np
from .solver_tree import topological_sort
from ..utils import proj_simplex

# Author : Ali Boudjema

# IMPORTANT : ON PREND COMME CONVENTION QUE LES FEUILLES SONT LES PREMIERS SOMMETS DE L'ARBRE


def wgm(values, weights):
    # Returns the weighted geometric median

    nx = get_backend(values, weights)

    sorted_indices = np.argsort(values, kind="stable")

    values_sorted = values[sorted_indices]
    weights_sorted = weights[sorted_indices]

    cum_weights = nx.cumsum(weights_sorted)

    id = nx.searchsorted(cum_weights, 0.5 - 1e9)

    return values_sorted[id]


def get_measure(z, tree, length):
    # Retrieves the measure from a vector after the wgm

    n = z.shape[0]

    nx = get_backend(length)

    measure = nx.zeros(n)

    for i in range(n):
        p = tree[i]

        if i == p:
            measure[i] += 1
        else:
            measure[i] += z[i] / length[i]
            measure[p] -= z[i] / length[i]

    return measure


def tree_barycenter(tree, length, measure, weights, topo_order=None):
    r"""
    Computes the tree wasserstein barycenter for a given tree between multiplie empirical distributions

    Parameters
    ----------
    tree : array_like, shape(n)
        ancestor of each node in the tree (ancestor of root is root)
    length : array_like, shape(n)
        length of the arc above each node (length of root is 0)
    measure : array_like, shape(m, n)
        distributions in the tree
    weights : array_like, shape(m)
        weight of each distribution

    Returns
    -------
    barycenter : array_like, shape(n)
        distribution of the barycenter

    Reference
    ---------
    The code is a direct implementation of the algorithm described in
    Tree-Wasserstein Barycenter for Large-Scale Multilevel Clustering and Scalable Bayes

    """
    n_measure = measure.shape[0]
    n_node = tree.shape[0]

    assert n_measure == weights.shape[0], "dimension error"

    nx = get_backend(measure, weights, length)

    z_measure = nx.zeros((n_measure, n_node))

    if topo_order is None:
        topo_order = topological_sort(tree)

    for cur_node in topo_order:
        p = tree[cur_node]

        for id_mes in range(n_measure):
            z_measure[id_mes][cur_node] += measure[id_mes][cur_node]

            if cur_node != p:
                z_measure[id_mes][p] += z_measure[id_mes][cur_node]

    z = nx.zeros(n_node)

    for cur_node in range(n_node):
        z_measure[:, cur_node] *= length[cur_node]

        z[cur_node] = wgm(z_measure[:, cur_node], weights)

    return get_measure(z, tree, length)


def get_B_matrix(tree, length, nb_leafs):
    nx = get_backend(length)

    rows = []
    col = []
    data = []

    for cur_leaf in range(nb_leafs):
        cur_node = cur_leaf

        while cur_node != tree[cur_node]:
            rows.append(cur_node)
            col.append(cur_leaf)
            data.append(length[cur_node])
            cur_node = tree[cur_node]

    nb_rows = len(tree)
    nb_col = nb_leafs

    B = nx.coo_matrix(data, rows, col, shape=(nb_rows, nb_col), type_as=length)

    return B


def fixed_support_tree_barycenter(tree, length, measures, nb_itr=100, step=0.1):
    nx = get_backend(length)
    nb_leafs = measures.shape[1]

    B = get_B_matrix(tree, length, nb_leafs)

    nb_mes = measures.shape[0]
    nb_nodes = tree.shape[0]

    cur_mes = nx.ones(nb_leafs) / nb_leafs

    B_mes = [B.dot(measures[i]) for i in range(nb_mes)]

    B_mes = np.asarray(B_mes)

    sigma = np.argsort(B_mes, axis=0)

    B_mes_sorted = np.take_along_axis(B_mes, sigma, axis=0)

    for itr in range(nb_itr):
        cur_B = B.dot(cur_mes)

        idx = np.zeros(nb_nodes, dtype=int)

        for node in range(nb_nodes):
            idx[node] = np.searchsorted(B_mes_sorted[:, node], cur_B[node]) + 1

        z = -nb_mes + 2 * idx - 2

        g = B.T.dot(z) / nb_mes

        cur_mes -= step * g

        cur_mes = proj_simplex(cur_mes)

    return cur_mes

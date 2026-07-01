from ..backend import get_backend
import numpy as np
from .solver_tree import topological_sort
from ..utils import proj_simplex
from ..utils import list_to_array

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


def get_gradient(cur_B, B_mes_sorted, B, nb_mes, nb_nodes):
    idx = np.zeros(nb_nodes, dtype=int)

    for node in range(nb_nodes):
        idx[node] = np.searchsorted(B_mes_sorted[:, node], cur_B[node]) + 1

    z = -nb_mes + 2 * idx - 2

    g = B.T.dot(z) / nb_mes

    return g


def fixed_support_tree_barycenter(tree, length, measures, nb_itr=100, step=0.1):
    nx = get_backend(length, measures)
    nb_leafs = measures.shape[1]

    B = get_B_matrix(tree, length, nb_leafs)

    nb_mes = measures.shape[0]
    nb_nodes = tree.shape[0]

    cur_mes = nx.ones(nb_leafs) / nb_leafs

    B_mes = list_to_array([B.dot(measures[i]) for i in range(nb_mes)])

    sigma = nx.argsort(B_mes, axis=0)

    B_mes_sorted = nx.take_along_axis(B_mes, sigma, axis=0)

    for itr in range(nb_itr):
        cur_B = B.dot(cur_mes)

        g = get_gradient(cur_B, B_mes_sorted, B, nb_mes, nb_nodes)

        cur_mes -= step * g

        cur_mes = proj_simplex(cur_mes)

    return cur_mes


def pre_process_trees(tree_list, length_list, measures):
    nx = get_backend(length_list, measures)

    nb_leafs = measures.shape[2]

    prepared_trees = []

    for tree, length, mes in zip(tree_list, length_list, measures):
        B = get_B_matrix(tree, length, nb_leafs)
        nb_mes = mes.shape[0]

        B_mes = list_to_array([B.dot(mes[i]) for i in range(nb_mes)])

        B_mes_sorted = nx.sort(B_mes, axis=0)

        prepared_trees.append(
            {
                "B": B,
                "B_mes_sorted": B_mes_sorted,
                "nb_nodes": tree.shape[0],
                "nb_mes": nb_mes,
            }
        )

    return prepared_trees


def sliced_fixed_support_tree_barycenter(
    tree_list, length_list, measures, nb_itr=100, step=0.01, tol=1e-5
):
    """
    Parameters
    -----------
    tree_list : array_like, shape (t, n)
    length_list : array_like, shape (t, n)
    measures : array_like, shape (t, m, k)
    """

    nx = get_backend(length_list, measures)

    nb_leafs = measures.shape[2]

    cur_mes = nx.ones(nb_leafs) / nb_leafs

    prepared_trees = pre_process_trees(tree_list, length_list, measures)

    nb_tree = len(prepared_trees)

    for itr in range(nb_itr):
        old_mes = cur_mes.copy()
        g_total = nx.zeros(nb_leafs)

        for tree_data in prepared_trees:
            B = tree_data["B"]
            B_mes_sorted = tree_data["B_mes_sorted"]
            nb_nodes = tree_data["nb_nodes"]
            nb_mes = tree_data["nb_mes"]

            cur_B = B.dot(cur_mes)

            g_tree = get_gradient(cur_B, B_mes_sorted, B, nb_mes, nb_nodes)
            g_total += g_tree

        g_mean = g_total / nb_tree

        g_mean /= np.linalg.norm(g_mean, ord=2)

        cur_mes -= step * g_mean

        cur_mes = proj_simplex(cur_mes)

        if np.linalg.norm(cur_mes - old_mes) < tol:
            break

    return cur_mes

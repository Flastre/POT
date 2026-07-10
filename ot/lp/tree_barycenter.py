from ..backend import get_backend
import numpy as np
from ..utils import proj_simplex
from ..utils import list_to_array

# Author : Ali Boudjema

# IMPORTANT : ON PREND COMME CONVENTION QUE LES FEUILLES SONT LES PREMIERS SOMMETS DE L'ARBRE


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


def fixed_support_tree_barycenter(
    tree_list, length_list, measures, nb_itr=100, step=0.01, tol=1e-5, init_measure=None
):
    """
    Computes the Tree-Wasserstein (or Tree-Sliced) barycenter for one or multiple trees,
    with the constraint that the support of the barycenter is fixed at the leaves.
    It is assumed that the leaves correspond to the first nodes of the tree (indices 0 to k-1).
    While the number of leaves (k) must be strictly identical across all structures to ensure
    consistent alignment, the total number of nodes (leaves + internal nodes) can
    freely vary from one tree to another.

    If a single tree structure is provided (e.g., 1D arrays for tree_list/length_list
    and 2D for measures), the function automatically expands their dimensions to 3D
    internal structures to handle them uniformly as a multi-tree setting with t=1.

    Parameters
    -----------
    tree_list : array_like
        A single tree of shape (n_t,) or a list of t trees where each tree has shape (n_t,).
        n_t is the total number of nodes in that specific tree. tree[i] contains the index
        of the parent of node i (with tree[root] == root).
    length_list : array_like
        The edge weights corresponding to tree_list. A single array of shape (n_t,) or a list
        of t arrays, where the t-th array has shape (n_t,) and contains the length of the edge
        connecting node i to its parent.
    measures : array_like, shape (t, m, k) or (m,k)
        The input probability distributions mapped to the leaves.
        k is the fixed number of leaves shared by all trees, and m is the number
        of measures. Accepts a 2D array of shape (m, k) for a single tree, or a 3D array
        of shape (t, m, k) in a multi-tree setting.
    nb_tr : int, optional
        the maximal number of iterations for the subgradient descent
    step : float, optional
        the step size of the descent
    tol : float, optional
        Convergence tolerance. The descent stops if the L2 norm of the difference
        between two consecutive iterations is smaller than tol.
    init_measure : array_like, shape (k), optional
        The starting point of the descent, default is None

    Returns
    -------
    cur_mes : array_like, shape (k)
        The computed fixed-support barycenter supported on the k leaves.

    References
    ----------
    "Fixed Support Tree-Sliced Wasserstein Barycenter"
    """

    nx = get_backend(measures)

    if tree_list.ndim == 1:
        tree_list = nx.reshape(tree_list, (1, *tree_list.shape))
        length_list = nx.reshape(length_list, (1, *length_list.shape))

    if measures.ndim == 2:
        measures = nx.reshape(measures, (1, *measures.shape))
        measures = nx.tile(measures, (tree_list.shape[0], 1, 1))

    assert (
        tree_list.shape[0] == length_list.shape[0] == measures.shape[0]
        and tree_list.shape[1] == length_list.shape[1]
    ), "dimension error in the input"

    prepared_trees = pre_process_trees(tree_list, length_list, measures)

    nb_leafs = measures.shape[2]

    if init_measure is None:
        cur_mes = nx.ones(nb_leafs) / nb_leafs
    else:
        cur_mes = init_measure

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

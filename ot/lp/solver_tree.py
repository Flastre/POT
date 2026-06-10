from ..backend import get_backend
import numpy as np
from collections import deque

"""
Solver for the tree wasserstein distance problem
"""

# Author : Ali Boudjema


def topological_sort(tree):
    r"""
    Computes a topological sort of the given tree
    The tree is an array : tree[i] is the direct ancestor of node i, and tree[root] = root

    Je ne vérifie pas que l'arbre est du bon format
    Pas de backend ici ?
    """

    n = len(tree)

    in_degree = np.zeros(n)

    for cur_node in range(n):
        if cur_node != tree[cur_node]:
            in_degree[tree[cur_node]] += 1

    queue = deque()

    for cur_node in range(n):
        if in_degree[cur_node] == 0:
            queue.append(cur_node)

    topo_order = []

    while queue:
        cur_node = queue.popleft()
        topo_order.append(cur_node)

        if cur_node != tree[cur_node]:
            in_degree[tree[cur_node]] -= 1

            if in_degree[tree[cur_node]] == 0:
                queue.append(tree[cur_node])

    return np.array(topo_order)


def tree_wasserstein(tree, length, u_weights, v_weights, topo_order=None):
    r"""
    Computes the tree wasserstein distance for a given tree between two empirical distributions

    Parameters
    ----------
    tree : array_like, shape(n)
        ancestor of each node in the tree (ancestor of root is root)
    length : array_like, shape(n)
        length of the arc above each node (length of root is 0)
    u_weights : array_like, shape(n, ...)
        weights of the first empirical distributions
    v_weights : array_like, shape(n, ...)
        weights of the second empirical distributions
    topo_order : array_like, shape(n)
        topological order of the tree, optional

    Returns
    -------
    cost : float/array_like, shape(...)
        The tree wasserstein distance
    """

    n = len(tree)

    assert (
        n == len(length) == u_weights.shape[0] == v_weights.shape[0]
    ), "dimension error in the input"

    if topo_order is None:
        topo_order = topological_sort(tree)

    nx = get_backend(length, u_weights, v_weights)

    u_cumweights = nx.copy(u_weights)
    v_cumweights = nx.copy(v_weights)

    cost = 0

    for i in range(n):
        cur_node = topo_order[i]

        cost += length[cur_node] * nx.abs(
            u_cumweights[cur_node] - v_cumweights[cur_node]
        )

        u_cumweights[tree[cur_node]] = (
            u_cumweights[cur_node] + u_cumweights[tree[cur_node]]
        )
        v_cumweights[tree[cur_node]] = (
            v_cumweights[cur_node] + v_cumweights[tree[cur_node]]
        )

    return cost

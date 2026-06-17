from ..backend import get_backend


def wgm(values, weights):
    # Returns the weighted geometric median

    nx = get_backend(values, weights)

    sorted_indices = nx.argsort(values)

    values_sorted = values[sorted_indices]
    weights_sorted = weights[sorted_indices]

    cum_weights = nx.cumsum(weights_sorted)

    id = nx.searchsorted(cum_weights, 0.5)

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


def tree_barycenter(tree, length, measure, weights):
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

    for cur_node in range(n_node):
        p = tree[cur_node]

        for id_mes in range(n_measure):
            z_measure[id_mes][cur_node] += measure[id_mes][cur_node]

            if cur_node != p:
                z_measure[id_mes][p] += measure[id_mes][cur_node]

    z = nx.zeros(n_node)

    for cur_node in range(n_node):
        z_measure[:, cur_node] *= length[cur_node]

        z[cur_node] = wgm(z_measure[:, cur_node], weights)

    return get_measure(z, tree, length)

"""Tests for the tree wassesterin barycenter"""

import numpy as np
import pytest

import ot
from ot.lp import fixed_support_tree_barycenter
from ot.utils import list_to_array


def test_symetry_reflexivity(nx):
    n = 50
    k = 30

    tree, length = ot.utils.random_tree_fixed_leaves(n, k)

    u = nx.rand(k)
    u = u / nx.sum(u)

    v = nx.rand(k)
    v = v / nx.sum(v)

    length = nx.from_numpy(length)

    np.testing.assert_allclose(
        fixed_support_tree_barycenter(tree, length, nx.stack([u, v])),
        fixed_support_tree_barycenter(tree, length, nx.stack([v, u])),
        rtol=1e-2,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        fixed_support_tree_barycenter(
            tree, length, nx.stack([u, u]), nb_itr=10000, step=0.001
        ),
        u,
        rtol=5e-4,
        atol=1e-3,
    )


def test_multiple_trees(nx):
    n = 40
    k = 20

    nb_trees = 5

    data = [ot.utils.random_tree_fixed_leaves(n, k) for _ in range(nb_trees)]
    trees, lengths = map(list, zip(*data))

    for i in range(nb_trees):
        lengths[i] = nx.from_numpy(lengths[i])
        trees[i] = nx.from_numpy(trees[i])

    trees = list_to_array(trees, nx=nx)
    lengths = list_to_array(lengths, nx=nx)

    u = nx.rand(k)
    u = u / nx.sum(u)

    np.testing.assert_allclose(
        fixed_support_tree_barycenter(
            trees, lengths, nx.stack([u, u]), nb_itr=10000, step=0.001
        ),
        u,
        rtol=5e-4,
        atol=1e-3,
    )

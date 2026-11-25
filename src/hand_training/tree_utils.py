"""训练状态数组的辅助检查：设备维增删、有限值检查和断点恢复一致性检查。"""
import jax
import jax.numpy as jnp
import numpy as np

def add_device_axis(array_tree):
    return jax.tree.map(lambda array: jnp.expand_dims(array, 0), array_tree)

def remove_device_axis(array_tree):
    return jax.tree.map(lambda array: array[0], array_tree)

def all_arrays_finite(array_tree):
    return all(np.isfinite(np.asarray(array)).all() for array in jax.tree.leaves(array_tree))

def assert_trees_equal(left_tree, right_tree):
    assert jax.tree.structure(left_tree) == jax.tree.structure(right_tree)
    for left_array, right_array in zip(jax.tree.leaves(left_tree), jax.tree.leaves(right_tree)):
        np.testing.assert_array_equal(left_array, right_array)

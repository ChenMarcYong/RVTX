import rvtx
import numpy as np

aabb = rvtx.AABB()

assert np.allclose(aabb.min, np.repeat(np.finfo(np.float32).max, 3))
assert np.allclose(aabb.max, np.repeat(np.finfo(np.float32).min, 3))

assert aabb.invalid == True

aabb = rvtx.AABB(0, 1)

assert np.allclose(aabb.min, [0, 0, 0])
assert np.allclose(aabb.max, [1, 1, 1])

aabb = rvtx.AABB([0, 0, 0], [1, 1, 1])

assert np.allclose(aabb.min, [0, 0, 0])
assert np.allclose(aabb.max, [1, 1, 1])

assert np.allclose(aabb.centroid, [0.5, 0.5, 0.5])

assert np.isclose(aabb.radius, np.linalg.norm(aabb.max - aabb.min) / 2)

aabb.grow([1, 1, 1])

assert np.allclose(aabb.min, [-1, -1, -1])
assert np.allclose(aabb.max, [2, 2, 2])

aabb.update([-5, 5, 0])

assert np.allclose(aabb.min, [-5, -1, -1])
assert np.allclose(aabb.max, [2, 5, 2])

aabb.update(rvtx.AABB([-10, -10, 0], [0, 10, 10]))

assert np.allclose(aabb.min, [-10, -10, -1])
assert np.allclose(aabb.max, [2, 10, 10])
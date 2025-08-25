import rvtx
import numpy as np

EPSILON = 1e-7

t = rvtx.Transform()

assert np.allclose(t.position, [0, 0, 0])
assert np.allclose(t.rotation, [1, 0, 0, 0])

t2 = t.copy()

assert np.allclose(t2.position, t.position) 
assert np.allclose(t2.rotation, t.rotation)

t2.position = [1, 2, 3]
t2.rotation = [4, 3, 2, 1]

assert np.allclose(t2.position, t.position) == False
assert np.allclose(t2.rotation, t.rotation) == False

assert np.allclose(t2.position, [1, 2, 3])
assert np.allclose(t2.rotation, [4, 3, 2, 1])

t3 = rvtx.Transform(position = [1, 2, 3], rotation = [1, 2, 3, 4])

assert np.allclose(t3.position, [1, 2, 3])
assert np.allclose(t3.rotation, [1, 2, 3, 4])

assert np.allclose(t.front, [0, 0, -1])
assert np.allclose(t.left, [-1, 0, 0])
assert np.allclose(t.up, [0, 1, 0])

assert np.allclose(t.get(), [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])

t.set(t3.get())

assert np.allclose(t.position, t3.position)
assert np.allclose(t.rotation, t3.rotation)
assert np.allclose(t.get(), t3.get())

t.reset()

assert np.allclose(t.position, [0, 0, 0])
assert np.allclose(t.rotation, [1, 0, 0, 0])

t.move([1, 1, 1])

assert np.allclose(t.position, [-1, 1, -1])

t.rotate([np.pi, 0, 0])

assert np.allclose(t.rotation, [0, 1, 0, 0], EPSILON, EPSILON)

t.rotate([0, -1, 0, 0])

assert np.allclose(t.rotation, [1, 0, 0, 0], EPSILON, EPSILON)

t.rotate_pitch(np.pi)

assert np.allclose(t.rotation, [0, -1, 0, 0], EPSILON, EPSILON)

t.rotate_pitch(-np.pi)
t.rotate_yaw(np.pi)

assert np.allclose(t.rotation, [0, 0, -1, 0], EPSILON, EPSILON)

t.rotate_yaw(-np.pi)
t.rotate_roll(np.pi)

assert np.allclose(t.rotation, [0, 0, 0, 1], EPSILON, EPSILON)

t.reset()
t.look_at([0, 1, 0])

assert np.allclose(t.front, [0, 1, 0], EPSILON, EPSILON)

aabb = rvtx.AABB(0, 1)

t.reset()
t.look_at(aabb, 60)

assert np.allclose(t.front, [0, 0, 1], EPSILON, EPSILON)


t.look_at([0, 1, 0], [0, 0, 1])

assert np.allclose(t.front, [0, 1, 0], EPSILON, EPSILON)
assert np.allclose(t.up, [0, 0, 1], EPSILON, EPSILON)
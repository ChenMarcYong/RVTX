import rvtx
import numpy as np

scene = rvtx.Scene()

camera = scene.create_camera()

assert np.allclose(camera.position, [0, 0, 0])
assert np.allclose(camera.rotation, [1, 0, 0, 0])
assert camera.controller == rvtx.Camera.Controller.Trackball
assert camera.projection == rvtx.Camera.Projection.Perspective
assert np.allclose(camera.viewport, [16, 9])
assert camera.near == 1
assert camera.far == 10000
assert np.isclose(camera.fov, np.pi / 4)
assert np.allclose(camera.target.position, [0, 0, 0])
assert camera.target.distance == 0
assert np.isclose(camera.aspect_ratio, 16 / 9)
assert camera.is_perspective == True

t = rvtx.Transform([1, 2, 3], [1, 2, 3, 4])
viewport = [9, 16]
target = rvtx.Camera.Target([10, 20, 30], 25)
projection = rvtx.Camera.Projection.Orthographic
controller = rvtx.Camera.Controller.Freefly

camera = scene.create_camera(t, target, projection, controller, viewport)

assert np.allclose(camera.position, [1, 2, 3])
assert np.allclose(camera.rotation, [1, 2, 3, 4])
assert camera.controller == rvtx.Camera.Controller.Freefly
assert camera.projection == rvtx.Camera.Projection.Orthographic
assert np.allclose(camera.viewport, [9, 16])
assert camera.near == 1
assert camera.far == 10000
assert np.isclose(camera.fov, np.pi / 4)
assert np.allclose(camera.target.position, [10, 20, 30])
assert camera.target.distance == 25
assert np.isclose(camera.aspect_ratio, 9 / 16)
assert camera.is_perspective == False
assert np.allclose(camera.current_projection_matrix, camera.orthographic_projection_matrix)
camera.projection = rvtx.Camera.Projection.Perspective
assert np.allclose(camera.current_projection_matrix, camera.perspective_projection_matrix)

camera = rvtx.create_camera(t, target, projection, controller, viewport)

camera.look_at([0, 0, 0])
t.look_at([0, 0, 0])

assert np.allclose(camera.rotation, t.rotation)
assert np.allclose(camera.position, t.position)
assert np.allclose(camera.front, t.front)
assert np.allclose(camera.up, t.up)
assert np.allclose(camera.left, t.left)

aabb = rvtx.AABB(-5, 17)

camera.look_at(aabb, 37)
t.look_at(aabb, 37)

assert np.allclose(camera.rotation, t.rotation)
assert np.allclose(camera.position, t.position)
assert np.allclose(camera.front, t.front)
assert np.allclose(camera.up, t.up)
assert np.allclose(camera.left, t.left)

camera.look_at([1, 2, 4], [5, 7, 8])
t.look_at([1, 2, 4], [5, 7, 8])

assert np.allclose(camera.rotation, t.rotation)
assert np.allclose(camera.position, t.position)
assert np.allclose(camera.front, t.front)
assert np.allclose(camera.up, t.up)
assert np.allclose(camera.left, t.left)

camera.rotate([1, 2, 3, 4])
t.rotate([1, 2, 3, 4])

assert np.allclose(camera.rotation, t.rotation)
assert np.allclose(camera.position, t.position)
assert np.allclose(camera.front, t.front)
assert np.allclose(camera.up, t.up)
assert np.allclose(camera.left, t.left)

camera.rotate([2, 3, 4])
t.rotate([2, 3, 4])

assert np.allclose(camera.rotation, t.rotation)
assert np.allclose(camera.position, t.position)
assert np.allclose(camera.front, t.front)
assert np.allclose(camera.up, t.up)
assert np.allclose(camera.left, t.left)

camera.rotate_pitch(0.8)
t.rotate_pitch(0.8)

assert np.allclose(camera.rotation, t.rotation)
assert np.allclose(camera.position, t.position)
assert np.allclose(camera.front, t.front)
assert np.allclose(camera.up, t.up)
assert np.allclose(camera.left, t.left)

camera.rotate_yaw(1.4)
t.rotate_yaw(1.4)

assert np.allclose(camera.rotation, t.rotation)
assert np.allclose(camera.position, t.position)
assert np.allclose(camera.front, t.front)
assert np.allclose(camera.up, t.up)
assert np.allclose(camera.left, t.left)

camera.rotate_roll(12.7)
t.rotate_roll(12.7)

assert np.allclose(camera.rotation, t.rotation)
assert np.allclose(camera.position, t.position)
assert np.allclose(camera.front, t.front)
assert np.allclose(camera.up, t.up)
assert np.allclose(camera.left, t.left)
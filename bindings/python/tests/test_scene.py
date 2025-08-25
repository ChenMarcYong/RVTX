import rvtx
import numpy as np

scene = rvtx.Scene()

camera = scene.create_camera()

scene.main_camera = camera

assert scene.main_camera == camera

camera = rvtx.create_camera()

scene.main_camera = camera

assert scene.main_camera == camera

rvtx.set_main_camera(camera)

assert rvtx.main_scene.main_camera == camera

rvtx.main_scene.main_camera = camera

assert rvtx.main_scene.main_camera == camera
assert rvtx.main_scene.main_camera == scene.main_camera
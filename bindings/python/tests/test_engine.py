import rvtx
import numpy as np

window = rvtx.Window("Test Engine", 800, 600)

engine = rvtx.Engine(window)

assert engine.controls_enabled == True

engine.controls_enabled = False

assert engine.controls_enabled == False

try:
    engine.render()
except RuntimeError as e:
    assert str(e) == "No camera provided in engine render call and scene main camera is null!"

try:
    engine.update()
except RuntimeError as e:
    assert str(e) == "No camera provided in engine update call and scene main camera is null!"

try:
    engine.screenshot()
except RuntimeError as e:
    assert str(e) == "No camera provided in engine screenshot call and scene main camera is null!"

camera = rvtx.create_camera()
rvtx.set_main_camera(camera)

engine.render() # Should not raise an exception

assert engine.update() == True

assert engine.screenshot() is not None

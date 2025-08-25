import rvtx
import numpy as np

for nb_points in [2, 3, 4, 20, 50, 10, 5, 13]:
    positions = np.random.rand(nb_points, 3)

    path = rvtx.Vec3Path(positions)

    assert np.allclose(path.values, positions)
    assert np.isclose(path.duration, 1.)
    assert path.spline_type == rvtx.SplineType.CatmullRom
    assert np.allclose(path.at(0.0), positions[0])
    assert np.allclose(path.at(1.0), positions[-1])

    path.spline_type = rvtx.SplineType.Linear

    assert path.spline_type == rvtx.SplineType.Linear

    path.duration = 10.

    assert np.isclose(path.duration, 10.)
    assert np.allclose(path.at(0.0), positions[0])
    assert np.allclose(path.at(10.0), positions[-1])

    positions = np.random.rand(nb_points, 3)

    path.values = positions

    assert np.allclose(path.values, positions)
    assert np.isclose(path.duration, 10.)
    assert path.spline_type == rvtx.SplineType.Linear
    assert np.allclose(path.at(0.0), positions[0])
    assert np.allclose(path.at(10.0), positions[-1])

    time_interplator = rvtx.Vec3PathTimeInterpolator(path)

    assert np.allclose(time_interplator.current, positions[0])
    assert np.allclose(time_interplator.current_value, positions[0])
    assert np.allclose(time_interplator.value, positions[0])
    assert time_interplator.current_time == 0.
    assert time_interplator.ended == False

    step_count = np.random.randint(1, 100)
    step = 1.0 / step_count
    for i in range(step_count):
        assert np.isclose(time_interplator.current_time, i * step)
        assert time_interplator.ended == False
        time_interplator.step(step)
    
    assert np.isclose(time_interplator.current_time, 1.0)
    assert time_interplator.ended == True

    time_interplator.reset()

    assert np.allclose(time_interplator.current, positions[0])
    assert time_interplator.current_time == 0.

    time_interplator.current_time = 0.5

    assert np.isclose(time_interplator.current_time, 0.5)

    framerate = np.random.rand() * 119 + 1
    keyframes_count = int(path.duration * framerate)
    keyframe_interplator = rvtx.Vec3PathKeyframeInterpolator(path, framerate)

    assert np.allclose(keyframe_interplator.current, positions[0])
    assert np.allclose(keyframe_interplator.current_value, positions[0])
    assert np.allclose(keyframe_interplator.value, positions[0])
    assert keyframe_interplator.frame_count == keyframes_count
    assert keyframe_interplator.current_frame == 0
    assert keyframe_interplator.ended == False

    for i in range(keyframes_count):
        assert keyframe_interplator.current_frame == i
        assert keyframe_interplator.ended == False
        keyframe_interplator.step()

    assert keyframe_interplator.current_frame == keyframes_count
    assert keyframe_interplator.ended == True

    keyframe_interplator.reset()

    assert np.allclose(keyframe_interplator.current, positions[0])
    assert keyframe_interplator.current_frame == 0

    keyframe_interplator.current_frame = int(keyframes_count / 2)

    assert keyframe_interplator.current_frame == int(keyframes_count / 2)


    rotations = np.random.rand(nb_points, 4)

    path = rvtx.QuatPath(rotations)

    assert np.allclose(path.values, rotations)
    assert np.isclose(path.duration, 1.)
    assert path.spline_type == rvtx.SplineType.CatmullRom
    assert np.allclose(path.at(0.0), rotations[0])
    assert np.allclose(path.at(1.0), rotations[-1])

    path.duration = 10.

    assert np.isclose(path.duration, 10.)
    assert np.allclose(path.at(0.0), rotations[0])
    assert np.allclose(path.at(10.0), rotations[-1])

    rotations = np.random.rand(nb_points, 4)

    path.values = rotations

    assert np.allclose(path.values, rotations)
    assert np.isclose(path.duration, 10.)
    assert path.spline_type == rvtx.SplineType.CatmullRom
    assert np.allclose(path.at(0.0), rotations[0])
    assert np.allclose(path.at(10.0), rotations[-1])

    time_interplator = rvtx.QuatPathTimeInterpolator(path)

    assert np.allclose(time_interplator.current, rotations[0])
    assert np.allclose(time_interplator.current_value, rotations[0])
    assert np.allclose(time_interplator.value, rotations[0])
    assert time_interplator.current_time == 0.
    assert time_interplator.ended == False

    step_count = np.random.randint(1, 100)
    step = 1.0 / step_count
    for i in range(step_count):
        assert np.isclose(time_interplator.current_time, i * step)
        assert time_interplator.ended == False
        time_interplator.step(step)
    
    assert np.isclose(time_interplator.current_time, 1.0)
    assert time_interplator.ended == True

    time_interplator.reset()

    assert np.allclose(time_interplator.current, rotations[0])
    assert time_interplator.current_time == 0.

    time_interplator.current_time = 0.5

    assert np.isclose(time_interplator.current_time, 0.5)

    framerate = np.random.rand() * 119 + 1
    keyframes_count = int(path.duration * framerate)
    keyframe_interplator = rvtx.QuatPathKeyframeInterpolator(path, framerate)

    assert np.allclose(keyframe_interplator.current, rotations[0])
    assert np.allclose(keyframe_interplator.current_value, rotations[0])
    assert np.allclose(keyframe_interplator.value, rotations[0])
    assert keyframe_interplator.frame_count == keyframes_count
    assert keyframe_interplator.current_frame == 0
    assert keyframe_interplator.ended == False

    for i in range(keyframes_count):
        assert keyframe_interplator.current_frame == i
        assert keyframe_interplator.ended == False
        keyframe_interplator.step()

    assert keyframe_interplator.current_frame == keyframes_count
    assert keyframe_interplator.ended == True

    keyframe_interplator.reset()

    assert np.allclose(keyframe_interplator.current, rotations[0])
    assert keyframe_interplator.current_frame == 0

    keyframe_interplator.current_frame = int(keyframes_count / 2)

    assert keyframe_interplator.current_frame == int(keyframes_count / 2)
import rvtx
import numpy as np

for nb_points in [1, 2, 99, 100, 101, 888]:
    rand_points_vec3 = np.random.rand(nb_points, 3)
    rand_colors_vec3 = np.random.rand(nb_points, 3)
    rand_radii = np.random.rand(nb_points)

    pc = rvtx.create_point_cloud(rand_points_vec3, rand_colors_vec3, rand_radii)

    rand_points_radii_concat = np.concatenate((rand_points_vec3, np.expand_dims(rand_radii, axis=1)), axis=1)
    assert np.allclose(pc.points, rand_points_radii_concat)
    assert np.allclose(pc.colors, np.concatenate((rand_colors_vec3, np.ones((nb_points, 1))), axis=1))

    rand_points_vec3 = np.random.rand(nb_points, 3)
    rand_color_vec3 = [1, 2, 3]
    rand_radii = 5

    pc = rvtx.create_point_cloud(rand_points_vec3, rand_color_vec3, rand_radii)

    assert np.allclose(pc.points, np.concatenate((rand_points_vec3, np.ones((nb_points, 1)) * rand_radii), axis=1))
    assert np.allclose(pc.colors, np.repeat([[1., 2., 3., 1.]], nb_points, axis=0))

    scene = rvtx.Scene()

    rand_points_vec4 = np.random.rand(nb_points, 4)
    rand_colors_vec4 = np.random.rand(nb_points, 4)

    pc = scene.create_point_cloud(rand_points_vec4, rand_colors_vec4)

    assert np.allclose(pc.points, rand_points_vec4)
    assert np.allclose(pc.colors, rand_colors_vec4)

    rand_points_vec4 = np.random.rand(nb_points, 4)
    rand_colors_vec3 = np.random.rand(nb_points, 3)

    pc = scene.create_point_cloud(rand_points_vec4, rand_colors_vec3)

    assert np.allclose(pc.points, rand_points_vec4)
    assert np.allclose(pc.colors, np.concatenate((rand_colors_vec3, np.ones((nb_points, 1))), axis=1))

    rand_points_vec4 = np.random.rand(nb_points, 4)
    rand_colors_vec4 = np.random.rand(nb_points, 4)

    pc.points = rand_points_vec4
    pc.colors = rand_colors_vec4

    assert np.allclose(pc.points, rand_points_vec4)
    assert np.allclose(pc.colors, rand_colors_vec4)
import rvtx
import numpy as np

for nb_points in [2, 99, 100, 101, 888]:
    nodes = np.random.rand(nb_points, 4)
    edges = [i for i in range(nb_points)]
    nodes_colors = np.random.rand(nb_points, 4)
    edges_data = np.random.rand(int(len(edges) / 2), 4)

    graph = rvtx.create_graph(nodes, edges, nodes_colors, edges_data)

    assert len(graph.nodes) == nb_points
    assert len(graph.nodes_colors) == nb_points
    assert len(graph.edges) == nb_points
    assert len(graph.edges_data) == int(len(edges) / 2)

    assert np.allclose(graph.nodes, nodes)
    assert np.allclose(graph.nodes_colors, nodes_colors)
    assert np.allclose(graph.edges, edges)
    assert np.allclose(graph.edges_data, edges_data)

    nodes = np.random.rand(nb_points, 3)
    edges = [i for i in range(nb_points)]
    nodes_radii = np.random.rand(nb_points)
    edges_radii = np.random.rand(int(len(edges) / 2))
    nodes_colors = np.random.rand(nb_points, 3)
    edges_colors = np.random.rand(int(len(edges) / 2), 3)

    graph = rvtx.create_graph(nodes, edges, nodes_radii, edges_radii, nodes_colors, edges_colors)

    assert len(graph.nodes) == nb_points
    assert len(graph.edges) == nb_points
    assert len(graph.nodes_colors) == nb_points
    assert len(graph.edges_data) == int(len(edges) / 2)

    nodes_radii_concat = np.concatenate((nodes, np.expand_dims(nodes_radii, axis=1)), axis=1)
    assert np.allclose(graph.nodes, nodes_radii_concat)
    assert np.allclose(graph.nodes_colors, np.concatenate((nodes_colors, np.ones((nb_points, 1))), axis=1))
    assert np.allclose(graph.edges, edges)
    edges_colors_radii_concat = np.concatenate((edges_colors, np.expand_dims(edges_radii, axis=1)), axis=1)
    assert np.allclose(graph.edges_data, edges_colors_radii_concat)


    nodes = np.random.rand(nb_points, 3)
    edges = [i for i in range(nb_points)]
    nodes_radii = 5
    edges_radii = 5
    nodes_colors = [1, 2, 3]
    edges_colors = [4, 5, 6]

    graph = rvtx.create_graph(nodes, edges, nodes_radii, edges_radii, nodes_colors, edges_colors)

    assert len(graph.nodes) == nb_points
    assert len(graph.nodes_colors) == nb_points
    assert len(graph.edges) == nb_points
    assert len(graph.edges_data) == int(len(edges) / 2)

    nodes_radii_concat = np.concatenate((nodes, np.ones((nb_points, 1)) * nodes_radii), axis=1)
    assert np.allclose(graph.nodes, nodes_radii_concat)
    assert np.allclose(graph.nodes_colors, np.repeat([[1., 2., 3., 1.]], nb_points, axis=0))
    assert np.allclose(graph.edges, edges)
    edges_colors_radii_concat = np.repeat([[4., 5., 6., 5.]], int(len(edges) / 2), axis=0)
    assert np.allclose(graph.edges_data, edges_colors_radii_concat)

    scene = rvtx.Scene()

    nodes = np.random.rand(nb_points, 3)
    edges = [i for i in range(nb_points)]
    edges_radius = 7
    edges_color = [3, 2, 1]

    graph = scene.create_graph(nodes, edges, edges_radius, edges_color)

    assert len(graph.nodes) == nb_points
    assert len(graph.nodes_colors) == nb_points
    assert len(graph.edges) == nb_points
    assert len(graph.edges_data) == int(len(edges) / 2)

    nodes_radii_concat = np.concatenate((nodes, np.ones((nb_points, 1)) * edges_radius), axis=1)
    assert np.allclose(graph.nodes, nodes_radii_concat)
    assert np.allclose(graph.nodes_colors, np.repeat([[3., 2., 1., 1.]], nb_points, axis=0))
    assert np.allclose(graph.edges, edges)
    edges_colors_radii_concat = np.repeat([[3., 2., 1., 7.]], int(len(edges) / 2), axis=0)
    assert np.allclose(graph.edges_data, edges_colors_radii_concat)

    nodes = np.random.rand(nb_points, 4)
    edges_radius = 9
    edges_color = [3, 3, 2]
    connection_type = rvtx.Graph.ConnectionType.Lines

    graph = scene.create_graph(nodes, edges_radius, edges_color, connection_type)

    assert len(graph.nodes) == nb_points
    assert len(graph.nodes_colors) == nb_points
    assert len(graph.edges) == nb_points - nb_points % 2
    assert len(graph.edges_data) == int(len(graph.edges) / 2)

    assert np.allclose(graph.nodes, nodes)
    assert np.allclose(graph.nodes_colors, np.repeat([[3., 3., 2., 1.]], nb_points, axis=0))
    edges_colors_radii_concat = np.repeat([[3., 3., 2., 9.]], int(len(graph.edges) / 2), axis=0)
    assert np.allclose(graph.edges_data, edges_colors_radii_concat)

    nodes = np.random.rand(nb_points, 3)
    edges_radius = 4
    edges_color = [9, 10, 11]
    connection_type = rvtx.Graph.ConnectionType.LineStrip

    graph = scene.create_graph(nodes, edges_radius, edges_color, connection_type)

    assert len(graph.nodes) == nb_points
    assert len(graph.nodes_colors) == nb_points
    assert len(graph.edges) == (nb_points - 1) * 2
    assert len(graph.edges_data) == int(len(graph.edges) / 2)

    nodes_radii_concat = np.concatenate((nodes, np.ones((nb_points, 1)) * edges_radius), axis=1)
    assert np.allclose(graph.nodes, nodes_radii_concat)
    assert np.allclose(graph.nodes_colors, np.repeat([[9., 10., 11., 1.]], nb_points, axis=0))
    edges_colors_radii_concat = np.repeat([[9., 10., 11., 4.]], int(len(graph.edges) / 2), axis=0)
    assert np.allclose(graph.edges_data, edges_colors_radii_concat)

    nodes = np.random.rand(nb_points, 3)
    edges_radius = 4
    edges_color = [9, 10, 11]
    connection_type = rvtx.Graph.ConnectionType.LineLoop

    graph = scene.create_graph(nodes, edges_radius, edges_color, connection_type)

    assert len(graph.nodes) == nb_points
    assert len(graph.nodes_colors) == nb_points
    assert len(graph.edges) == nb_points * 2
    assert len(graph.edges_data) == int(len(graph.edges) / 2)

    nodes_radii_concat = np.concatenate((nodes, np.ones((nb_points, 1)) * edges_radius), axis=1)
    assert np.allclose(graph.nodes, nodes_radii_concat)
    assert np.allclose(graph.nodes_colors, np.repeat([[9., 10., 11., 1.]], nb_points, axis=0))
    edges_colors_radii_concat = np.repeat([[9., 10., 11., 4.]], int(len(graph.edges) / 2), axis=0)
    assert np.allclose(graph.edges_data, edges_colors_radii_concat)
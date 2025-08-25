import rvtx
import numpy as np

atom = rvtx.Atom()

assert atom.symbol == rvtx.Atom.Symbol.Unknown
assert np.isclose(atom.radius, 1.2)
assert atom.name == "Unknown"

atom.symbol = rvtx.Atom.Symbol.C

assert atom.symbol == rvtx.Atom.Symbol.C
assert np.isclose(atom.radius, 1.77)
assert atom.name == "Carbon"

assert atom.residue_id == 0

atom.residue_id = 5

assert atom.residue_id == 5

chain = rvtx.Chain()

assert chain.id == ""
assert chain.name == ""
assert chain.residues.start == 0
assert chain.residues.end == 0

chain.id = "A"
chain.name = "AAA"
chain.residues = rvtx.Range(5, 10)

assert chain.id == "A"
assert chain.name == "AAA"
assert chain.residues.start == 5
assert chain.residues.end == 10

residue = rvtx.Residue()

assert residue.name == ""
assert residue.type == rvtx.Residue.Type.Molecule
assert residue.id == 0
assert residue.atoms.start == 0
assert residue.atoms.end == 0
assert residue.bonds.start == 0
assert residue.bonds.end == 0
assert residue.chain_id == 0

residue.name = "ALA"
residue.type = rvtx.Residue.Type.Water
residue.id = 5
residue.atoms = rvtx.Range(5, 10)
residue.bonds = rvtx.Range(10, 15)
residue.chain_id = 10

assert residue.name == "ALA"
assert residue.type == rvtx.Residue.Type.Water
assert residue.id == 5
assert residue.atoms.start == 5
assert residue.atoms.end == 10
assert residue.bonds.start == 10
assert residue.bonds.end == 15
assert residue.chain_id == 10

bond = rvtx.Bond()

assert bond.first == 0
assert bond.second == 0

bond.first = 5
bond.second = 10

assert bond.first == 5
assert bond.second == 10

molecule = rvtx.load_molecule("./data/1AGA.mmtf")

assert molecule.id == "1AGA"
assert molecule.name == "THE AGAROSE DOUBLE HELIX AND ITS FUNCTION IN AGAROSE GEL STRUCTURE"
assert len(molecule.data) == 126
assert len(molecule.atoms) == 126
assert len(molecule.chains) == 2
assert len(molecule.residues) == 12
assert np.allclose(molecule.position, [0, 0, 0])
assert np.allclose(molecule.rotation, [1, 0, 0, 0])
assert np.allclose(molecule.position, molecule.transform.position)
assert np.allclose(molecule.rotation, molecule.transform.rotation)
assert np.allclose(molecule.position, molecule.T.position)
assert np.allclose(molecule.rotation, molecule.T.rotation)
assert np.allclose(molecule.transform.position, molecule.T.position)
assert np.allclose(molecule.transform.rotation, molecule.T.rotation)

assert molecule.representation == rvtx.Molecule.Representation.vanDerWaals

molecule.representation = rvtx.Molecule.Representation.BallAndStick

assert molecule.representation == rvtx.Molecule.Representation.BallAndStick

molecule.position = [1, 2, 3]
molecule.rotation = [1, 2, 3, 4]

assert np.allclose(molecule.position, [1, 2, 3])
assert np.allclose(molecule.rotation, [1, 2, 3, 4])
assert np.allclose(molecule.position, molecule.transform.position)
assert np.allclose(molecule.rotation, molecule.transform.rotation)
assert np.allclose(molecule.position, molecule.T.position)
assert np.allclose(molecule.rotation, molecule.T.rotation)
assert np.allclose(molecule.transform.position, molecule.T.position)
assert np.allclose(molecule.transform.rotation, molecule.T.rotation)

scene = rvtx.Scene()

molecule = scene.load_molecule("./data/1AGA.mmtf", rvtx.Molecule.Representation.BallAndStick)

assert molecule.id == "1AGA"
assert molecule.name == "THE AGAROSE DOUBLE HELIX AND ITS FUNCTION IN AGAROSE GEL STRUCTURE"
assert len(molecule.data) == 126
assert len(molecule.atoms) == 126
assert len(molecule.chains) == 2
assert len(molecule.residues) == 12
assert np.allclose(molecule.position, [0, 0, 0])
assert np.allclose(molecule.rotation, [1, 0, 0, 0])
assert np.allclose(molecule.position, molecule.transform.position)
assert np.allclose(molecule.rotation, molecule.transform.rotation)
assert np.allclose(molecule.position, molecule.T.position)
assert np.allclose(molecule.rotation, molecule.T.rotation)
assert np.allclose(molecule.transform.position, molecule.T.position)
assert np.allclose(molecule.transform.rotation, molecule.T.rotation)

assert molecule.representation == rvtx.Molecule.Representation.BallAndStick

molecule.representation = rvtx.Molecule.Representation.vanDerWaals

assert molecule.representation == rvtx.Molecule.Representation.vanDerWaals

molecule.position = [1, 2, 3]
molecule.rotation = [1, 2, 3, 4]

assert np.allclose(molecule.position, [1, 2, 3])
assert np.allclose(molecule.rotation, [1, 2, 3, 4])
assert np.allclose(molecule.position, molecule.transform.position)
assert np.allclose(molecule.rotation, molecule.transform.rotation)
assert np.allclose(molecule.position, molecule.T.position)
assert np.allclose(molecule.rotation, molecule.T.rotation)
assert np.allclose(molecule.transform.position, molecule.T.position)
assert np.allclose(molecule.transform.rotation, molecule.T.rotation)
import rvtx
import numpy as np

r = rvtx.Range()

assert r.start == 0
assert r.end == 0

r.start = 5
r.end = 10

assert r.start == 5
assert r.end == 10
assert r.size == 5

r = rvtx.Range(2, 12)

assert r.start == 2
assert r.end == 12
assert r.size == 10
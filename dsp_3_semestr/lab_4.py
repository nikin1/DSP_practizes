import numpy as np

def pl(d, d0):
    return 74 + 10 * np.log10(d / d0)


Gt = 5
Gr = 1
Pt = 35
Qs = 6
d0 = 100
d = [1000, 3000, 5000]


pl1 = pl(d[0])
pl2 = pl(d[1])
pl3 = pl(d[2])



import rpy
import numpy as np

import cwpath, lasso
import pylab
reload(cwpath)

rpy.r("library(lars)")
rpy.r("data(diabetes)")
X = rpy.r("diabetes$x")
Y = rpy.r("diabetes$y")

c = cwpath.CoordWise((X,Y), lasso.Lasso)
p = c.problem.penalty
p.value = 1.e+03

c.fit(1.0)
p.value = 1.0e+03
print c.current

c.path()

def plot_path(c):
    b = np.asarray([a[1][0] for a in c.results])
    l1 = np.sum(np.fabs(b), axis=1)
    for i in range(10):
        pylab.scatter(l1, b[:,i])
    pylab.show()
    print b.shape
    return b

b = plot_path(c)



import numpy as np
import time

from periodic_kdtree import PeriodicKDTree, PeriodicCKDTree



m = 3
n = 10000
r = 1000

bounds = np.ones(m)
data = np.concatenate((np.random.randn(n//2,m),
    np.random.randn(n-n//2,m)+np.ones(m)))
queries = np.concatenate((np.random.randn(r//2,m),
    np.random.randn(r-r//2,m)+np.ones(m)))

print "dimension %d, %d points" % (m,n)

t = time.time()
T1 = PeriodicKDTree(bounds, data)
print "PeriodicKDTree constructed:\t%g" % (time.time()-t)
t = time.time()
T2 = PeriodicCKDTree(bounds, data)
print "PeriodicCKDTree constructed:\t%g" % (time.time()-t)

t = time.time()
w = T1.query(queries)
print "PeriodicKDTree %d lookups:\t%g" % (r, time.time()-t)
del w

t = time.time()
w = T2.query(queries)
print "PeriodicCKDTree %d lookups:\t%g" % (r, time.time()-t)
del w

T3 = PeriodicCKDTree(bounds,data,leafsize=n)
t = time.time()
w = T3.query(queries)
print "flat PeriodicCKDTree %d lookups:\t%g" % (r, time.time()-t)
del w

t = time.time()
w1 = T1.query_ball_point(queries, 0.2)
print "PeriodicKDTree %d ball lookups:\t%g" % (r, time.time()-t)

t = time.time()
w2 = T2.query_ball_point(queries, 0.2)
print "PeriodicCKDTree %d ball lookups:\t%g" % (r, time.time()-t)

t = time.time()
w3 = T3.query_ball_point(queries, 0.2)
print "flat PeriodicCKDTree %d ball lookups:\t%g" % (r, time.time()-t)

all_good = True
for a, b in zip(w1, w2):
    if sorted(a) != sorted(b):
        all_good = False
for a, b in zip(w1, w3):
    if sorted(a) != sorted(b):
        all_good = False

print "Ball lookups agree? %s" % str(all_good)

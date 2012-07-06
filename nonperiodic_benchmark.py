import numpy as np
import time

from scipy.spatial import KDTree, cKDTree



m = 3
n = 10000
r = 1000

data = np.concatenate((np.random.randn(n//2,m),
    np.random.randn(n-n//2,m)+np.ones(m)))
queries = np.concatenate((np.random.randn(r//2,m),
    np.random.randn(r-r//2,m)+np.ones(m)))

print "dimension %d, %d points" % (m,n)

t = time.time()
T1 = KDTree(data)
print "KDTree constructed:\t%g" % (time.time()-t)
t = time.time()
T2 = cKDTree(data)
print "cKDTree constructed:\t%g" % (time.time()-t)

t = time.time()
w = T1.query(queries)
print "KDTree %d lookups:\t%g" % (r, time.time()-t)
del w

t = time.time()
w = T2.query(queries)
print "cKDTree %d lookups:\t%g" % (r, time.time()-t)
del w

T3 = cKDTree(data,leafsize=n)
t = time.time()
w = T3.query(queries)
print "flat cKDTree %d lookups:\t%g" % (r, time.time()-t)
del w

t = time.time()
w1 = T1.query_ball_point(queries, 0.2)
print "KDTree %d ball lookups:\t%g" % (r, time.time()-t)

t = time.time()
w2 = T2.query_ball_point(queries, 0.2)
print "cKDTree %d ball lookups:\t%g" % (r, time.time()-t)

t = time.time()
w3 = T3.query_ball_point(queries, 0.2)
print "flat cKDTree %d ball lookups:\t%g" % (r, time.time()-t)

all_good = True
for a, b in zip(w1, w2):
    if sorted(a) != sorted(b):
        all_good = False
for a, b in zip(w1, w3):
    if sorted(a) != sorted(b):
        all_good = False

print "Ball lookups agree? %s" % str(all_good)

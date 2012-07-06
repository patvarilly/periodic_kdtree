# test_periodic_kdtree.py
#
# Unit tests for periodic_kdtree.py
#
# Written by Patrick Varilly, 6 Jul 2012
# Released under the scipy license

import numpy as np
from periodic_kdtree import PeriodicKDTree, PeriodicCKDTree

from numpy.testing import assert_equal, assert_array_equal, \
    assert_array_almost_equal, assert_almost_equal, \
    assert_, run_module_suite

from scipy.spatial import minkowski_distance

class ConsistencyTests:
    def distance(self, x, y, p):
        return minkowski_distance(np.zeros(x.shape), self.pbcs(x - y), p)

    def pbcs(self, x):
        return x - np.where(self.bounds > 0,
                            (np.round(x / self.bounds) * self.bounds), 0.0)

    def test_nearest(self):
        x = self.x
        d, i = self.kdtree.query(x, 1)
        assert_almost_equal(d**2,np.sum(self.pbcs(x-self.data[i])**2))
        eps = 1e-8
        assert_(np.all(np.sum(self.pbcs(self.data-x[np.newaxis,:])**2,axis=1)>d**2-eps))

    def test_m_nearest(self):
        x = self.x
        m = self.m
        dd, ii = self.kdtree.query(x, m)
        d = np.amax(dd)
        i = ii[np.argmax(dd)]
        assert_almost_equal(d**2,np.sum(self.pbcs(x-self.data[i])**2))
        eps = 1e-8
        assert_equal(np.sum(np.sum(self.pbcs(self.data-x[np.newaxis,:])**2,axis=1)<d**2+eps),m)

    def test_points_near(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, distance_upper_bound=d)
        eps = 1e-8
        hits = 0
        for near_d, near_i in zip(dd,ii):
            if near_d==np.inf:
                continue
            hits += 1
            assert_almost_equal(near_d**2,np.sum(self.pbcs(x-self.data[near_i])**2))
            assert_(near_d<d+eps, "near_d=%g should be less than %g" % (near_d,d))
        assert_equal(np.sum(np.sum(self.pbcs(self.data-x[np.newaxis,:])**2,axis=1)<d**2+eps),hits)

    def test_points_near_l1(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=1, distance_upper_bound=d)
        eps = 1e-8
        hits = 0
        for near_d, near_i in zip(dd,ii):
            if near_d==np.inf:
                continue
            hits += 1
            assert_almost_equal(near_d,self.distance(x,self.data[near_i],1))
            assert_(near_d<d+eps, "near_d=%g should be less than %g" % (near_d,d))
        assert_equal(np.sum(self.distance(self.data,x,1)<d+eps),hits)
    def test_points_near_linf(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=np.inf, distance_upper_bound=d)
        eps = 1e-8
        hits = 0
        for near_d, near_i in zip(dd,ii):
            if near_d==np.inf:
                continue
            hits += 1
            assert_almost_equal(near_d,self.distance(x,self.data[near_i],np.inf))
            assert_(near_d<d+eps, "near_d=%g should be less than %g" % (near_d,d))
        assert_equal(np.sum(self.distance(self.data,x,np.inf)<d+eps),hits)

    def test_approx(self):
        x = self.x
        k = self.k
        eps = 0.1
        d_real, i_real = self.kdtree.query(x, k)
        d, i = self.kdtree.query(x, k, eps=eps)
        assert_(np.all(d<=d_real*(1+eps)))


class test_random(ConsistencyTests):
    def setUp(self):
        self.n = 100
        self.m = 4
        self.data = np.random.randn(self.n, self.m)
        self.bounds = np.ones(self.m)
        self.kdtree = PeriodicKDTree(self.bounds, self.data,leafsize=2)
        self.x = np.random.randn(self.m)
        self.d = 0.2
        self.k = 10

class test_random_far(test_random):
    def setUp(self):
        test_random.setUp(self)
        self.x = np.random.randn(self.m)+10

class test_small(ConsistencyTests):
    def setUp(self):
        self.data = np.array([[0,0,0],
                              [0,0,1],
                              [0,1,0],
                              [0,1,1],
                              [1,0,0],
                              [1,0,1],
                              [1,1,0],
                              [1,1,1]])
        self.bounds = 1.1 * np.ones(3)
        self.kdtree = PeriodicKDTree(self.bounds, self.data)
        self.n = self.kdtree.n
        self.m = self.kdtree.m
        self.x = np.random.randn(3)
        self.d = 0.5
        self.k = 4

    def test_nearest(self):
        assert_array_equal(
                self.kdtree.query((0,0,0.1), 1),
                (0.1,0))
    def test_nearest_two(self):
        assert_array_almost_equal(
                self.kdtree.query((0,0,0.1), 2),
                ([0.1,np.sqrt(0.1**2 + 0.1**2)],[0,2]))
class test_small_nonleaf(test_small):
    def setUp(self):
        test_small.setUp(self)
        self.kdtree = PeriodicKDTree(self.bounds, self.data,leafsize=1)

class test_small_compiled(test_small):
    def setUp(self):
        test_small.setUp(self)
        self.kdtree = PeriodicCKDTree(self.bounds, self.data)
class test_small_nonleaf_compiled(test_small):
    def setUp(self):
        test_small.setUp(self)
        self.kdtree = PeriodicCKDTree(self.bounds, self.data,leafsize=1)
class test_random_compiled(test_random):
    def setUp(self):
        test_random.setUp(self)
        self.kdtree = PeriodicCKDTree(self.bounds, self.data)
class test_random_far_compiled(test_random_far):
    def setUp(self):
        test_random_far.setUp(self)
        self.kdtree = PeriodicCKDTree(self.bounds, self.data)

class test_vectorization:
    def setUp(self):
        self.data = np.array([[0,0,0],
                              [0,0,1],
                              [0,1,0],
                              [0,1,1],
                              [1,0,0],
                              [1,0,1],
                              [1,1,0],
                              [1,1,1]])
        self.bounds = 1.1 * np.ones(3)
        self.kdtree = PeriodicKDTree(self.bounds, self.data)

    def test_single_query(self):
        d, i = self.kdtree.query(np.array([0,0,0]))
        assert_(isinstance(d,float))
        assert_(np.issubdtype(i, int))

    def test_vectorized_query(self):
        d, i = self.kdtree.query(np.zeros((2,4,3)))
        assert_equal(np.shape(d),(2,4))
        assert_equal(np.shape(i),(2,4))

    def test_single_query_multiple_neighbors(self):
        s = 23
        kk = 27*self.kdtree.n+s
        d, i = self.kdtree.query(np.array([0,0,0]),k=kk)
        assert_equal(np.shape(d),(kk,))
        assert_equal(np.shape(i),(kk,))
        assert_(np.all(~np.isfinite(d[-s:])))
        assert_(np.all(i[-s:]==self.kdtree.n))

    def test_vectorized_query_multiple_neighbors(self):
        s = 23
        kk = 27*self.kdtree.n+s
        d, i = self.kdtree.query(np.zeros((2,4,3)),k=kk)
        assert_equal(np.shape(d),(2,4,kk))
        assert_equal(np.shape(i),(2,4,kk))
        assert_(np.all(~np.isfinite(d[:,:,-s:])))
        assert_(np.all(i[:,:,-s:]==self.kdtree.n))

    def test_single_query_all_neighbors(self):
        d, i = self.kdtree.query([0,0,0],k=None,distance_upper_bound=1.1)
        assert_(isinstance(d,list))
        assert_(isinstance(i,list))

    def test_vectorized_query_all_neighbors(self):
        d, i = self.kdtree.query(np.zeros((2,4,3)),k=None,distance_upper_bound=1.1)
        assert_equal(np.shape(d),(2,4))
        assert_equal(np.shape(i),(2,4))

        assert_(isinstance(d[0,0],list))
        assert_(isinstance(i[0,0],list))

class test_vectorization_compiled:
    def setUp(self):
        self.data = np.array([[0,0,0],
                              [0,0,1],
                              [0,1,0],
                              [0,1,1],
                              [1,0,0],
                              [1,0,1],
                              [1,1,0],
                              [1,1,1]])
        self.bounds = 1.1 * np.ones(3)
        self.kdtree = PeriodicCKDTree(self.bounds, self.data)

    def test_single_query(self):
        d, i = self.kdtree.query([0,0,0])
        assert_(isinstance(d,float))
        assert_(isinstance(i,int))

    def test_vectorized_query(self):
        d, i = self.kdtree.query(np.zeros((2,4,3)))
        assert_equal(np.shape(d),(2,4))
        assert_equal(np.shape(i),(2,4))

    def test_vectorized_query_noncontiguous_values(self):
        qs = np.random.randn(3,1000).T
        ds, i_s = self.kdtree.query(qs)
        for q, d, i in zip(qs,ds,i_s):
            assert_equal(self.kdtree.query(q),(d,i))


    def test_single_query_multiple_neighbors(self):
        s = 23
        kk = 27*self.kdtree.n+s
        d, i = self.kdtree.query([0,0,0],k=kk)
        assert_equal(np.shape(d),(kk,))
        assert_equal(np.shape(i),(kk,))
        assert_(np.all(~np.isfinite(d[-s:])))
        assert_(np.all(i[-s:]==self.kdtree.n))

    def test_vectorized_query_multiple_neighbors(self):
        s = 23
        kk = 27*self.kdtree.n+s
        d, i = self.kdtree.query(np.zeros((2,4,3)),k=kk)
        assert_equal(np.shape(d),(2,4,kk))
        assert_equal(np.shape(i),(2,4,kk))
        assert_(np.all(~np.isfinite(d[:,:,-s:])))
        assert_(np.all(i[:,:,-s:]==self.kdtree.n))

class ball_consistency:
    def distance(self, x, y, p):
        return minkowski_distance(np.zeros(x.shape), self.pbcs(x - y), p)

    def pbcs(self, x):
        return x - np.where(self.bounds > 0,
                            (np.round(x / self.bounds) * self.bounds), 0.0)

    def test_in_ball(self):
        l = self.T.query_ball_point(self.x, self.d, p=self.p, eps=self.eps)
        for i in l:
            assert_(self.distance(self.data[i],self.x,self.p)<=self.d*(1.+self.eps))

    def test_found_all(self):
        c = np.ones(self.T.n,dtype=np.bool)
        l = self.T.query_ball_point(self.x, self.d, p=self.p, eps=self.eps)
        c[l] = False
        assert_(np.all(self.distance(self.data[c],self.x,self.p)>=self.d/(1.+self.eps)))

class test_random_ball(ball_consistency):

    def setUp(self):
        n = 100
        m = 4
        self.data = np.random.randn(n,m)
        self.bounds = np.ones(m)
        self.T = PeriodicKDTree(self.bounds, self.data,leafsize=2)
        self.x = np.random.randn(m)
        self.p = 2.
        self.eps = 0
        self.d = 0.2

class test_random_ball_compiled(ball_consistency):

    def setUp(self):
        n = 100
        m = 4
        self.data = np.random.randn(n,m)
        self.bounds = np.ones(m)
        self.T = PeriodicCKDTree(self.bounds, self.data,leafsize=2)
        self.x = np.random.randn(m)
        self.p = 2.
        self.eps = 0
        self.d = 0.2

class test_random_ball_approx(test_random_ball):

    def setUp(self):
        test_random_ball.setUp(self)
        self.eps = 0.1

class test_random_ball_approx_compiled(test_random_ball_compiled):

    def setUp(self):
        test_random_ball_compiled.setUp(self)
        self.eps = 0.1

class test_random_ball_far(test_random_ball):

    def setUp(self):
        test_random_ball.setUp(self)
        self.d = 2.

class test_random_ball_far_compiled(test_random_ball_compiled):

    def setUp(self):
        test_random_ball_compiled.setUp(self)
        self.d = 2.

class test_random_ball_l1(test_random_ball):

    def setUp(self):
        test_random_ball.setUp(self)
        self.p = 1

class test_random_ball_l1_compiled(test_random_ball_compiled):

    def setUp(self):
        test_random_ball_compiled.setUp(self)
        self.p = 1

class test_random_ball_linf(test_random_ball):

    def setUp(self):
        test_random_ball.setUp(self)
        self.p = np.inf

class test_random_ball_linf_compiled(test_random_ball_compiled):

    def setUp(self):
        test_random_ball_compiled.setUp(self)
        self.p = np.inf

def test_random_ball_vectorized():

    n = 20
    m = 5
    bounds = np.ones(m)
    T = PeriodicKDTree(bounds, np.random.randn(n,m))

    r = T.query_ball_point(np.random.randn(2,3,m),1)
    assert_equal(r.shape,(2,3))
    assert_(isinstance(r[0,0],list))

def test_random_ball_vectorized_compiled():

    n = 20
    m = 5
    bounds = np.ones(m)
    T = PeriodicCKDTree(bounds, np.random.randn(n,m))

    r = T.query_ball_point(np.random.randn(2,3,m),1)
    assert_equal(r.shape,(2,3))
    assert_(isinstance(r[0,0],list))


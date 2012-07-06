SciPy-based kd-tree with periodic boundary conditions

Copyright 2012 Patrick Varilly.
Released under the scipy license

Rationale
---------

When running and/or analyzing molecular simulations, finding the closest
neighbors of a particles is a very common operation.  Usually, Verlet lists
and/or cell lists are used for this, but these usually work well when the
maximum neighbor distance is fixed and not too large.  kd-trees are the
general answer to the problem of finding nearest neighbors, and SciPy
implements kd-trees quite well.  However, molecular simulations are usually
run in periodic boxes, not open boxes, so the act of finding "close"
neighbors must take the periodic boundary conditions into account.

This module implements two classes, PeriodicKDTree and PeriodicCKDTree,
that work as drop-in replacements for scipy.spatial.KDTree and
scipy.spatial.cKDTree, respectively (the latter of these is a Cython-optimized
version of a kd-tree but with more limited functionality).

For the moment, only the query and query_ball_point methods are implemented.
These should see you through most situations in analyzing molecular
simulations.

Installation
------------

The uwham package relies on NumPy and SciPy to do its work.  Make sure
these are installed (for Mac OS X users, see note below).  Then run the
following command inside the uwham directory:

  python setup.py install --user

The query_ball_point method in PeriodicCKDTree relies on the extension to
cKDTree I recently wrote for SciPy.  See
https://github.com/scipy/scipy/pull/262

Basic usage
-----------

  from periodic_kdtree import PeriodicCKDTree
  import numpy as np

  # Boundaries (0 or negative means open boundaries in that dimension)
  bounds = np.array([30.0, 30.0, -1])   # xy periodic, open along z

  # Points
  n = 10000
  data = 30.0 * np.random.randn(n, 3)

  # Build kd-tree
  T = PeriodicCKDTree(bounds, data)

  # Find 4 closest neighbors to a random point
  # (d[j], i[j]) = distance and index of jth closest point
  d, i = T.query([45.0, 10.0, 10.0], k=4)

  # Find neighbors within a fixed distance of a point
  neighbors = T.query_ball_point([45.0, 10.0, 10.0], r=3.0)

Tests and benchmarks
--------------------

See test_periodic_kdtree.py, benchmark.py and nonperiodic_benchmark.py
(based off of Anne M. Archibald's benchmarks)

Simple queries seem to be substantially slower for now, but ball lookups
aren't that unacceptably slower.

Sample periodic benchmarks (time in seconds)
++++++++++++++++++++++++++++++++++++++++++++

dimension 3, 10000 points
PeriodicKDTree constructed:	0.0793428
PeriodicCKDTree constructed:	0.00343394
PeriodicKDTree 1000 lookups:	14.8779
PeriodicCKDTree 1000 lookups:	1.73712
flat PeriodicCKDTree 1000 lookups:	14.2604
PeriodicKDTree 1000 ball lookups:	39.7997
PeriodicCKDTree 1000 ball lookups:	0.222894
flat PeriodicCKDTree 1000 ball lookups:	1.23763
Ball lookups agree? True

Sample nonperiodic benchmarks
+++++++++++++++++++++++++++++
dimension 3, 10000 points
KDTree constructed:	0.0820239
cKDTree constructed:	0.00158691
KDTree 1000 lookups:	0.561787
cKDTree 1000 lookups:	0.00447702
flat cKDTree 1000 lookups:	0.335597
KDTree 1000 ball lookups:	5.60061
cKDTree 1000 ball lookups:	0.0196991
flat cKDTree 1000 ball lookups:	0.336687
Ball lookups agree? True

Installation in OS X
--------------------

In Mac OS X, the usual automatic downloading of dependencies during a
python setup.py install isn't able to successfully install NumPy and SciPy
(in my machine, it's the lack of a Fortran compiler, but the SciPy docs
point to other potential problems).  So you have to download and install
NumPy and SciPy manually.

The good news: binaries are easily available
The bad news: they only work with the version of Python from
  http://www.python.org, not the version that ships with OS X!

So you have to download and install three things:

1. Python 2.7.2 from http://www.python.org/download
2. Latest version of NumPy from http://numpy.org
3. Latest version of SciPy from http://scipy.org

This is far less painful than it sounds.  In my own case, the disk images
that I ended up downloading were:

1. python-2.7.2-macosx10.6.dmg
2. numpy-1.6.1-py2.7-python.org-macosx10.6.dmg
3. scipy-0.10.0-py2.7-python.org-macosx10.6.dmg

CAREFUL: for some of these packages, the "link to the latest version" that
SourceForge suggests may be incorrect!  Do look at the full list of downloads
available and pick the one that is most appropriate to your own setup

Once you've installed python2.7, numpy and scipy, you can run the command

    python setup.py install --user

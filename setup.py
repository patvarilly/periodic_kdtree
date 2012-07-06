#!/usr/bin/env python
#
# Setup script for periodic_kdtree module.
# Written by Patrick Varilly, 6 Jul 2012

import sys

if sys.version_info[0] >= 3:
    print "WARNING: periodic_kdtree package has not been tested on Python 3"

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Programming Language :: Python
Topic :: Scientific/Engineering

"""

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

# On OS X, the automatic downloading and installation of numpy and scipy
# is very problematic.  Avoid it
if sys.platform == 'darwin':
    try:
        import numpy
        import scipy
    except ImportError:
        print """

FATAL ERROR: NumPy and SciPy both need to be installed to install periodic_kdtree.
Ordinarily, they would be installed automatically, but this seems to be
very problematic in OS X.  Fortunately, there are easy-to-install binaries
available for download.

Please read the README.txt file for installation instructions.

"""
        exit()

setup(name="periodic_kdtree",
      version="1.0",
      author="Patrick Varilly",
      author_email="pv271@cam.ac.uk",
      description="SciPy-based kd-tree with periodic boundary conditions",
      url="http://github.com/patvarilly/periodic_kdtree",
      license='BSD',
      classifiers=[_ for _ in CLASSIFIERS.split('\n') if _],
      platforms=["Linux", "Unix", "Mac OS-X", "Windows", "Solaris"],
      install_requires=[
        'numpy>=1.3.0',
        'scipy>=0.7.0',
        ],
      
      py_modules=['periodic_kdtree'],
      )

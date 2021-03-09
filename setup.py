#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Installation script

Version handling borrowed from pandas (http://pandas.pydata.org).
Pretty much everything else borrowed from geopandas (http://geopandas.org).
"""

# Copyright (C) 2015, Carson Farmer <carsonfarmer@gmail.com>
# All rights reserved. MIT Licensed.

import sys
import os
import warnings

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

PACKAGE_NAME = "streamhist"

LONG_DESCRIPTION = "A streaming approximate histogram based on the algorithm"\
    " found in Ben-Haim & Tom-Tov's Streaming Parallel Decision Tree Algorithm."

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    try:
        import subprocess
        try:
            pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                                    stdout=subprocess.PIPE).stdout
        except OSError:
            # msysgit compatibility
            pipe = subprocess.Popen(
                ["git.cmd", "describe", "HEAD"],
                stdout=subprocess.PIPE).stdout
        rev = pipe.read().strip()
        # Makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        FULLVERSION = '%d.%d.%d.dev-%s' % (MAJOR, MINOR, MICRO, rev)
    except:
        warnings.warn("WARNING: Couldn't get git revision")
else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), PACKAGE_NAME, 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

write_version_py()

setup(name=PACKAGE_NAME,
      version=FULLVERSION,
      description='Streaming approximate histograms in Python.',
      license='MIT',
      author='Carson Farmer',
      author_email='carsonfarmer@gmail.com',
      url='http://carsonfarmer.com/',
      keywords="streaming data histogram data summary",
      long_description=LONG_DESCRIPTION,
      packages=find_packages(".", exclude=["licenses", "docs", "examples"]),
      install_requires=["sortedcontainers"],
      zip_safe=True,
      classifiers=["Development Status :: 2 - Pre-Alpha",
                   "Environment :: Console",
                   "Intended Audience :: Science/Research",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Information Technology",
                   "License :: OSI Approved :: MIT License",
                   "Natural Language :: English",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering :: Information Analysis",
                   "Topic :: System :: Distributed Computing",
                   ]
      )

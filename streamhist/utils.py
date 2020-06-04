#!/usr/bin/env python
"""Some useful utility functions and classes."""

import ctypes as _ctypes
import sys as _sys
from sys import platform as _platform
from math import log, sqrt
import types

if _sys.version_info >= (3, 3):
    from collections.abc import Iterable
else:
    from collections import Iterable

iterator_types = (types.GeneratorType, Iterable)

if _sys.version_info.major >= 3:
    _izip = zip
else:
    from itertools import izip as _izip

try:
    from itertools import accumulate
except ImportError:
    # itertools.accumulate only in Py3.x
    def accumulate(iterable):
        it = iter(iterable)
        total = next(it)
        yield total
        for element in it:
            total += element
            yield total


_E = 2.718281828459045

__all__ = ["next_after", "argmin", "bin_diff", "accumulate"]

if _platform == "linux" or _platform == "linux2":
    _libm = _ctypes.cdll.LoadLibrary('libm.so.6')
    _funcname = 'nextafter'
elif _platform == "darwin":
    _libm = _ctypes.cdll.LoadLibrary('libSystem.dylib')
    _funcname = 'nextafter'
elif _platform == "win32":
    _libm = _ctypes.cdll.LoadLibrary('msvcrt.dll')
    _funcname = '_nextafter'
else:
    # these are the ones I have access to...
    # fill in library and function name for your system math dll
    print("Platform", repr(_platform), "is not supported")
    _sys.exit(0)

_nextafter = getattr(_libm, _funcname)
_nextafter.restype = _ctypes.c_double
_nextafter.argtypes = [_ctypes.c_double, _ctypes.c_double]


def next_after(x, y):
    """Returns the next floating-point number after x in the direction of y."""
    # This implementation comes from here:
    # http://stackoverflow.com/a/6163157/1256988
    return _nextafter(x, y)


def _diff(a, b, weighted):
        diff = b.value - a.value
        if weighted:
            diff *= log(_E + min(a.count, b.count))
        return diff


def bin_diff(array, weighted=False):
    return [_diff(a, b, weighted) for a, b in _izip(array[:-1], array[1:])]


def argmin(array):
    # Turns out Python's min and max functions are super fast!
    # http://lemire.me/blog/archives/2008/12/17/fast-argmax-in-python/
    return array.index(min(array))


def linspace(start, stop, num):
    """Custom version of numpy's linspace to avoid numpy depenency."""
    if num == 1:
        return stop
    h = (stop - start) / float(num)
    values = [start + h * i for i in range(num+1)]
    return values


def roots(a, b, c):
    """Super simple quadratic solver."""
    d = b**2.0 - (4.0 * a * c)
    if d < 0:
        raise(ValueError("This equation has no real solution!"))
    elif d == 0:
        x = (-b + sqrt(d)) / (2.0 * a)
        return (x, x)
    else:
        x1 = (-b + sqrt(d)) / (2.0 * a)
        x2 = (-b - sqrt(d)) / (2.0 * a)
        return (x1, x2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A streaming approximate histogram based on the algorithms found in Ben-Haim &
Tom-Tov's Streaming Parallel Decision Tree Algorithm [1]_. Histogram bins do
not have a preset size. As values stream into the histogram, bins are
dynamically added and merged.

The accurate method of calculating quantiles (like percentiles) requires data
to be sorted. Streaming histograms make it possible to approximate quantiles
without sorting (or even individually storing) values.

A maximum bin size is passed as an argument to the init methods. A larger bin
size yields more accurate approximations at the cost of increased memory
utilization and performance. There is no "optimal" bin count, but somewhere
between 20 and 80 bins should be sufficient.

The historgram class implented here is based on VividCortex's "Streaming
approximate histograms in Go" [2]_, which is released under the MIT Open
Source license. Additional algorithmic adjustments and methods were adapted from
BigLM's "Streaming Histograms for Clojure/Java" [3]_, which is released under
the Apache License, Version 2.0.

References
----------
.. [1] http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf
.. [2] https://github.com/VividCortex/gohistogram
.. [3] https://github.com/bigmlcom/histogram
.. [4] https://vividcortex.com/blog/2013/07/08/streaming-approximate-histograms/
"""

# Copyright © 2015 Carson Farmer <carsonfarmer@gmail.com>
# Copyright © 2013 VividCortex
# All rights reserved. MIT Licensed.
# Copyright © 2013, 2014, 2015 BigML
# Licensed under the Apache License, Version 2.0

from __future__ import print_function

import sys
from bisect import bisect_left

from sortedcontainers import SortedListWithKey
from utils import (next_after, bin_diff, accumulate, linspace,
                   iterator_types, argmin, bin_sums, roots)

_all__ = ["StreamHist", "Bin"]


class StreamHist(object):
    """A StreamHist implementation."""

    def __init__(self, maxbins=64, weighted=False, freeze=None):
        """Create a Histogram with a max of n bins."""
        super(StreamHist, self).__init__()
        # self.bins = []
        self.bins = SortedListWithKey(key=lambda b: b.value)
        self.maxbins = maxbins  # A useful property
        self.total = 0
        self.weighted = weighted
        self._min = None   # A useful property
        self._max = None   # A useful property
        self.freeze = freeze
        self.missing_count = 0

    def update(self, n, count=1):
        """Add a point to the histogram."""
        if n is None:
            # We simply keep a count of the number of missing values
            self.missing_count += count
            return self
        if isinstance(n, iterator_types):
            # Shortcut for updating a histogram with an iterable
            # This works for anything that supports iteration, including
            # file-like objects and readers
            # This also means that nested lists (and similar structures) will
            # be 'unpacked' and added to the histogram 'automatically'
            for p in n:
                self.update(p, count)  # Count is assumed to apply for all
        else:
            self.insert(n, count)
        return self.trim()

    def insert(self, n, count):
        """Inserts a point to the histogram.

        This method implements Steps 1-4 from Algorithm 1 (Update) in ref [1].

        Notes
        -----
        It is better to use `update` when inserting data into the histogram,
        as `insert` does not automatically update the total point count, or
        call `trim` after the insertion. For large batches of inserts, insert
        may be more efficient, but you are responsible for updating counts
        and trimming the bins 'manually'.

        Examples
        --------
        >>> # Using insert
        >>> h = StreamHist().insert(1).insert(2).insert(3)
        >>> h.update_total(3)
        >>> h.trim()

        >>> # Using update
        >>> h = StreamHist().update([1, 2, 3])
        """
        self.update_total(count)
        if self._min is None or self._min > n:
            self._min = n
        if self._max is None or self._max < n:
            self._max = n
        b = Bin(value=n, count=count)
        if b in self.bins:
            index = self.bins.index(b)
            self.bins[index].count += count
        else:
            if self.freeze is not None and self.total >= self.freeze:
                index = self.bins.bisect(Bin(n, count))
                if index:
                    prev_dist = n - self.bins[index-1].value
                else:
                    prev_dist = sys.float_info.max
                if index and index < len(self.bins):
                    next_dist = self.bins[index].value - n
                else:
                    next_dist = sys.float_info.max
                if prev_dist < next_dist:
                    self.bins[index-1].count += count
                else:
                    self.bins[index].count += count
            else:
                self.bins.add(b)

    def cdf(self, x):
        """Return the value of the cumulative distribution function at x."""
        return self.sum(x) / self.total

    def pdf(self, x):
        """Return the value of the probability density function at x."""
        return self.density(x) / self.total

    def bounds(self):
        """Return the upper (max( and lower (min) bounds of the distribution."""
        if len(self):
            return (self._min, self._max)
        return (None, None)

    def count(self):
        """Return the number of bins in this histogram."""
        return self.total

    def median(self):
        """Return a median for the points inserted into the histogram.

        This will be the true median whenever the histogram has less than
        the maximum number of bins, otherwise it will be an approximation.
        """
        if self.total == 0:
            return None
        if len(self.bins) >= self.maxbins:
            # Return the approximate median
            return self.quantiles(0.5)[0]
        else:
            # Return the 'exact' median when possible
            mid = (self.total)/2
            if self.total % 2 == 0:
                return (self.bins[mid-1] + self.bins[mid]).value
            return self.bins[mid].value

    def mean(self):
        """Return the sample mean of the distribution."""
        if self.total == 0:
            return None
        s = 0.0  # Sum
        for b in self.bins:
            s += b.value * b.count
        return s / float(self.total)

    def var(self):
        """Return the variance of the distribution."""
        if self.total < 2:
            return None
        s = 0.0
        m = self.mean()  # Mean
        for b in self.bins:
            s += (b.count * (b.value - m)**2)
        return s / float(self.total)

    def min(self):
        """Return the minimum value in the histogram."""
        return self._min

    def max(self):
        """Return the maximum value in the histogram."""
        return self._max

    def trim(self):
        """Merge adjacent bins to decrease bin count to the maximum value.

        This method implements Steps 5-6 from Algorithm 1 (Update) in ref [1].
        """
        while len(self.bins) > self.maxbins:
            index = argmin(bin_diff(self.bins, self.weighted))
            bin = self.bins.pop(index)
            bin += self.bins.pop(index)
            self.bins.add(bin)
        return self

    def scale_down(self, exclude):
        pass  # By default, we do nothing

    def __str__(self):
        """Return a string reprentation of the histogram."""
        if len(self.bins):
            string = "Mean\tCount\n----\t-----\n"
            for b in self.bins:
                string += "%d\t%i\n" % (b.value, b.count)
            string += "----\t-----\n"
            string += "Missing values: %s\n" % self.missing_count
            string += "Total count: %s" % self.total
            return string
        return "Empty histogram"

    def to_dict(self):
        """Return a dictionary representation of the histogram."""
        bins = list()
        for b in self.bins:
            bins.append({"mean": b.value, "count": b.count})
        info = dict(missing_count=self.missing_count,
                    maxbins=self.maxbins,
                    weighted=self.weighted,
                    freeze=self.freeze)
        return dict(bins=bins, info=info)

    @classmethod
    def from_dict(cls, d):
        """Create a StreaHist object from a dictionary representation.

        The dictionary must be in the format given my `to_dict`. This class
        method, combined with the `to_dict` instance method, can facilitate
        communicating StreamHist objects across processes or networks.
        """
        info = d["info"]
        bins = d["bins"]
        hist = cls(info["maxbins"], info["weighted"], info["freeze"])
        hist.missing_count = info["missing_count"]
        for b in bins:
            count = b["count"]
            value = b["mean"]
            hist.bins.add(Bin(value, count))
        return hist

    def __len__(self):
        """Return the number of bins in this histogram."""
        return len(self.bins)

    def update_total(self, size=1):
        """Update the internally-stored total number of points."""
        self.total += size

    def __add__(self, other):
        """Merge two StreamHist objects into one."""
        res = self.copy()
        return res.merge(other)

    def __iadd__(self, other):
        """Merge another StreamHist object into this one."""
        return self.merge(other)

    def __radd__(self, other):
        """Reverse merge two objects.

        This is useful for merging a list of histograms via sum or similar.
        """
        return self + other

    def merge(self, other, size=None):
        """Merge another StreamHist object into this one.

        This method implements Algorithm 2 (Merge) in ref [1].
        """
        if other == 0:   # Probably using sum here...
            return self  # This is a little hacky...
        for b in other.bins:
            self.bins.add(b)
        self.total += other.total
        if size is not None:
            self.maxbins = size
        self.trim()
        if self._min is None:
            self._min = other._min
        else:
            if other._min is not None:
                self._min = min(self._min, other._min)
        if self._max is None:
            self._max = other._max
        else:
            if other._max is not None:
                self._max = max(self._max, other._max)
        self.missing_count += other.missing_count
        return self

    def copy(self):
        """Make a deep copy of this histogram."""
        res = type(self)(int(self.maxbins), bool(self.weighted))
        res.bins = self.bins.copy()
        res._min = float(self._min) if self._min is not None else None
        res._max = float(self._max) if self._max is not None else None
        res.total = int(self.total)
        res.missing_count = int(self.missing_count)
        res.freeze = int(self.freeze) if self.freeze is not None else None
        return res

    def describe(self, quantiles=[0.25, 0.50, 0.75]):
        """Generate various summary statistics."""
        data = [self.count(), self.mean(), self.var(), self.min()]
        data += self.quantiles(*quantiles) + [self.max()]
        names = ["count", "mean", "var", "min"]
        names += ["%i%%" % round(q*100., 0) for q in quantiles] + ["max"]
        return dict(zip(names, data))

    def compute_breaks(self, n=50):
        """Return output like that of numpy.histogram."""
        last = 0.0
        counts = []
        bounds = linspace(*self.bounds(), num=n)
        for e in bounds[1:]:
            new = self.sum(e)
            counts.append(new-last)
            last = new
        return counts, bounds

    def print_breaks(self, num=50):
        """Print a string reprentation of the histogram."""
        string = ""
        for c, b in zip(*self.compute_breaks(num)):
            bar = str()
            for i in range(int(c/float(self.total)*200)):
                bar += "."
            string += str(b) + "\t" + bar + "\n"
        print(string)

    def sum(self, x):
        """Return the estimated number of points in the interval [−∞, b]."""
        x = float(x)
        if x < self._min:
            ss = 0.0  # Sum is zero!
        elif x >= self._max:
            ss = float(self.total)
        elif x == self.bins[-1].value:
            # Shortcut for when i == max bin (see Steps 3-6)
            last = self.bins[-1]
            ss = float(self.total) - (float(last.count) / 2.0)
        # elif x <= self.bins[0].value:
        #     # Shortcut for when i == min bin (see Steps 3-6)
        #     first = self.bins[0]
        #     ss = float(first.count) / 2.0
        else:
            bin_i = self.floor(x)
            if bin_i is None:
                bin_i = Bin(value=self._min, count=0)
            bin_i1 = self.higher(x)
            if bin_i1 is None:
                bin_i1 = Bin(value=self._max, count=0)
            if bin_i.value == self._min:
                prev_sum = self.bins[0].count / 2.0
            else:
                temp = bin_sums(self.bins, less=x)
                if len(temp):
                    prev_sum = sum(temp)
                else:
                    prev_sum = 0.0
            ss = _compute_sum(x, bin_i, bin_i1, prev_sum)
        return ss

    def density(self, p):
        p = float(p)
        if p < self._min or p > self._max:
            dd = 0.0
        elif p == self._min and p == self._max:
            dd = float('inf')
        elif Bin(value=p, count=0) in self.bins:
            high = next_after(p, float("inf"))
            low = next_after(p, -float("inf"))
            dd = (self.density(low) + self.density(high)) / 2.0
        else:
            bin_i = self.lower(p)
            if bin_i is None:
                bin_i = Bin(value=self._min, count=0)
            bin_i1 = self.higher(p)
            if bin_i1 is None:
                bin_i1 = Bin(value=self._max, count=0)
            dd = _compute_density(p, bin_i, bin_i1)
        return dd

    def quantiles(self, *quantiles):
        """Return the estimated data value for the given quantile(s).

        The requested quantile(s) must be between 0 and 1. Note that even if a
        single quantile is input, a list is always returned.
        """
        temp = bin_sums(self.bins)
        sums = list(accumulate(temp))
        result = []
        for x in quantiles:
            target_sum = x * self.total
            if x <= 0:
                qq = self._min
            elif x >= self.total:
                qq = self._max
            else:
                index = bisect_left(sums, target_sum)
                bin_i = self.bins[index]
                if index < len(sums):
                    bin_i1 = self.bins[index+1]
                else:
                    bin_i1 = self.bins[index]
                if index:
                    prev_sum = sums[index-1]
                else:
                    prev_sum = 0.0
                qq = _compute_quantile(target_sum, bin_i, bin_i1, prev_sum+1)
            result.append(qq)
        return result

    def floor(self, p):
        hbin = Bin(p, 0)
        index = self.bins.bisect_left(hbin)
        if hbin not in self.bins:
            index -= 1
        return self.bins[index] if index >= 0 else None

    def ceiling(self, p):
        hbin = Bin(p, 0)
        index = self.bins.bisect_right(hbin)
        if hbin in self.bins:
            index -= 1
        return self.bins[index] if index < len(self.bins) else None

    def lower(self, p):
        index = self.bins.bisect_left(Bin(p, 0)) - 1
        return self.bins[index] if index >= 0 else None

    def higher(self, p):
        index = self.bins.bisect_right(Bin(p, 0))
        return self.bins[index] if index < len(self.bins) else None


# Utility functions (should not be included in __all__)


def _compute_density(p, bin_i, bin_i1):
    """Finding the density starting from the sum.

    s = p + (1/2 + r - r^2/2)*i + r^2/2*i1
    r = (x - m) / (m1 - m)
    s_dx = i - (i1 - i) * (x - m) / (m1 - m)
    """
    b_diff = p - bin_i.value
    p_diff = bin_i1.value - bin_i.value
    bp_ratio = b_diff / p_diff

    inner = (bin_i1.count - bin_i.count) * bp_ratio
    return (bin_i.count + inner) * (1.0 / (bin_i1.value - bin_i.value))


def _compute_quantile(x, bin_i, bin_i1, prev_sum):
    d = x - prev_sum
    a = bin_i1.count - bin_i.count
    if a == 0:
        offset = d / ((bin_i.count + bin_i1.count) / 2.0)
        u = bin_i.value + (offset * (bin_i1.value - bin_i.value))
    else:
        b = 2.0 * bin_i.count
        c = -2.0 * d
        z = _find_z(a, b, c)
        u = (bin_i.value + (bin_i1.value - bin_i.value) * z)
    return u


def _compute_sum(x, bin_i, bin_i1, prev_sum):
    b_diff = x - bin_i.value
    p_diff = bin_i1.value - bin_i.value
    bp_ratio = b_diff / p_diff

    i1Term = 0.5 * bp_ratio**2.0
    iTerm = bp_ratio - i1Term

    first = prev_sum + bin_i.count * iTerm
    ss = first + bin_i1.count * i1Term
    return ss


def _find_z(a, b, c):
    result_root = None
    candidate_roots = roots(a, b, c)
    for candidate_root in candidate_roots:
        if candidate_root >= 0 and candidate_root <= 1:
            result_root = candidate_root
            break
    return result_root


class Bin(object):
    """Histogram bin object.

    This class implements a simple (value, count) histogram bin pair with
    several added features such as the ability to merge two bins, comparison
    methods, and the ability to export and import from dictionaries . The Bin
    class should be used in conjunction with the StreamHist.
    """
    __slots__ = ['value', 'count']

    def __init__(self, value, count=1):
        """Create a Bin with a given mean and count.

        Parameters
        ----------
        value : float
            The mean of the bin.
        count : int (default=1)
            The number of points in this bin. It is assumed that there are
            `count` points surrounding `value`, of which `count/2` points are
            to the left and `count/2` points are to the right.
        """
        super(Bin, self).__init__()
        self.value = value
        self.count = count

    @classmethod
    def from_dict(cls, d):
        """Create a bin instance from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary must at a minimum a `mean` or `value` key. In
            addition, it may contain a `count` key which contains the number
            of points in the bin.
        """
        value = d.get("mean", d.get("value", None), None)
        if value is None:
            raise ValueError("Dictionary must contain a mean or value key.")
        return cls(value=value, count=d.get("count", 1))

    def __getitem__(self, index):
        """Alternative method for getting the bin's mean and count.

        Parameters
        ----------
        index : int
            The index must be either 0 or 1, where 0 gets the mean (value),
            and 1 gets the count.
        """
        if index == 0:
            return self.value
        elif index == 1:
            return self.count
        raise IndexError("Invalid index (must be 0 or 1).")

    def __repr__(self):
        """Simple representation of a histogram bin.

        Returns
        -------
        Bin(value=`value`, count=`count`) where value and count are the bin's
        stored mean and count.
        """
        return "Bin(value=%d, count=%d)" % (self.value, self.count)

    def __iter__(self):
        """Iterator over the mean and count of this bin."""
        yield ("mean", self.value)
        yield ("count", self.count)

    def __str__(self):
        """String representation of a histogram bin."""
        return str(dict(self))

    def __eq__(self, obj):
        """Tests for equality of two bins.

        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value == obj.value

    def __lt__(self, obj):
        """Tests if this bin has a lower mean than another bin.

        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value < obj.value

    def __gt__(self, obj):
        """Tests if this bin has a higher mean than another bin.

        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value > obj.value

    def __add__(self, obj):
        """Merge this bin with another bin and return the result.

        This method implements Step 7 from Algorithm 1 (Update) in ref [1].

        Parameters
        ----------
        obj : Bin
            The bin that will be merged with this bin.
        """
        count = float(self.count + obj.count)  # Summed heights
        if count:
            # Weighted average
            value = (self.value*float(self.count) + obj.value*float(obj.count))
            value /= count
        else:
            value = 0.0
        return Bin(value=value, count=int(count))

    def __iadd__(self, obj):
        """Merge another bin into this one.

        Parameters
        ----------
        obj : Bin
            The bin that will be merged into this bin.
        """
        out = self + obj
        self = out
        return self

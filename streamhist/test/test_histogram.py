#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""StreamHist testing module.

Most tests in this module are ported/adapted from the Clojure tests developed
for BigMl's "Streaming Histograms for Clojure/Java" [1].

References
----------
[1] https://github.com/bigmlcom/histogram
"""

# Copyright © 2015 Carson Farmer <carsonfarmer@gmail.com>
# Copyright © 2013, 2014, 2015 BigML
# Licensed under the Apache License, Version 2.0


import random
import operator
from builtins import range

import pytest
from pytest import approx

from streamhist import StreamHist

try:
    from functools import reduce
except ImportError:
    pass  # Can just use reduce now


def make_uniform(size):
    return [random.random() for _ in range(size)]


def make_normal(size):
    return [random.normalvariate(0.0, 1.0) for _ in range(size)]


def rand_int(big):
    return random.randint(0, big)


def about(v1, v2, epsilon):
    return abs(float(v1) - float(v2)) <= epsilon


def test_update_vs_insert():
    points = 1000
    data = make_normal(points)
    h1 = StreamHist(maxbins=50)
    h1.update(data)
    h2 = StreamHist(maxbins=50)
    for i, p in enumerate(data):
        h2.insert(p, 1)
        h2.trim()
    h2.trim()

    assert h1.to_dict() == h2.to_dict()


def test_cdf_pdf():
    points = 10000
    h = StreamHist()
    data = make_normal(points)
    h.update(data)
    assert about(h.sum(0), points/2.0, points/50.0)


def test_bounds():
    points = range(15)
    h = StreamHist(maxbins=8)
    h.update(points)
    assert h.bounds() == (0, 14)

    h = StreamHist()
    assert h.bounds() == (None, None)


def test_count():
    points = 15
    h = StreamHist().update(make_normal(points))
    assert h.count() == h.total == points


def test_median_mean():
    points = 10000
    h = StreamHist()
    for p in make_uniform(points):
        h.update(p)
    assert about(h.median(), 0.5, 0.05)

    h = StreamHist()
    for p in make_normal(points):
        h.update(p)
    assert about(h.median(), 0, 0.05)
    assert about(h.mean(), 0, 0.05)


def test_exact_median():
    points = range(15)  # Odd number of points
    h = StreamHist(maxbins=17)
    h.update(points)
    assert h.median() == 7

    points = range(16)  # Even number of points
    h = StreamHist(maxbins=17)
    h.update(points)
    assert h.median() == 7.5


def test_mean():
    points = 1001
    h = StreamHist()
    for p in range(points):
        h.update(p)
    assert h.mean() == (points-1)/2.0


def test_var():
    assert StreamHist().update(1).var() is None
    h = StreamHist()
    for p in [1, 1, 2, 3, 4, 5, 6, 6]:
        h.update(p)
    assert h.var() == 3.75
    h = StreamHist()
    for p in make_normal(10000):
        h.update(p)
    assert about(h.var(), 1, 0.05)


def test_min_max():
    h = StreamHist()
    assert h.min() is None
    assert h.max() is None

    for _ in range(1000):
        h.update(rand_int(10))

    assert h.min() == 0
    assert h.max() == 10

    h1 = StreamHist()
    h2 = StreamHist()
    for p in range(4):
        h1.update(p)
        h2.update(p+2)
    merged = h1.merge(h2)

    assert merged.min() == 0
    assert merged.max() == 5


def test_trim():
    points = 1000
    h = StreamHist(maxbins=10)
    for _ in range(points):
        h.update(rand_int(10))
    assert len(h.bins) == 10 and h.total == points

    h = StreamHist(maxbins=10)
    for _ in range(points):
        h.insert(rand_int(10), 1)
        h.trim()
    assert len(h.bins) == 10 and h.total == points


def test_string():
    h = StreamHist(maxbins=5)
    assert str(h) == "Empty histogram"

    h.update(range(5))
    string = "Mean\tCount\n----\t-----\n"
    string += "0\t1\n1\t1\n2\t1\n3\t1\n4\t1"
    string += "\n----\t-----\nMissing values: 0\nTotal count: 5"
    assert str(h) == string


def test_round_trip():
    # Tests to_dict and from_dict
    h = StreamHist().update([1, 1, 4])
    assert h.to_dict() == h.from_dict(h.to_dict()).to_dict()


def test_len():
    h = StreamHist(maxbins=5)
    assert len(h) == 0
    h.update(range(5))
    assert len(h) == len(h.bins) == 5
    h.update(range(5))
    assert len(h) == len(h.bins) == 5


def test_update_total():
    h = StreamHist(maxbins=5)
    h.update(range(5))
    assert h.total == h.count() == 5
    h.update(range(5))
    assert h.total == h.count() == 10


def test_merge():
    assert len(StreamHist().merge(StreamHist()).bins) == 0
    assert len(StreamHist().merge(StreamHist().update(1)).bins) == 1
    assert len(StreamHist().update(1).merge(StreamHist()).bins) == 1

    points = 1000
    count = 10
    hists = []
    for c in range(count):
        h = StreamHist()
        for p in make_normal(points):
            h.update(p)
        hists.append(h)
    merged = reduce(lambda a, b: a.merge(b), hists)
    assert about(merged.sum(0), (points*count)/2.0, (points*count)/50.0)

    h1 = StreamHist().update(1).update(None)
    h2 = StreamHist().update(2).update(None)
    merged = h1.merge(h2)
    assert merged.total == 2


def test_copy():
    h1 = StreamHist()
    h2 = h1.copy()
    assert h1.bins == h2.bins
    h1.update(make_normal(1000))
    assert h1.bins != h2.bins
    h2 = h1.copy()
    assert h1.bins == h2.bins
    h1 = StreamHist().update([p for p in range(4)])
    h2 = h1.copy()
    assert h1.to_dict() == h2.to_dict()


def test_describe():
    points = 10000
    data = make_uniform(points)
    h = StreamHist().update(data)
    d = h.describe(quantiles=[0.5])
    print(d)
    assert about(d["50%"], 0.5, 0.05)
    assert about(d["min"], 0.0, 0.05)
    assert about(d["max"], 1.0, 0.05)
    assert about(d["mean"], 0.5, 0.05)
    assert about(d["var"], 0.08, 0.05)
    assert d["count"] == points


def test_compute_breaks():
    points = 10000
    bins = 25
    from numpy import histogram, allclose
    data = make_normal(points)
    h1 = StreamHist().update(data)
    h2, es2 = histogram(data, bins=bins)
    h3, es3 = h1.compute_breaks(bins)

    assert allclose(es2, es3)
    assert allclose(h2, h3, rtol=1, atol=points/(bins**2))


def test_sum():
    points = 10000
    h = StreamHist()
    data = make_normal(points)
    h.update(data)
    assert about(h.sum(0), points/2.0, points/50.0)


def test_paper_example():
    """Test Appendix A example from Ben-Haim paper."""
    from numpy import allclose
    h = StreamHist(maxbins=5)
    h.update((23,19,10,16,36,2,9))
    assert allclose(
        [(bin.value, bin.count) for bin in h.bins],
        [(2,1), (9.5,2), (17.5,2), (23,1), (36,1)])
    h2 = StreamHist(maxbins=5)
    h2.update((32,30,45))
    h3 = h + h2
    assert allclose(
        [(bin.value, bin.count) for bin in h3.bins],
        [(2,1), (9.5,2), (19.33,3), (32.67,3), (45,1)],
        rtol=1e-3)
    assert about(h3.sum(15), 3.275, 1e-3)


def test_sum_first_half_of_first_bin():
    # test sum at point between min and first bin value
    # https://github.com/carsonfarmer/streamhist/issues/13
    h = StreamHist(maxbins=5)
    h.update((1, 2, 3, 4, 5, .5))
    assert h.min() == 0.5
    bin0 = h.bins[0]
    assert bin0.value == 0.75
    assert bin0.count == 2
    assert h.sum(h.min()) == 0
    assert h.sum((h.min() + bin0.value) / 2) == (.5 ** 2) * bin0.count / 2


def test_quantiles():
    points = 10000
    h = StreamHist()
    for p in make_uniform(points):
        h.update(p)
    assert about(h.quantiles(0.5)[0], 0.5, 0.05)

    h = StreamHist()
    for p in make_normal(points):
        h.update(p)
    a, b, c = h.quantiles(0.25, 0.5, 0.75)
    assert about(a, -0.66, 0.05)
    assert about(b, 0.00, 0.05)
    assert about(c, 0.66, 0.05)


def test_histogram_exact():
    """A StreamHist which is not at capacity matches numpy statistics"""
    max_bins = 50
    points = [random.expovariate(1/5) for _ in range(max_bins)]
    h = StreamHist(max_bins)
    h.update(points)

    q = [i / 100 for i in range(101)]
    import numpy as np
    assert h.quantiles(*q) == approx(np.quantile(points, q))
    assert h.mean() == approx(np.mean(points))
    assert h.var() == approx(np.var(points))
    assert h.min() == min(points)
    assert h.max() == max(points)
    assert h.count() == max_bins


@pytest.mark.parametrize("max_bins,num_points,expected_error", [
    (50, 50, 1e-6),
    (100, 150, 1.5),
    (100, 1000, 1),
    (250, 1000, .5),
])
def test_histogram_approx(max_bins, num_points, expected_error):
    """Test accuracy of StreamHist over capacity, especially quantiles."""
    points = [random.expovariate(1/5) for _ in range(num_points)]
    h = StreamHist(max_bins)
    h.update(points)

    import numpy as np
    q = [i / 100 for i in range(101)]
    err_sum = 0  # avg percent error across samples
    for p, b, b_np, b_np_min, b_np_max in zip(
            q,
            h.quantiles(*q),
            np.quantile(points, q),
            np.quantile(points, [0] * 7 + q),
            np.quantile(points, q[7:] + [1] * 7)):
        err_denom = b_np_max - b_np_min
        err_sum += abs(b - b_np) / err_denom
    assert err_sum <= expected_error
    assert h.mean() == approx(np.mean(points))
    assert h.var() == approx(np.var(points), rel=.05)
    assert h.min() == min(points)
    assert h.max() == max(points)
    assert h.count() == num_points


def test_density():
    h = StreamHist()
    for p in [1., 2., 2., 3.]:
        h.update(p)
    assert about(0.0, h.density(0.0), 1e-10)
    assert about(0.0, h.density(0.5), 1e-10)
    assert about(0.5, h.density(1.0), 1e-10)
    assert about(1.5, h.density(1.5), 1e-10)
    assert about(2.0, h.density(2.0), 1e-10)
    assert about(1.5, h.density(2.5), 1e-10)
    assert about(0.5, h.density(3.0), 1e-10)
    assert about(0.0, h.density(3.5), 1e-10)
    assert about(0.0, h.density(4.0), 1e-10)


def test_weighted_gap():
    """
    Histograms using weighted gaps are less eager to merge bins with large
    counts. This test builds weighted and non-weighted histograms using samples
    from a normal distribution. The non-weighted histogram should spend more of
    its bins capturing the tails of the distribution. With that in mind this
    test makes sure the bins bracketing the weighted histogram have larger
    counts than the bins bracketing the non-weighted histogram.
    """
    points = 10000
    h1 = StreamHist(maxbins=32, weighted=True)
    h2 = StreamHist(maxbins=32, weighted=False)
    for p in make_normal(points):
        h1.update(p)
        h2.update(p)
    wt = h1.bins
    nm = h2.bins

    assert wt[0].count + wt[-1].count > nm[0].count + nm[-1].count


def test_hist():
    assert StreamHist() is not None


def test_negative_densities():
    points = 10000
    h = StreamHist()
    data = make_normal(points)
    h.update(data)

    from numpy import linspace
    x = linspace(h.min(), h.max(), 100)
    assert all([h.pdf(t) >= 0. for t in x])


def test_weighted():
    data = [1, 2, 2, 3, 4]
    h = StreamHist(maxbins=3, weighted=True)
    for p in data:
        h.update(p)
    assert h.total == len(data)


def test_missing():
    data = [1, None, 1, 4, 6]
    h = StreamHist(maxbins=2)
    for p in data:
        h.update(p)
    assert h.missing_count == 1
    assert len(h.bins) == 2
    assert h.bins[0][0] == 1 and h.bins[1][0] == 5


def test_missing_merge():
    h1 = StreamHist(maxbins=8).update(None)
    h2 = StreamHist(maxbins=8)
    assert h1.merge(h2) is not None

    h1 = StreamHist().update(None)
    h2 = StreamHist().update(None)
    merged = StreamHist().merge(h1.merge(h2))

    assert merged.missing_count == 2


def test_negative_zero():
    assert len(StreamHist().update(0.0).update(-0.0).bins) == 1


def test_freeze():
    points = 100000
    h = StreamHist(freeze=500)
    for p in make_normal(points):
        h.update(p)
    assert about(h.sum(0), points/2.0, points/50.0)
    assert about(h.median(), 0, 0.05)
    assert about(h.mean(), 0, 0.05)
    assert about(h.var(), 1, 0.05)


def test_counts():
    data = [605, 760, 610, 615, 605, 780, 605, 905]
    h = StreamHist(maxbins=4, weighted=False)
    for p in data:
        h.update(p)
    counts = [b[1] for b in h.bins]
    assert len(data) == reduce(operator.add, counts) == h.total


def test_multi_merge():
    points = 100000
    data = make_uniform(points)
    samples = [data[x:x+100] for x in range(0, len(data), 100)]
    hists = [StreamHist().update(s) for s in samples]
    h1 = sum(hists)
    h2 = StreamHist().update(data)

    q1 = h1.quantiles(.1, .2, .3, .4, .5, .6, .7, .8, .9)
    q2 = h2.quantiles(.1, .2, .3, .4, .5, .6, .7, .8, .9)
    from numpy import allclose
    assert allclose(q1, q2, rtol=1, atol=0.025)


def test_exception():
    with pytest.raises(TypeError):
        StreamHist().sum(5)
        StreamHist().update(4).sum(None)


def test_point_density_at_zero():
    h = StreamHist().update(-1).update(0).update(1)
    assert h.density(0) == 1

    h = StreamHist().update(0)
    assert h.density(0) == float("inf")


def test_sum_edges():
    h = StreamHist().update(0).update(10)
    assert h.sum(5) == 1
    assert h.sum(0) == 0.5
    assert h.sum(10) == 2


def test_iterable():
    h = StreamHist().update([p for p in range(4)])
    assert h.total == 4
    nested = [[1, 2, 3], 4, [5, 6], 7, 8, [9], [10, 11, 12], 13, 14, 15]
    h = StreamHist().update(nested)
    assert h.total == 15
    assert h.mean() == 8


# def test_print_counts():
#     # This is a dummy test to test printing things...
#     points = 10000
#     bins = 25
#     data = make_normal(points)
#     h = StreamHist().update(data)
#     h.print_counts(bins)
#     assert True


# def test_median_with_data():
#     h = StreamHist()
#     with open("test_data.csv", "r") as f:
#         points = 0
#         for row in f:
#             h.update(float(row.replace("\n", "")))
#             points += 1
#
#     assert about(h.median(), 0.0, 0.05)

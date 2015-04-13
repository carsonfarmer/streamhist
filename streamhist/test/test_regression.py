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

from streamhist import StreamHist


def make_normal(size):
    return [random.normalvariate(0.0, 1.0) for _ in range(size)]


def test_regression():
    random.seed(1700)
    data = make_normal(10000)
    hist1 = StreamHist(maxbins=5)
    hist2 = StreamHist(maxbins=5, weighted=True)
    # hist3 = StreamHist(maxbins=5, weighted=True)
    hist4 = StreamHist(maxbins=5)

    hist1.update(data)
    hist2.update(data)
    hist3 = hist2 + hist1
    hist4.update(range(10000))

    reg = [{'count': 1176.0, 'mean': -1.622498097884402},
           {'count': 5290.0, 'mean': -0.3390892100898127},
           {'count': 3497.0, 'mean': 1.0310297400593385},
           {'count': 35.0, 'mean': 2.2157182954841126},
           {'count': 2.0, 'mean': 3.563619987633774}]
    assert hist1.to_dict()["bins"] == reg

    reg = [-1.022649473089556, -0.5279748744244142, 0.1476067074922296,
           0.9815338358189885, 1.6627248917927795]
    assert hist1.quantiles(0.1, 0.25, 0.5, 0.75, 0.9) == reg

    reg = [{'count': 579.0, 'mean': -2.017257931684027},
           {'count': 1902.0, 'mean': -1.0677091300958608},
           {'count': 3061.0, 'mean': -0.24660751313691653},
           {'count': 2986.0, 'mean': 0.5523120572161528},
           {'count': 1472.0, 'mean': 1.557598912751095}]
    assert hist2.to_dict()["bins"] == reg

    reg = [-1.1941285587341846, -0.6041467139342105, 0.08840996549170466,
           0.8247014091807423, 1.557598912751095]
    assert hist2.quantiles(0.1, 0.25, 0.5, 0.75, 0.9) == reg

    reg = [{'count': 1755.0, 'mean': -1.7527351028815432},
           {'count': 1902.0, 'mean': -1.0677091300958608},
           {'count': 8351.0, 'mean': -0.3051906980106826},
           {'count': 6483.0, 'mean': 0.8105375295133331},
           {'count': 1509.0, 'mean': 1.5755221868037264}]
    assert hist3.to_dict()["bins"] == reg

    reg = [-1.0074328972882012, -0.5037558708214145, 0.11958766584785563,
           0.8874923692642509, 1.432517386448461]
    assert hist3.quantiles(0.1, 0.25, 0.5, 0.75, 0.9) == reg

    reg = [{'count': 1339.0, 'mean': 669.0},
           {'count': 2673.0, 'mean': 2675.0},
           {'count': 1338.0, 'mean': 4680.5},
           {'count': 2672.0, 'mean': 6685.5},
           {'count': 1978.0, 'mean': 9010.5}]
    assert hist4.to_dict()["bins"] == reg

    reg = [1830.581598358843, 3063.70150218845, 5831.110283907479,
           8084.851093080222, 9010.5]
    assert hist4.quantiles(0.1, 0.25, 0.5, 0.75, 0.9) == reg


def test_iris_regression():
    sepal_length = [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8,
                    4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1,
                    4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5.0,
                    5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6,
                    5.3, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2,
                    5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1,
                    6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0,
                    5.4, 6.0, 6.7, 6.3, 5.6, 5.5, 5.5, 6.1, 5.8, 5.0, 5.6, 5.7,
                    5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3,
                    6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0,
                    6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2, 7.4, 7.9,
                    6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8,
                    6.7, 6.7, 6.3, 6.5, 6.2, 5.9]

    h = StreamHist(maxbins=32)
    h.update(sepal_length)

    b = [{'count': 1, 'mean': 4.3}, {'count': 4, 'mean': 4.425000000000001},
         {'count': 4, 'mean': 4.6}, {'count': 7, 'mean': 4.771428571428571},
         {'count': 6, 'mean': 4.8999999999999995},
         {'count': 10, 'mean': 5.0}, {'count': 9, 'mean': 5.1},
         {'count': 4, 'mean': 5.2}, {'count': 1, 'mean': 5.3},
         {'count': 6, 'mean': 5.3999999999999995},
         {'count': 7, 'mean': 5.5},
         {'count': 6, 'mean': 5.6000000000000005},
         {'count': 15, 'mean': 5.746666666666667},
         {'count': 3, 'mean': 5.900000000000001},
         {'count': 6, 'mean': 6.0},
         {'count': 6, 'mean': 6.1000000000000005},
         {'count': 4, 'mean': 6.2}, {'count': 9, 'mean': 6.299999999999999},
         {'count': 7, 'mean': 6.3999999999999995},
         {'count': 5, 'mean': 6.5}, {'count': 2, 'mean': 6.6},
         {'count': 8, 'mean': 6.700000000000001}, {'count': 3, 'mean': 6.8},
         {'count': 4, 'mean': 6.9}, {'count': 1, 'mean': 7.0},
         {'count': 1, 'mean': 7.1}, {'count': 3, 'mean': 7.2},
         {'count': 1, 'mean': 7.3}, {'count': 1, 'mean': 7.4},
         {'count': 1, 'mean': 7.6}, {'count': 4, 'mean': 7.7},
         {'count': 1, 'mean': 7.9}]
    assert h.to_dict()["bins"] == b

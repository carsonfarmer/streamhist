#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2015 Carson Farmer <carsonfarmer@gmail.com>
# Copyright © 2013 VividCortex
# All rights reserved. MIT Licensed.
# Copyright © 2013, 2014, 2015 BigML
# Licensed under the Apache License, Version 2.0

from streamhist import StreamHist

__all__ = ["WeightedHist"]


class WeightedHist(StreamHist):
    """Exponentially weighted streaming histogram.

    WeightedHist implements bin values as exponentially-weighted moving
    averages. This allows the histogram to be used for long periods of time
    with many values without worrying too much about overflows. Moving
    averages also give more recent values more weight (via the alpha
    parameter), which in turn means we can approximate quantiles with recency
    factored in.
    """
    def __init__(self, maxbins=50, weighted=False, alpha=None):
        """Create a histogram with a max of n bins and a decay factor of alpha.

        Alpha should be set to 2 / (N+1), where N represents the average age of
        the moving window. For example, a 60-second window with an average age
        of 30 seconds would yield an alpha of 0.064516129.
        """
        super(StreamHist, self).__init__(maxbins, weighted)
        self.alpha = alpha if alpha is not None else _estimate_alpha(30)

    def scale_down(self, exclude):
        # Not a very optimal way to implement this...
        for i, b in enumerate(self.bins):
            if i != exclude:
                b.count = _ewma(b.count, 0, self.alpha)

    def update_total(self):
        # Not a very optimal way to implement this...
        total = 0
        for b in self.bins:
            total += b.count
        self.total = total


def _ewma(old, new, alpha):
    return new*(1 - alpha) + old*alpha


def _estimate_alpha(N):
    return 2.0 / (N + 1)

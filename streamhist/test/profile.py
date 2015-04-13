import random

from streamhist import StreamHist


def make_normal(size):
    return [random.normalvariate(0.0, 1.0) for _ in range(size)]

points = 10000
data = make_normal(points)
h1 = StreamHist(maxbins=50)
h1.update(data)

# Times (in seconds)
# 1.421 - bins (getter/setter)
# 0.955 - bins (direct access)
# 0.977 - bins (slots)
# 0.824 - bins (slots) w/out numpy
# 0.737 - current version

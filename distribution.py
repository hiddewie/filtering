import math
import random


class Distribution:
    """ Base class for a distribution """

    def expectation(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError


class NormalDistribution(Distribution):
    """ Represents a N(mu, sigma) distribution """

    def __init__(self, mu, sigma):
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def expectation(self):
        return self.mu

    def variance(self):
        return self.sigma ** 2

    def draw(self):
        return random.normalvariate(self.mu, self.sigma)

    def pdf(self, x):
        return math.exp(- (x - self.mu) ** 2 / (2 * self.sigma ** 2)) / (self.sigma * math.sqrt(2 * math.pi))

    def cdf(self, x):
        raise NotImplementedError


class NoiseDistribution(NormalDistribution):
    """ Represents a N(0, sigma) distribution, for generating white noise """

    def __init__(self, sigma):
        super().__init__(0, sigma)

    def cdf(self, x):
        raise NotImplementedError


class DiscreteDistribution(Distribution):
    """ Represents a discrete distribution with w_i given """

    def __init__(self, values):
        # The sum of the weights must be positive
        s = sum(w for (x, w) in values)
        assert s > 0

        # Normalize weights
        for i in range(len(values)):
            values[i][1] /= s

        self.values = values
        # Generate a distribution to draw from
        self.U = UniformDistribution(0, 1)

    def expectation(self):
        return sum(w * x for (x, w) in self.values)

    def variance(self):
        e = self.expectation()
        return sum(w * (x - e) ** 2 for (x, w) in self.values)

    def draw(self):
        u = self.U.draw()
        s = 0
        for (x, w) in self.values:
            if s + w > u:
                return x
            s += w
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError


class UniformDistribution(Distribution):
    """ Represents a uniform U(a, b) distribution with a < b """

    def __init__(self, a, b):
        assert a < b
        self.a = a
        self.b = b

    def expectation(self):
        return (self.b + self.a) / 2

    def variance(self):
        return (self.b - self.a) ** 2 / 12

    def draw(self):
        return random.uniform(self.a, self.b)

    def pdf(self, x):
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        return 0

    def cdf(self, x):
        if x < self.a:
            return 0
        if x <= self.b:
            return (x - self.a) / (self.b - self.a)
        return 1


class StaticDistribution(Distribution):
    """ Represents a distribution where P(X = a) == 1 for a single value of a """

    def __init__(self, q):
        self.q = q

    def expectation(self):
        return self.q

    def variance(self):
        return 0

    def draw(self):
        return self.q

    def pdf(self, x):
        if x == self.q:
            return float('inf')
        return 0

    def cdf(self, x):
        if x < self.q:
            return 0
        return 1

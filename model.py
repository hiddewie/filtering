
from distribution import *


class FilterModel:
    def __init__(self, X0: Distribution, V: NoiseDistribution, W: NoiseDistribution, f, F, h, H):
        self.xs = []
        self.ys = []
        self.k = 0

        self.X0 = X0
        self.f = f
        self.F = F
        self.h = h
        self.H = H

        # Generate x0
        self.x = X0.draw()
        self.y = None

        self.V = V
        self.W = W

        # Use the same truth
        self.sameTruth = False

    def generate(self):
        """
        Given self.x = x_k, generate x_{k+1} = x_k + v_k and y_k = x_k + w_k
        """
        if self.sameTruth:
            random.seed(len(self.xs))
        self.x = self.f(self.x) + self.V.draw()
        if self.sameTruth:
            random.seed()
        self.y = self.h(self.x) + self.W.draw()

        self.xs.append(self.x)
        self.ys.append(self.y)

        self.k += 1

        return self.y

    @property
    def name(self):
        return 'Filter model'


class LinearFilterModel(FilterModel):
    def __init__(self, X0: Distribution, V: NoiseDistribution, W: NoiseDistribution, F, H):
        super().__init__(X0, V, W, lambda x: F * x, lambda _: F, lambda x: H * x, lambda _: H)

    @property
    def name(self):
        return 'Linear filter model'

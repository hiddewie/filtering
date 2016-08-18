from model import *


class Filter:
    """ Base abstract filter class """

    def __init__(self):
        """ Default constructor """
        self.k = 0
        self.xs = []
        self.mses = []

    def update(self, y):
        """ Process a new observation """
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()


class ExtendedKalmanFilter(Filter):
    def __init__(self, model: FilterModel):
        super().__init__()

        self.x = model.X0.expectation()
        self.sig = model.X0.variance()

        self.Q = model.V.variance()
        self.R = model.W.variance()
        self.f = model.f
        self.F = model.F
        self.h = model.h
        self.H = model.H

    def update(self, y):
        self.x = self.f(self.x)
        self.sig = self.F(self.x) * self.sig * self.F(self.x) + self.Q

        # Find the S and K
        S = self.H(self.x) * self.sig * self.H(self.x) + self.R
        K = self.sig * self.H(self.x) / S

        # Update the X and Sigma
        self.x += K * (y - self.h(self.x))
        self.sig *= 1 - K * self.H(self.x)

        # Add the X and MSE to the history
        self.xs.append(self.x)
        self.mses.append(self.sig)

        # Increase number of observations
        self.k += 1

    @property
    def name(self):
        return 'Extended Kalman'


class KalmanFilter(ExtendedKalmanFilter):
    """ Basic linear Kalman filter """

    def __init__(self, model: LinearFilterModel):
        super().__init__(model)

    @property
    def name(self):
        return 'Kalman'


class ParticleFilter(Filter):
    def __init__(self, model: FilterModel, N: int, resampleThreshold: float):
        # Inititialize filter
        super().__init__()

        # Store data
        self.f = model.f
        self.h = model.h
        self.V = model.V
        self.W = model.W

        self.N = N
        self.particleHistory = []
        self.particles = []

        # Generate particles
        self.X0 = model.X0
        for i in range(N):
            self.particles.append([self.X0.draw(), 1 / N])

        # Store resample threshold
        self.resampleThreshold = resampleThreshold

        # Calculate initial expected x
        self.x = sum(x * w for (x, w) in self.particles)

    def update(self, y):
        # Update particles
        for i in range(self.N):
            self.particles[i][0] = self.f(self.particles[i][0]) + self.V.draw()

        # Update weights
        totalweight = 0
        for i in range(self.N):
            self.particles[i][1] *= self.W.pdf(y - self.h(self.particles[i][0]))
            totalweight += self.particles[i][1]

        assert totalweight > 0

        # Normalize weights
        for i in range(self.N):
            self.particles[i][1] /= totalweight

        # Save a particle history
        self.particleHistory.append(self.particles[:])

        # Calculate new expected x
        self.x = sum(x * w for (x, w) in self.particles)
        self.xs.append(self.x)

        # Calculate MSE
        self.mses.append(sum(w * (x - self.x) ** 2 for (x, w) in self.particles))

        # Optional resampling, if the number of effective particles is below the threshold
        if self.effectiveparticles() / self.N < self.resampleThreshold:
            self.resample()

        self.k += 1

    def effectiveparticles(self):
        """ Find the number of effective particles """
        return 1 / sum(w ** 2 for (x, w) in self.particles)

    def resample(self):
        """ Generate new samples """
        discrete = DiscreteDistribution(self.particles)
        newparticles = []
        for i in range(self.N):
            newparticles.append([discrete.draw(), 1 / self.N])
        self.particles = newparticles

    @property
    def name(self):
        return 'Particle (%d)' % (self.N,)

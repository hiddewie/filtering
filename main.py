
import matplotlib.pyplot as plt
from filter import *
from model import *


def generatedata(model: FilterModel, n):
    for i in range(model.k):
        yield model.ys[i]

    while model.k < n:
        yield model.generate()


def filterdata(filter: Filter, model: FilterModel, n):
    for data in generatedata(model, n):
        filter.update(data)


def plot(model: FilterModel, filters: [Filter]):
    fig = plt.figure()

    fig.suptitle('Value of generating model and filters')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(model.xs)
    plt.xlabel('Time step')
    plt.ylabel('State value')
    for filter in filters:
        ax.plot(filter.xs)
        if isinstance(filter, ParticleFilter):
            index = 0
            for particles in filter.particleHistory:
                ax.plot([index] * len(particles), [x for (x, w) in particles], 'b,')
                index += 1
    ax.legend([model.name] + [filter.name for filter in filters])

    fig = plt.figure()
    fig.suptitle('Real and expected error of filters')

    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('Time step')
    plt.ylabel('State value')
    for filter in filters:
        data = [(filter.xs[i] - model.xs[i]) ** 2 for i in range(len(model.xs))]
        ax.plot(data)
        ax.plot(filter.mses)

    legend = []
    for filter in filters:
        legend.append('Actual error of ' + filter.name)
        legend.append('MSE of ' + filter.name)
    ax.legend(legend)

    plt.show()


def simulate(n: int, model: FilterModel, N: int, resampleThreshold: float):
    assert 0 <= resampleThreshold <= 1

    filters = [ExtendedKalmanFilter(model),
               ParticleFilter(model, N, resampleThreshold)]

    for filter in filters:
        filterdata(filter, model, n)

    plot(model, filters)


if __name__ == '__main__':
    n = 100
    T = 1

    # ---
    # Linear model, x := x + v, y = x + w
    # ---
    model = LinearFilterModel(NormalDistribution(10, 1), NoiseDistribution(1), NoiseDistribution(1), 1, 1)
    # Uncomment the line below to use the same truth
    #model.sameTruth = True
    simulate(n, model, 100, T)

    # ---
    # Nonlinear model, x := sin(x) + v, y = cos(x) + w
    # ---
    model = FilterModel(NormalDistribution(10, 1), NoiseDistribution(1), NoiseDistribution(1), math.sin, math.cos, math.cos, lambda x: -math.sin(x))
    # Uncomment the line below to use the same truth
    #model.sameTruth = True
    # Uncomment the line below to use the model with different parameters
    #model = FilterModel(NormalDistribution(0.5, 1), NoiseDistribution(0.1), NoiseDistribution(0.02), math.sin, math.cos, math.cos, lambda x: -math.sin(x))
    #simulate(n, model, 1000, 1)

    # ---
    # Nonlinear model, x := 10/(1+x^2) + v, y = x + w
    # ---
    model = FilterModel(NormalDistribution(10, 1), NoiseDistribution(1), NoiseDistribution(1), lambda x: 10/(1+x*x), lambda x: -20*x/(x*x+1)**2, lambda x: x, lambda _: 1)
    #simulate(n, model, 100, T)

    # ---
    # Nonlinear model, x := x/(1+x^2) + v, y = x + w
    # ---
    model = FilterModel(NormalDistribution(10, 1), NoiseDistribution(1), NoiseDistribution(1), lambda x: x/(1+x*x), lambda x: (1-x*x)/(x*x+1)**2, lambda x: x, lambda _: 1)
    #simulate(n, model, 100, T)


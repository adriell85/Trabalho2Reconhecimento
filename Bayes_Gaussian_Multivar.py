import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D  # Importação necessária para projeção 3D

def plot_gaussian_distribution(means, covariances, classes, feature_indices=(0, 1), grid_range=(-3, 3), resolution=0.1):
    """
    Plota a distribuição gaussiana multivariada para duas características específicas.

    Parâmetros:
    - means: array de médias por classe.
    - covariances: array de matrizes de covariância por classe.
    - classes: array de identificadores de classe.
    - feature_indices: tupla contendo os índices das duas características a serem plotadas.
    - grid_range: tupla definindo o intervalo da grade para os eixos.
    - resolution: resolução da grade de pontos.
    """
    f1, f2 = feature_indices
    x, y = np.mgrid[grid_range[0]:grid_range[1]:resolution, grid_range[0]:grid_range[1]:resolution]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots()

    for i, c in enumerate(classes):
        mean = means[i][[f1, f2]]
        covariance = covariances[i][[f1, f2], [f1, f2]]
        rv = multivariate_normal(mean, covariance)
        ax.contourf(x, y, rv.pdf(pos), levels=100, cmap='Blues', alpha=0.5)
        ax.set_title(f'Distribuição Gaussiana Multivariada - Características {f1} e {f2}')
        ax.set_xlabel(f'Característica {f1}')
        ax.set_ylabel(f'Característica {f2}')

    plt.show()

def regularize_covariance(covariance, alpha=1e-5):
    regularized_covariance = covariance + alpha * np.eye(covariance.shape[0])
    return regularized_covariance

def plot_gaussian_distribution_3d(means, covariances, classes, feature_indices=(0, 1), grid_range=(-1, 1),
                                  resolution=0.1):
    """
    Plota a distribuição gaussiana multivariada em 3D para duas características específicas.

    Parâmetros:
    - means: array de médias por classe.
    - covariances: array de matrizes de covariância por classe.
    - classes: array de identificadores de classe.
    - feature_indices: tupla contendo os índices das duas características a serem plotadas.
    - grid_range: tupla definindo o intervalo da grade para os eixos.
    - resolution: resolução da grade de pontos.
    """
    f1, f2 = feature_indices
    x, y = np.mgrid[grid_range[0]:grid_range[1]:resolution, grid_range[0]:grid_range[1]:resolution]
    pos = np.dstack((x, y))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, c in enumerate(classes):
        mean = means[i][[f1, f2]]
        covariance = covariances[i][[f1, f2], [f1, f2]]
        # regularized_cov = regularize_covariance(covariance)

        rv = multivariate_normal(mean=mean, cov=covariance, allow_singular=True)
        # Calculando a PDF para os pontos na grade
        z = rv.pdf(pos)

        # Criando uma superfície de plotagem
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.5)
        ax.set_title(f'Distribuição Gaussiana Multivariada - Características {f1} e {f2}')
        ax.set_xlabel(f'Característica {f1}')
        ax.set_ylabel(f'Característica {f2}')
        ax.set_zlabel('Probabilidade')

    plt.show()

def NaiveBayesGaussianMultivar(x_train, y_train, x_test):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = x_train.shape[1]
    priors = np.zeros(n_classes)
    means = np.zeros((n_classes, n_features))
    covariances = np.zeros((n_classes, n_features, n_features))

    for i, c in enumerate(classes):
        x_c = x_train[y_train == c]
        priors[i] = x_c.shape[0] / x_train.shape[0]
        means[i] = np.mean(x_c, axis=0)
        covariances[i] = np.cov(x_c, rowvar=False)

    plot_gaussian_distribution_3d(means, covariances, classes, feature_indices=(0, 1))
    def multivariate_gaussian_pdf(x, mean, covariance):
        det_cov = np.linalg.det(covariance)
        inv_cov = np.linalg.inv(covariance)
        numerator = np.exp(-0.5 * (x - mean) @ inv_cov @ (x - mean).T)
        denominator = np.sqrt((2 * np.pi) ** n_features * det_cov)
        return numerator / denominator

    def classify(x):
        posteriors = np.zeros(n_classes)
        for i in range(n_classes):
            likelihood = multivariate_gaussian_pdf(x, means[i], covariances[i])
            posteriors[i] = likelihood * priors[i]
        return classes[np.argmax(posteriors)]

    predictions = np.array([classify(x) for x in x_test])

    return predictions





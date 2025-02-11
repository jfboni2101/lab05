import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from avengers import main_kmeans_img
from cluster import KMeans
from datasets import gaussians_dataset, two_moon_dataset

plt.ion()


def kmeans_2d(X: np.ndarray, cl: np.ndarray, n_cl: int) -> None:
    """
    Main function to run kmeans the synthetic gaussian dataset.
    """

    # visualize the dataset
    _, ax = plt.subplots(1, 2)

    ax[0].axis('off')
    ax[1].axis('off')

    ax[0].scatter(X[:, 0], X[:, 1], c=cl, s=40)
    plt.waitforbuttonpress(timeout=2.00)

    # solve kmeans optimization
    cl_algo = KMeans(n_cl=n_cl, verbose=True, n_init=5)

    labels = cl_algo.fit_predict(X)

    # visualize results
    ax[1].scatter(X[:, 0], X[:, 1], c=labels, s=40)
    plt.waitforbuttonpress(timeout=2.00)


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('gaussians', 'twomoon', 'avengers'),
                        default='gaussians')
    p = parser.parse_args()

    if p.dataset == 'gaussians':
        X, y = gaussians_dataset(3, [300, 400, 200], [[1, 1], [-4, 6], [6, 6]],
                                 [[2.5, 2.5], [4.5, 4.5], [1.5, 1.5]])
        kmeans_2d(X, y, 3)
    if p.dataset == 'twomoon':
        X, y = two_moon_dataset(n_samples=500, noise=0.1)
        kmeans_2d(X, y, 2)
    if p.dataset == 'avengers':
        main_kmeans_img('img/tony.jpg', 'img/aereo.jpg')
    """


    # Gaussian Dataset
    X, y = gaussians_dataset(3, [300, 400, 200], [[1, 1], [-4, 6], [6, 6]],
                             [[2.5, 2.5], [4.5, 4.5], [1.5, 1.5]])
    kmeans_2d(X, y, 3)


    """
    # TwoMoon Dataset
    X, y = two_moon_dataset(n_samples=500, noise=0.1)
    kmeans_2d(X, y, 2)
    """

    """
    # Avengers Dataset
    main_kmeans_img('img/tony.jpg', 'img/aereo.jpg')
    """
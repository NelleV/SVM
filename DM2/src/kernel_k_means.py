import numpy as np
from matplotlib import pyplot as pl
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import rbf_kernel as gaussian_kernel
from sklearn.decomposition.pca import PCA
from sklearn.cluster import KMeans

import data


def get_kernel_func(kernel):
    """
    """
    if kernel not in ['linear', 'gaussian']:
        raise ValueError("Unknown kernel")

    if kernel == 'linear':
        return linear_kernel

    else:
        return gaussian_kernel


def kernel_k_means(X, k=8, kernel='linear', tol=1e-15, max_iter=30):
    """
    Computes kernel CPA

    Parameters
    ----------
    X: array

    k: int, optional
        number of clusters

    kernel: string, optional
        linear
        gaussian

    tol: float, optional
        tolerance

    max_iter: int, optional
        maximum number of iteration
    """
    kernel_func = get_kernel_func(kernel)
    n_samples, n_features = X.shape

    # Initialisation - done randomly
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)
    seeds = idxs[:k]
    centers = X[seeds]

    for it in range(max_iter):
        XX = kernel_func(X, X)
        dist_X = np.tile(np.diag(XX), (15, 1)).T
        if it == 0:
            old_extra = 0
            dist_centers = np.tile(np.diag(kernel_func(centers, centers)),
                                   (360, 1))
            extra = kernel_func(X, centers)
        else:
            mask = np.tile(np.arange(15), (360, 1))
            elements = np.tile(labels, (15, 1)).T
            mask = (mask == elements).astype('float')
            dist_centers = np.tile((mask * dist_X).sum(axis=0) * \
                                        1. / mask.sum(axis=0),
                                   (360, 1))
            extra = np.dot(XX, mask) * 1. / mask.sum(axis=0)
        distances = dist_X + dist_centers - 2 * extra
        labels = distances.argmin(axis=1)

        inertia = extra.sum()
        print it, inertia
        if ((old_extra - extra)**2).sum() < tol:
            print "finished at iteration %d" % it
            break
        old_extra = extra.copy()

    return labels


if __name__ == "__main__":
    X, Y = data.libras_movement()
    labels = kernel_k_means(X, k=15)

    # Pour representer les donnees, prendre le PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    Xt = pca.transform(X)

    fig = pl.figure()

    colors = ['#334433',
              '#6699aa',
              '#88aaaa',
              '#aacccc',
              '#447799',
              '#225533',
              '#44bbcc',
              '#88dddd',
              '#bbeeff',
              '#0055bb',
              '#220000',
              '#880022',
              '#ff3300',
              '#ffee22',
              '#9988cc'
              ]

    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(15), colors):

        my_members = labels == k
        ax.plot(Xt[my_members, 0], Xt[my_members, 1], 'o',
                markerfacecolor=col, marker='+', markersize=6)
    glabels = kernel_k_means(X, k=15, kernel='gaussian')

    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(15), colors):

        my_members = glabels == k
        ax.plot(Xt[my_members, 0], Xt[my_members, 1], 'o',
                markerfacecolor=col, marker='+', markersize=6)
               
    km = KMeans(k=15)
    km.fit(X)
    ax = fig.add_subplot(1, 3, 3)
    for k, col in zip(range(15), colors):
        my_members = km.labels_ == k
        ax.plot(Xt[my_members, 0], Xt[my_members, 1], 'o',
                markerfacecolor=col, marker='+', markersize=6)

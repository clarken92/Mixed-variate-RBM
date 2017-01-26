import numpy as np
import scipy as sci
from architecture.abstract.data_type import InputType
from utils.preprocess_utils import separate_train_test, standardize
from utils.anomaly_utils import random_noise_on_data


def rand_spm(size, normalize=False):
    m = np.random.randn(size, size)
    m = np.dot(m, m.T)
    if normalize:
        return m/np.linalg.det(m)
    else:
        return m


def one_hot(x, m):
    return np.eye(m, dtype=np.int32)[x]


def generate_toy_data(n_clusters, n_points_per_cluster,
                      gau_dim, bin_dim, cat_dim,
                      n_cat_vals, anomaly_ratio, train_ratio=None):
    print "\nGenerate toy mixed data"
    dim = gau_dim + bin_dim + cat_dim * n_cat_vals

    v_types = []
    v_ranges = []

    gau_range = np.arange(0, gau_dim)
    if len(gau_range) > 0:
        v_types.append(InputType.gaussian)
        v_ranges.append(gau_range)

    bin_range = np.arange(gau_dim, gau_dim + bin_dim)
    if len(bin_range) > 0:
        v_types.append(InputType.binary)
        v_ranges.append(bin_range)

    cat_ranges = []
    cat_ix = gau_dim + bin_dim
    for i in xrange(cat_dim):
        cat_ranges.append(np.arange(cat_ix, cat_ix + n_cat_vals))
        v_types.append(InputType.categorical)
        v_ranges.append(np.arange(cat_ix, cat_ix + n_cat_vals))

        cat_ix += n_cat_vals

    data = None
    label = None

    for i in xrange(n_clusters):
        cov = rand_spm(dim)
        mean = np.random.uniform(low=-5, high=5, size=dim)
        cluster_data = sci.random.multivariate_normal(mean=mean, cov=cov,
                                                      size=n_points_per_cluster)

        # Standardize gaussian and outliers
        cluster_data[:, gau_range] = standardize(cluster_data[:, gau_range])

        # Generate binary values for cluster and outliers
        bin_max = np.max(cluster_data[:, bin_range], axis=0)
        bin_min = np.min(cluster_data[:, bin_range], axis=0)
        p = np.random.uniform(low=0.0, high=1.0, size=bin_dim)
        bin_thresh = bin_min + p * (bin_max - bin_min)
        cluster_data[:, bin_range] = (cluster_data[:, bin_range] > bin_thresh).astype(np.int32)

        # Generate categorical values
        for cat_range in cat_ranges:
            max_ix = np.argmax(cluster_data[:, cat_range], axis=1)
            cluster_data[:, cat_range] = one_hot(max_ix, n_cat_vals)

        if data is None:
            data = cluster_data
            label = i + np.zeros(n_points_per_cluster, dtype=np.int32)
        else:
            data = np.concatenate([data, cluster_data], axis=0)
            label = np.concatenate([label, i + np.zeros(n_points_per_cluster, dtype=np.int32)], axis=0)

    outliers, _ = random_noise_on_data(data, v_types, v_ranges, add_new=True,
                                       noise_ratio=anomaly_ratio, noise_attr_prob=0.7)
    anomaly_label = np.zeros(len(outliers), dtype=np.int32) - 1

    noise_data = np.concatenate([data, outliers], axis=0)
    noise_label = np.concatenate([label, anomaly_label], axis=0)

    rand_indices = np.random.permutation(len(noise_data))

    noise_data = noise_data[rand_indices]
    noise_label = noise_label[rand_indices]

    if train_ratio is not None:
        train_indices, test_indices = separate_train_test(label, train_ratio, shuffle=True)

        return ([noise_data[train_indices], noise_label[train_indices], v_types, v_ranges, anomaly_ratio],
                [noise_data[test_indices], noise_label[test_indices], v_types, v_ranges, anomaly_ratio])
    else:
        return [(noise_data, noise_label, v_types, v_ranges, anomaly_ratio), None]
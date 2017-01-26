import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics

from architecture.abstract.data_type import InputType


# Normal class is a list of number indicate normal number
# noise_ratio is the ratio between anomaly data and all data
def get_threshold(rec_errors, noise_ratio):
    thresh = np.percentile(rec_errors, q=(1 - noise_ratio) * 100, interpolation='midpoint')

    print "\nMax value of neg_log_llh/rec_err is: %.4f" % rec_errors.max()
    print "Min value of neg_log_llh/rec_err is: %.4f" % rec_errors.min()
    print "Threshold value is: %.4f" % thresh

    return thresh


def draw_labeled_error_hist(rec_errors, labels, cmap, thresh=None,
                            save_file=None, title=None, **kwargs):
    fig, ax = plt.subplots()

    classes, counts = np.unique(labels, return_counts=True)

    color_map = {classes[i]: cmap(1.0 * i / len(classes)) for i in xrange(len(classes))}
    sorted_ids = np.argsort(counts)[::-1]

    # Sort to draw class that has most element first
    classes = classes[sorted_ids]

    classes_ids = []
    for i in xrange(len(classes)):
        classes_ids.append(np.arange(len(labels))[labels == classes[i]])

    print "\nGeneral error information of each class:"
    for i in xrange(len(classes)):
        num = len(classes_ids[i])
        min_err = rec_errors[classes_ids[i]].min()
        max_err = rec_errors[classes_ids[i]].max()

        anomaly_num = np.sum((rec_errors[classes_ids[i]] > thresh).astype(np.int32))
        normal_num = num - anomaly_num
        print "Class {} has normal: {} and anomaly: {} over {} instances. Min error: {:.3f} | " \
              "Max error: {:.3f}".format(classes[i], normal_num, anomaly_num, num, min_err, max_err)

    legend_patches = [mpatches.Patch(color=value, label=key)\
                      for key, value in color_map.iteritems()]

    plt.legend(handles=legend_patches, loc='upper right', ncol=2, prop={'size': 12})

    for i in xrange(len(classes)):
        _n, _bins, _patches = ax.hist(x=rec_errors[classes_ids[i]], color=color_map.get(classes[i]), **kwargs)

    if thresh is not None:
        plt.axvline(x=thresh, color='r')

    if title is not None:
        plt.suptitle(title)

    plt.gcf().set_size_inches(16, 8)

    if save_file is not None:
        plt.savefig(save_file, dpi=100)
    else:
        plt.show()


def indices_for_each_class(label):
    if len(label.shape) > 1 and not (np.issubdtype(label.dtype, int) or np.issubdtype(label.dtype, np.object_)):
        raise ValueError("Only vectors of type int or object are supported")

    values = np.unique(label)

    value_map = {values[i]: i for i in xrange(len(values))}
    indices = [None] * len(values)

    for i in xrange(len(label)):
        k = value_map.get(label[i])
        if indices[k] is None:
            indices[k] = [i]
        else:
            indices[k].append(i)
    return values, indices


def unfold_types_ranges(v_types, v_ranges):
    attrs = []
    types = []
    for i in xrange(len(v_types)):
        # Neu la categorical
        if v_types[i] == InputType.categorical:
            attrs.append(v_ranges[i])
            types.append(v_types[i])
        # Neu khong phai categorical
        else:
            attrs.extend(v_ranges[i])
            types.extend([v_types[i]] * len(v_ranges[i]))

    return attrs, types


def anomaly_binary(data):
    return 1 - data


def anomaly_categorical(data):
    # Note that the return array (of multinomial) is 2D so we have to take the first value
    return np.apply_along_axis(lambda x: np.random.multinomial(n=1, pvals=(1.-x)/(data.shape[1]-1.), size=1)[0],
                               axis=1, arr=data)


def anomaly_gaussian(data, mean, std):
    # data + 2.5 std and with different side with means
    return data + 1.*(data-mean)/abs(data-mean) * np.random.uniform(low=2.0, high=3.0, size=len(data)) * std


def anomaly_poisson(data, mean, std):
    noise_data = data + 1.*(data-mean)/abs(data-mean) * np.random.uniform(low=2.0, high=3.0, size=len(data)) * std
    return np.clip(noise_data, a_min=0., a_max=np.inf).astype(dtype=np.int32) #Clip the values to be positive


def random_noise_on_data(data, v_types, v_ranges, noise_ratio=0.1, add_new=True,
                         noise_attr_prob=1.0):

    indices = np.arange(0, data.shape[0])
    noise_count = int(noise_ratio * len(indices))

    np.random.shuffle(indices)
    noise_indices = indices[0: noise_count]

    attrs, types = unfold_types_ranges(v_types, v_ranges)

    noise_attr_count = int(noise_attr_prob * len(attrs))
    # noise_attr_ids = np.arange(len(attrs))[0: noise_attr_count]
    noise_attr_ids = np.sort(np.random.permutation(len(attrs))[0: noise_attr_count])

    noise_data = np.array(data[noise_indices, :], copy=True)

    for id in noise_attr_ids:
        if types[id] == InputType.binary:
            # new_data[noise_indices, attrs[id]] = anomaly_binary(new_data[noise_indices, attrs[id]])
            noise_data[:, attrs[id]] = anomaly_binary(data[noise_indices, attrs[id]])

        elif types[id] == InputType.gaussian:
            mean = np.mean(data[:, attrs[id]], axis=0)
            std = np.std(data[:, attrs[id]], axis=0, ddof=1)
            noise_data[:, attrs[id]] = anomaly_gaussian(data[noise_indices, attrs[id]], mean, std)

        elif types[id] == InputType.poisson:
            mean = np.mean(data[:, attrs[id]], axis=0)
            std = np.std(data[:, attrs[id]], axis=0, ddof=1)
            noise_data[:, attrs[id]] = anomaly_poisson(data[noise_indices, attrs[id]], mean, std)

        elif types[id] == InputType.categorical:
            cols, rows = np.meshgrid(attrs[id], noise_indices, indexing='xy')
            noise_data[:, cols] = anomaly_categorical(data[rows, cols])

    if add_new:
        return noise_data, None
    else:
        return noise_data, noise_indices  # new_label


def separate_normal_anomaly(label, normal_classes, noise_ratio, anomaly_equal=True):
    if len(label.shape) > 1 and not (np.issubdtype(label.dtype, int) or np.issubdtype(label.dtype, np.object_)):
        raise ValueError("Only vectors of type int or object are supported")

    is_normal = np.in1d(label, normal_classes)

    normal_idx = np.where(is_normal)
    anomaly_idx = np.where(np.logical_not(is_normal))

    if type(normal_idx) is tuple:
        normal_idx = normal_idx[0]

    if type(anomaly_idx) is tuple:
        anomaly_idx = anomaly_idx[0]

    # Note that we get all normal data, while only
    # get a small amount of anomaly instance corresponding to noise_ratio (and normal_idx)
    noise_amount = int(noise_ratio / (1.0 - noise_ratio) * len(normal_idx))
    if noise_amount > len(anomaly_idx):
        raise ValueError("The amount of data in other class is not enough for noise ratio {}".format(noise_ratio))

    # Anomaly equally distributed among other classes
    if anomaly_equal:
        # Anomaly class values and count
        a_vals, a_counts = np.unique(label[anomaly_idx], return_counts=True)
        sorted_ids = np.argsort(a_counts)

        a_vals = a_vals[sorted_ids] # Sort anomaly class based on count
        a_counts = a_counts[sorted_ids] # Sort anomaly class based on count
        a_ids_map = {a_vals[i]: [] for i in xrange(len(a_vals))}

        # Update the anomaly id map
        for i in xrange(len(label)):
            a_id = a_ids_map.get(label[i])
            if a_id is not None:
                a_id.append(i)

        for key, item in a_ids_map.iteritems():
            item = np.asarray(item, dtype=np.int32)
            np.random.shuffle(item)

        # Allocate instance of each class for anomaly data
        remain_count = noise_amount
        remain_classes = len(a_vals)
        small_anomaly_idx = []

        for i in xrange(len(a_vals)):
            ave_amount = remain_count * 1.0 / remain_classes
            if a_counts[i] >= ave_amount:
                small_anomaly_idx.append(a_ids_map.get(a_vals[i])[0:int(ave_amount)])
                remain_count -= int(ave_amount)
            else:
                small_anomaly_idx.append(a_ids_map.get(a_vals[i]))
                remain_count -= len(a_ids_map.get(a_vals[i]))
            remain_classes -= 1

        small_anomaly_idx = np.concatenate(small_anomaly_idx, axis=0)
        np.random.shuffle(small_anomaly_idx)

    else:
        np.random.shuffle(anomaly_idx)
        small_anomaly_idx = anomaly_idx[0: int(noise_ratio / (1.0 - noise_ratio) * len(normal_idx))]

    result_ids = np.concatenate([normal_idx, small_anomaly_idx], axis=0)
    # Shuffle se tao ra ti le giua cac class tuong ung voi phan bo
    np.random.shuffle(result_ids)

    return result_ids


def roc_draw(fprs, tprs, save_file=None, **kwargs):
    fig, ax = plt.subplots()

    ax.plot(fprs, tprs, **kwargs)
    ax.set_xlabel('False positive rate', fontsize=36)
    ax.set_ylabel('True positive rate', fontsize=36)

    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    if kwargs.get('title') is not None:
        ax.set_title(kwargs.get('title'), fontsize=36)
        #plt.title(kwargs.get('title'))
    else:
        #plt.title('ROC curve')
        ax.set_title('ROC curve', fontsize=36)
    fig.set_size_inches(15, 10)
    if save_file is not None:
        plt.savefig(save_file, dpi=1000)
    else:
        plt.show()


def compute_auc(normal_classes, test_labels, test_scores, draw_roc=False, save_file=None):
    min_score = np.min(test_scores)
    max_score = np.max(test_scores)
    thresh_range = np.linspace(start=min_score, stop=max_score, num=min(1e4, len(test_scores)),
                               endpoint=True)
    thresh_range = thresh_range[::-1] # Be cause the larger the score, the more likely the anomaly

    tprs = []
    fprs = []

    for thresh in thresh_range:

        test_anomaly = test_scores > thresh

        true_anomaly = np.in1d(test_labels, normal_classes, invert=True)

        true_pos = 1.0 * np.sum(np.logical_and(true_anomaly, test_anomaly).astype(np.int32))

        true_neg = 1.0 * np.sum(np.logical_and(np.logical_not(true_anomaly),
                                               np.logical_not(test_anomaly)).astype(np.int32))

        false_pos = 1.0 * np.sum(np.logical_and(np.logical_not(true_anomaly),
                                                       test_anomaly).astype(np.int32))
        false_neg = 1.0 * np.sum(np.logical_and(true_anomaly,
                                                np.logical_not(test_anomaly)).astype(np.int32))

        tpr = 0.0 if true_pos == 0 else true_pos/(true_pos + false_neg)
        fpr = 0.0 if false_pos == 0 else false_pos/(false_pos + true_neg)

        tprs.append(tpr)
        fprs.append(fpr)

    auc = metrics.auc(fprs, tprs)
    # auc = metrics.auc(fprs, tprs, reorder=True)
    print "\nAUC: {}".format(auc)

    if draw_roc:
        roc_draw(fprs, tprs, save_file=save_file, c='r', ls='-', lw=2.0)

    return auc


# test_labels is a list of true labels of test data
# test_scores a list of scores returned by architecture over test data
# Test score (reconstruction error) cang be thi likelihood la normal cang lon
# thresh is a threshold returned by architecture based on train data
def anomaly_detection(normal_classes, test_labels, test_scores, thresh, print_result=True):
    # test_anomaly = True if instance is anomaly
    # test_anomaly = False if instance is normal
    if thresh is not None:
        test_anomaly = np.asarray(test_scores) > thresh
    else:
        test_anomaly = np.asarray(test_scores, dtype=np.int32) == 1

    true_anomaly = np.in1d(test_labels, normal_classes, invert=True)
    indices = np.arange(len(test_labels))

    TruePos_indices = indices[np.logical_and(true_anomaly, test_anomaly)]
    true_pos = len(TruePos_indices) * 1.0

    TrueNeg_indices = indices[np.logical_and(np.logical_not(true_anomaly),
                                             np.logical_not(test_anomaly))]
    true_neg = len(TrueNeg_indices) * 1.0

    FalsePos_indices = indices[np.logical_and(np.logical_not(true_anomaly), test_anomaly)]
    false_pos = len(FalsePos_indices) * 1.0

    FalseNeg_indices = indices[np.logical_and(true_anomaly,
                                              np.logical_not(test_anomaly))]
    false_neg = len(FalseNeg_indices) * 1.0

    # Compute statistics
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos > 0) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg > 0) else 0.0
    tpr = 0.0 if true_pos == 0 else true_pos / (true_pos + false_neg)
    fpr = 0.0 if false_pos == 0 else false_pos / (false_pos + true_neg)
    if precision == 0. and recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    if print_result:
        print "\nTrue pos (detected as anomaly and truly anomaly) %.4f" % true_pos
        print "True neg (detected as normal and truly normal) %.4f" % true_neg
        print "False pos (detected as anomaly but actually normal) %.4f" % false_pos
        print "False neg (detected as normal but actually anomaly) %.4f" % false_neg

        print "\nAccuracy is: %.4f" % accuracy
        print "Precision is: %.4f" % precision
        print "Recall is: %.4f" % recall
        print "True Positive Rate is: %.4f" % tpr
        print "False Positive Rate is: %.4f" % fpr

        print "F1 is: %.4f" % f1

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
            'tpr': tpr, 'fpr': fpr, 'tp_indices': TruePos_indices, 'tn_indices': TrueNeg_indices,
            'fp_indices': FalsePos_indices, 'fn_indices': FalseNeg_indices}

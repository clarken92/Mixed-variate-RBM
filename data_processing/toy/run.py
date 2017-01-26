import cPickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn import manifold
import theano

from toy_gen import generate_toy_data
from utils.general_utils import plot_2D_embedding
from utils.anomaly_utils import get_threshold, anomaly_detection
from architecture.unsupervised.rbm.mixed_rbm import MixedRBM

from lasagne.updates import momentum

train_file = 'data-train.pkl'
test_file = 'data-test.pkl'

n_clusters = 3
n_points_per_cluster = 1000
gau_dim = 3
bin_dim = 3
cat_dim = 3
n_cat_vals = 3
anomaly_ratio = 0.05
train_ratio = 0.7


([train_data, train_labels, v_types, v_ranges, anomaly_ratio],
 [test_data, test_labels, _, _, _]) = generate_toy_data(n_clusters=n_clusters,
                                                        n_points_per_cluster=n_points_per_cluster,
                                                        gau_dim=gau_dim, bin_dim=bin_dim, cat_dim=cat_dim,
                                                        n_cat_vals=n_cat_vals, anomaly_ratio=anomaly_ratio,
                                                        train_ratio=train_ratio)

with open(train_file, 'wb') as f:
    cPickle.dump((train_data, train_labels, v_types, v_ranges, anomaly_ratio), f)

with open(test_file, 'wb') as f:
    cPickle.dump((test_data, test_labels, v_types, v_ranges, anomaly_ratio), f)


#-----------------------------------------------#
# Plot embedding
print("t-SNE embedding of toy data")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
fitted_train = tsne.fit_transform(train_data)

cmap = plt.cm.get_cmap('hsv')
plot_2D_embedding(fitted_train, train_labels, cmap=cmap)


#-----------------------------------------------#
# Run anomaly detection
train_data = np.asarray(train_data, dtype=theano.config.floatX)
test_data = np.asarray(test_data, dtype=theano.config.floatX)

classes = np.unique(train_labels)
normal_classes = classes[classes != -1]

v_dim = train_data.shape[1]
h_dims = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200]

colors = [cmap(1.0 * i / len(classes)) for i in xrange(len(h_dims))]

n_epochs = 100
batch_size = train_data.shape[0]
lr = 0.01
n_runs = 10

f_scores = []
errs = []

for h_dim in h_dims:
    print "#--------------------------------#"
    print "\nThe number of hidden units is: {}".format(h_dim)

    run_f_scores = []
    for i in xrange(n_runs):
        mixed_rbm = MixedRBM(v_dim=v_dim, h_dim=h_dim,
                             v_types=v_types, v_indices=v_ranges)

        mixed_rbm.set_learning_algor(momentum, momentum=0.8)
        train_errs = mixed_rbm.train(train_x=train_data, valid_x=None, lr=lr,
                        n_epochs=n_epochs, batch_size=batch_size, CD_k=2)

        model_file = ('mixed_rbm-h{}-run_{}.zip'.format(h_dim, i))
        mixed_rbm.save(model_file)

        train_scores = mixed_rbm.score(train_data)
        thresh = get_threshold(train_scores, anomaly_ratio)

        test_scores = mixed_rbm.score(test_data)
        result = anomaly_detection(normal_classes=normal_classes, test_labels=test_labels,
                                   test_scores=test_scores, thresh=thresh, print_result=True)
        run_f_scores.append(result.get('f1'))

    f_scores.append(np.mean(run_f_scores))

h_dims = np.asarray(h_dims)
fig, ax = plt.subplots()

width = 0.8

ax.plot(h_dims, f_scores, c='b', ls='-')
ax.set_xlabel('hidden units', fontsize=18)
ax.set_ylabel('F-measure', fontsize=18)

blue_line = mlines.Line2D([], [], color='blue', label='F-measure')
plt.legend(handles=[blue_line])

plt.show()

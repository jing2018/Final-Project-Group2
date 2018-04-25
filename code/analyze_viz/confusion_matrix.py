import sys
import caffe
import numpy as np
import lmdb
import argparse
from collections import defaultdict


def flat_shape(x):
    "Returns x without singleton dimension, eg: (1,28,28) -> (28,28)"
    return x.reshape(filter(lambda s: s > 1, x.shape))


def lmdb_reader(fpath):
    import lmdb
    lmdb_env = lmdb.open(fpath)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, flat_shape(image), label)


def leveldb_reader(fpath):
    import leveldb
    db = leveldb.LevelDB(fpath)

    for key, value in db.RangeIter():
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, flat_shape(image), label)


def npz_reader(fpath):
    npz = np.load(fpath)

    xs = npz['arr_0']
    ls = npz['arr_1']

    for i, (x, l) in enumerate(np.array([xs, ls]).T):
        yield (i, x, l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--mean', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lmdb', type=str, default=None)
    group.add_argument('--leveldb', type=str, default=None)
    group.add_argument('--npz', type=str, default=None)
    args = parser.parse_args()

    # Extract mean from the mean image file
    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
    f = open(args.mean, 'rb')
    mean_blobproto_new.ParseFromString(f.read())
    mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    f.close()

    count = 0
    correct = 0
    matrix = defaultdict(int)  # (real,pred) -> int
    labels_set = set()

    # CNN reconstruction and loading the trained weights
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()
    if args.lmdb != None:
        reader = lmdb_reader(args.lmdb)
    if args.leveldb != None:
        reader = leveldb_reader(args.leveldb)
    if args.npz != None:
        reader = npz_reader(args.npz)


    actual_labels = []
    predicted_labels = []
    for i, image, label in reader:
        actual_labels.append(label)

        image_caffe = image.reshape(1, *image.shape)
        out = net.forward_all(data=np.asarray([image_caffe]) - mean_image)


        predicted_labels.append(int(out['prob'][0].argmax(axis=0)))
        plabel = int(out['prob'][0].argmax(axis=0))

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import itertools
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt

    classes = ('anger', 'disgust', 'fear', 'happy', 'sad', 'suprised', 'normal')

    #predicted = predicted_labels

    Actual_labels = actual_labels
    predicted_labels = predicted_labels

    Confusion_matrix = confusion_matrix(Actual_labels, predicted_labels)


    def plot_confusion_matrix(cm, classes, normalize, title, cmap=plt.cm.Greens):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Unnormalized Confusion matrix')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black")

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')


    plt.figure(1)
    plot_confusion_matrix(Confusion_matrix, classes=classes, normalize=False,
                          title='Unnormalized Confusion matrix')
    # plt.show()

    plt.figure(2)
    plot_confusion_matrix(Confusion_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()

    #####Calculate the accuracy and misclassification rate.
    aggregate_true = 0
    for i, j in itertools.product(range(Confusion_matrix.shape[0]), range(Confusion_matrix.shape[1])):
        if i == j:
            aggregate_true += Confusion_matrix[i, j]

    Total_test = Confusion_matrix.sum()

    accuracy = float(float(aggregate_true) / float(Total_test))
    print('Accuracy calculated from the confusion matrix: ' + str(100 * accuracy) + '%')

    aggregate_wrong = 0
    for i, j in itertools.product(range(Confusion_matrix.shape[0]), range(Confusion_matrix.shape[1])):
        if i != j:
            aggregate_wrong += Confusion_matrix[i, j]

    misclassification_rate = float(float(aggregate_wrong) / float(Total_test))
    print('Misclassification Rate calculated from the confusion matrix: ' + str(100 * misclassification_rate) + '%')

    # 22


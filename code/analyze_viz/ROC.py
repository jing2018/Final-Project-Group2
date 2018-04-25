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

        predicted_labels.append(out['prob'][0][0:7])
        plabel = int(out['prob'][0].argmax(axis=0))



    # ROC

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt

    classes = ('anger', 'disgust', 'fear', 'happy' , 'sad' , 'suprised','normal')

    actual_test = label_binarize(actual_labels, classes=[0, 1, 2, 3, 4, 5, 6])

    predicted_test = predicted_labels    #label_binarize(predicted_labels, classes=[0, 1, 2, 3, 4, 5, 6])
    n_classes =7

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(actual_test[:, i], predicted_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    plt.figure(1)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=classes[i] + ' - ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating character')
        plt.legend(loc="lower right")

    for i in range(n_classes):
        plt.figure(i + 2)
        plt.plot(fpr[i], tpr[i], label=classes[i] + ' - ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(classes[i] +' - Receiver operating character')  #
        plt.legend(loc="lower right")

    plt.show()



import caffe
import matplotlib.pyplot as plt
import numpy as np
import time

caffe.set_mode_gpu()


solver = caffe.SGDSolver('AlexNet_450000_size_solver.prototxt')

niter = 100000
test_interval = 1000
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
start = time.time()

for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')

    if it % test_interval == 0:
        acc = solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...', 'accuracy:', acc
        test_acc[it // test_interval] = acc

end = time.time() - start
print ('Time is {}.'.format(end))

#----------------------------------------------------------------------------------------------
###########################Plotting Intermediate Layers, Weight################################
#---------------------------------------Define Functions---------------------------------------

def vis_square_f(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an iamage
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data,cmap='Greys',interpolation='nearest');
    plt.axis('off')
#----------------------------------------------------------------------------------------------
#------------------------------Plot Loss and Accuracy------------------------------------------
plt.figure(1)
plt.semilogy(np.arange(niter), train_loss)
plt.xlabel('Number of Iteration')
plt.ylabel('Training Loss Values')
plt.title('Training Loss')

plt.figure(2)
plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy Values')
plt.title('Test Accuracy')

#----------------------------------------------------------------------------------------------
#------------------------------Plot All Feature maps Functions---------------------------------
net = solver.net
f1_0 = net.blobs['conv1'].data[0, :96]
plt.figure(3)
vis_square_f(f1_0)
plt.title('Feature Maps for Conv1')

f2_0 = net.blobs['conv2'].data[0, :256]
plt.figure(4)
vis_square_f(f2_0)
plt.title('Feature Maps for Conv2')

f3_0 = net.blobs['conv3'].data[0, :384]
plt.figure(5)
vis_square_f(f3_0)
plt.title('Feature Maps for Conv3')

f4_0 = net.blobs['conv4'].data[0, :384]
plt.figure(6)
vis_square_f(f4_0)
plt.title('Feature Maps for Conv4')

f5_0 = net.blobs['conv5'].data[0, :256]
plt.figure(7)
vis_square_f(f5_0)
plt.title('Feature Maps for Conv5')

plt.show()
#----------------------------------------------------------------------------------------------
#---------------------------Print Shape ans Sizes for all Layers--------------------------------

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)





import matplotlib
matplotlib.use('Agg')
from model2 import compile_model
import numpy
import math
import sys
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import load_data

numpy.random.seed(7)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def train_model(filenames, batch_size, epochs, file_iterations, train, samples=None, uuid=None):
    model = compile_model()
    test_data = None
    test_labels = None
    loss = []
    acc = []
    for f in range(0, file_iterations):
        print('-- File Iteration -- {}'.format(f + 1))
        for file in filenames:
            print('-- New File -- {}'.format(file))
            shuffled_data_flat, shuffled_one_hot, shuffled_labels = load_data(file)
            if samples is None:
                samples = len(shuffled_one_hot)
            train_count = int(math.floor(samples * train))
            batches = int(math.floor(train_count / batch_size))
            for e in range(0, epochs):
                print('Epoch {}/{}'.format(e+1, epochs))
                for i in range(0, batches - 1):
                    metrics = model.train_on_batch(shuffled_data_flat[i*batch_size:(i+1)*batch_size, :],
                                         shuffled_one_hot[i*batch_size:(i+1)*batch_size])
                    print('{:6d} / {:6d} - {:5s} {:1.4f} - {:5s} {:1.4f}'.format((i+1)*batch_size,
                                                                                 train_count,
                                                                                 model.metrics_names[0], metrics[0],
                                                                                 model.metrics_names[1], metrics[1]))
                    loss.append(metrics[0])
                    acc.append(metrics[1])
                metrics = model.train_on_batch(shuffled_data_flat[(batches-1) * batch_size:train_count, :],
                                            shuffled_one_hot[(batches-1) * batch_size:train_count])
                print('{:6d} / {:6d} - {:5s} {:1.4f} - {:5s} {:1.4f}'.format(train_count,
                                                                            train_count,
                                                                            model.metrics_names[0], metrics[0],
                                                                            model.metrics_names[1], metrics[1]))
                loss.append(metrics[0])
                acc.append(metrics[1])
            if f == 0:
                if test_data is None:
                    test_data = shuffled_data_flat[train_count:samples, :]
                    test_labels = shuffled_one_hot[train_count:samples]
                else:
                    test_data = numpy.concatenate([test_data, shuffled_data_flat[train_count:samples, :]])
                    test_labels = numpy.concatenate([test_labels, shuffled_one_hot[train_count:samples]])
    time = datetime.datetime.now()
    scores = model.evaluate(test_data, test_labels)
    model.save('{date:%Y-%m-%d %H:%M:%S}-{uuid}-{score:1.4f}.h5'.format(uuid=uuid, date=time, score=scores[1] * 100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    plot(loss, acc, time, uuid)


def plot(loss, acc, time, uuid):
    plt.plot(loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('{date:%Y-%m-%d %H:%M:%S}-loss-{uuid}.png'.format(uuid=uuid, date=time))
    plt.clf()
    plt.plot(acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Batch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('{date:%Y-%m-%d %H:%M:%S}-acc-{uuid}.png'.format(uuid=uuid, date=time))


if __name__ == '__main__':
    files = ['rf_data/training_data_chunk_0.pkl',
             'rf_data/training_data_chunk_1.pkl',
             'rf_data/training_data_chunk_2.pkl',
             'rf_data/training_data_chunk_3.pkl',
             'rf_data/training_data_chunk_4.pkl',
             'rf_data/training_data_chunk_5.pkl',
             'rf_data/training_data_chunk_6.pkl',
             'rf_data/training_data_chunk_7.pkl',
             'rf_data/training_data_chunk_8.pkl',
             'rf_data/training_data_chunk_9.pkl',
             'rf_data/training_data_chunk_10.pkl',
             'rf_data/training_data_chunk_11.pkl',
             'rf_data/training_data_chunk_12.pkl',
             'rf_data/training_data_chunk_13.pkl',
             'rf_data/training_data_chunk_14.pkl',
             ]

    # files = ['rf_data/training_data_chunk_0.pkl', 'rf_data/training_data_chunk_1.pkl']
    if len(sys.argv) > 1:
        uuid = sys.argv[1]
    else:
        uuid = 'model'
    train_model(files, 64, 4, 1, .8, uuid=uuid)

import matplotlib
matplotlib.use('Agg')
from model2 import compile_model
import numpy
import math
import sys
import datetime
import gc
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import load_data, load_data_lstm, load_data_conv

numpy.random.seed(12)


def train_model(filenames, train_names, batch_size, epochs, file_iterations, train_count=None, uuid=None, load=False):
    model, scaler = compile_model()
    loss = []
    acc = []
    ev = []
    k = []
    train_shuffled_data_flat = None
    train_shuffled_one_hot = None
    temp_data, temp_one_hot = None, None
    model_file = None
    training_file_name = 'training.temp'
    loss_array_fname = 'loss.temp'
    acc_array_fname = 'acc.temp'
    ev_array_fname = 'ev.temp'
    k_array_fname = 'k.temp'
    e_start = 0
    e_end = epochs
    f_start = 0
    f_end = file_iterations
    if train_count is None:
        num_of_vals = 288000
    else:
        num_of_vals = train_count
    time = datetime.datetime.now()
    scores = (0, 0, 0)
    if os.path.isfile(training_file_name):
        with open(training_file_name, 'r') as f:
            tmp = f.readline()
            e_start, e_end, f_start, f_end, model_file_tmp, uuid = tmp.split(',')
        loss = numpy.load(loss_array_fname + '.npy').tolist()
        acc = numpy.load(acc_array_fname + '.npy').tolist()
        ev = numpy.load(ev_array_fname + '.npy').tolist()
        k = numpy.load(k_array_fname + '.npy').tolist()
        model.load_weights(model_file_tmp)
    if load:
        shuffled_data_flat = numpy.zeros((num_of_vals * len(filenames), 2048), dtype=numpy.float32)
        shuffled_one_hot = numpy.zeros((num_of_vals * len(filenames), 24), dtype=numpy.float32)
        count = 0
        print('Loading on startup...')
        for f in filenames:
            print('Loading: ' + f)
            temp_data, temp_one_hot = load_data(f, scaler)
            if train_count is not None:
                temp_data = temp_data[:train_count, :]
                temp_one_hot = temp_one_hot[:train_count, :]
            for i in range(temp_data.shape[0]):
                shuffled_data_flat[i + count * num_of_vals] = numpy.asarray(temp_data[i], dtype=numpy.float32)
                shuffled_one_hot[i + count * num_of_vals] = numpy.asarray(temp_one_hot[i], dtype=numpy.float32)
            count += 1
            gc.collect()
        filenames = ['Loaded_On_Startup']
    for f in train_names:
        print('Loading: ' + f)
        if train_shuffled_data_flat is None:
            train_shuffled_data_flat, train_shuffled_one_hot = load_data(f, scaler)
        else:
            temp_data, temp_one_hot = load_data(f, scaler)
            train_shuffled_data_flat = numpy.concatenate((train_shuffled_data_flat, temp_data), axis=0)
            train_shuffled_one_hot = numpy.concatenate((train_shuffled_one_hot, temp_one_hot), axis=0)
    if train_count is not None:
        train_shuffled_data_flat = train_shuffled_data_flat[:train_count, :]
        train_shuffled_one_hot = train_shuffled_one_hot[:train_count, :]
    del temp_data, temp_one_hot
    for f in range(int(f_start), int(f_end)):
        print('-- File Iteration {} --'.format(f + 1))
        for file in filenames:
            print('-- New File {} --'.format(file))
            if not load:
                shuffled_data_flat, shuffled_one_hot = load_data(file, scaler)
            if load:
                train_count = len(shuffled_one_hot)
            batches = int(math.floor(train_count / batch_size))
            for e in range(int(e_start), int(e_end)):
                print('Epoch {}/{}'.format(e+1, epochs))
                for i in range(0, batches - 1):
                    metrics = model.train_on_batch(shuffled_data_flat[i*batch_size:(i+1)*batch_size, :],
                                                   shuffled_one_hot[i*batch_size:(i+1)*batch_size])
                    print_metrics((i+1)*batch_size, train_count, model, metrics, scores, e)
                    loss.append(metrics[0])
                    acc.append(metrics[1])
                    k.append(metrics[2])
                metrics = model.train_on_batch(shuffled_data_flat[(batches-1) * batch_size:train_count, :],
                                               shuffled_one_hot[(batches-1) * batch_size:train_count])
                print_metrics(train_count, train_count, model, metrics, scores, e+1)
                loss.append(metrics[0])
                acc.append(metrics[1])
                k.append(metrics[2])
                scores = model.evaluate(train_shuffled_data_flat, train_shuffled_one_hot)
                ev.append(list(scores))
                print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
                if model_file is not None:
                    os.remove(model_file)
                model_file = '{date:%Y-%m-%d %H:%M:%S}-{uuid}-{score:1.4f}.h5'.format(uuid=uuid, date=time,
                                                                                    score=scores[1] * 100)
                model.save(model_file)
                with open(training_file_name, 'w') as file:
                    file.write("{},{},{},{},{},{}".format(e+1, e_end, f, f_end, model_file, uuid))
                numpy.save(loss_array_fname, loss)
                numpy.save(acc_array_fname, acc)
                numpy.save(ev_array_fname, ev)
                numpy.save(k_array_fname, k)
                print('Current progress saved')
            e_start = 0
    scores = model.evaluate(train_shuffled_data_flat, train_shuffled_one_hot)
    model.save('{date:%Y-%m-%d %H:%M:%S}-{uuid}-{score:1.4f}.h5'.format(uuid=uuid, date=time, score=scores[1] * 100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    os.remove(training_file_name)
    plot(loss, acc, numpy.asarray(ev), k, time, uuid)


def print_metrics(step, total, model, metrics, scores, e):
    print(('{:6d} / {:6d} - {:5s} {:1.4f} - {:5s} {:1.4f} - {:5s} {:1.4f}' +
           ' - {:7s} {:1.4f} - {:7s} {:1.4f} - {:8s} {:1.4f} - epoch {}').format(step,
                                                                                 total,
                                                                                 model.metrics_names[0], metrics[0],
                                                                                 model.metrics_names[1], metrics[1],
                                                                                 model.metrics_names[2], metrics[2],
                                                                                 'lst los', scores[0],
                                                                                 'lst acc', scores[1],
                                                                                 'lst top_2', scores[2],
                                                                                 e))


def train_lstm(filenames, train_names, batch_size, epochs, file_iterations, train_count=None, uuid=None, evaluate=True):
    model = compile_model()
    loss = []
    acc = []
    ev = []
    k = []
    for f in range(0, file_iterations):
        print('-- File Iteration -- {}'.format(f + 1))
        for file in filenames:
            print('-- New File -- {}'.format(file))
            shuffled_data_flat, shuffled_one_hot = load_data_lstm(file)
            if train_count is None:
                train_count = len(shuffled_one_hot)
            batches = int(math.floor(train_count / batch_size))
            for e in range(0, epochs):
                print('Epoch {}/{}'.format(e+1, epochs))
                for i in range(0, batches - 1):
                    metrics = model.train_on_batch(shuffled_data_flat[i * batch_size:(i + 1) * batch_size, :],
                                                   shuffled_one_hot[i * batch_size:(i + 1) * batch_size])
                    print_metrics((i + 1) * batch_size, train_count, model, metrics)
                    loss.append(metrics[0])
                    acc.append(metrics[1])
                    k.append(metrics[2])
                metrics = model.train_on_batch(shuffled_data_flat[(batches - 1) * batch_size:train_count, :],
                                               shuffled_one_hot[(batches - 1) * batch_size:train_count])
                print_metrics(train_count, train_count, model, metrics)
                loss.append(metrics[0])
                acc.append(metrics[1])
                k.append(metrics[2])
            if evaluate:
                shuffled_data_flat, shuffled_one_hot = load_data_lstm(train_names[0])
                scores = model.evaluate(shuffled_data_flat, shuffled_one_hot)
                ev.append(list(scores))
                print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
            del shuffled_data_flat
            del shuffled_one_hot
    time = datetime.datetime.now()
    shuffled_data_flat, shuffled_one_hot = load_data_lstm(train_names[0])
    scores = model.evaluate(shuffled_data_flat, shuffled_one_hot)
    model.save('{date:%Y-%m-%d %H:%M:%S}-{uuid}-{score:1.4f}.h5'.format(uuid=uuid, date=time, score=scores[1] * 100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    plot(loss, acc, ev, k, time, uuid)


def train_conv(filenames, train_names, batch_size, epochs, file_iterations, train_count=None, uuid=None, evaluate=True):
    model = compile_model()
    loss = []
    acc = []
    ev = []
    k = []
    for f in range(0, file_iterations):
        print('-- File Iteration -- {}'.format(f + 1))
        for file in filenames:
            print('-- New File -- {}'.format(file))
            shuffled_data_flat, shuffled_one_hot = load_data_conv(file)
            if train_count is None:
                train_count = len(shuffled_one_hot)
            batches = int(math.floor(train_count / batch_size))
            for e in range(0, epochs):
                print('Epoch {}/{}'.format(e+1, epochs))
                for i in range(0, batches - 1):
                    metrics = model.train_on_batch(shuffled_data_flat[i*batch_size:(i+1)*batch_size, :],
                                                   shuffled_one_hot[i*batch_size:(i+1)*batch_size])
                    print('{:6d} / {:6d} - {:5s} {:1.4f} - {:5s} {:1.4f} - {:5s} {:1.4f}'.format((i+1)*batch_size,
                                                                                 train_count,
                                                                                 model.metrics_names[0], metrics[0],
                                                                                 model.metrics_names[1], metrics[1],
                                                                                 model.metrics_names[2], metrics[2]))
                    loss.append(metrics[0])
                    acc.append(metrics[1])
                    k.append(metrics[2])
                metrics = model.train_on_batch(shuffled_data_flat[(batches-1) * batch_size:train_count, :],
                                               shuffled_one_hot[(batches-1) * batch_size:train_count])
                print('{:6d} / {:6d} - {:5s} {:1.4f} - {:5s} {:1.4f} - {:5s} {:1.4f}'.format(train_count,
                                                                             train_count,
                                                                             model.metrics_names[0], metrics[0],
                                                                             model.metrics_names[1], metrics[1],
                                                                             model.metrics_names[2], metrics[2]))
                loss.append(metrics[0])
                acc.append(metrics[1])
                k.append(metrics[2])
            if evaluate:
                shuffled_data_flat, shuffled_one_hot = load_data_conv(train_names[0])
                scores = model.evaluate(shuffled_data_flat, shuffled_one_hot)
                ev.append(scores)
                print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
            del shuffled_data_flat
            del shuffled_one_hot
    time = datetime.datetime.now()
    shuffled_data_flat, shuffled_one_hot = load_data_conv(train_names[0])
    scores = model.evaluate(shuffled_data_flat, shuffled_one_hot)
    model.save('{date:%Y-%m-%d %H:%M:%S}-{uuid}-{score:1.4f}.h5'.format(uuid=uuid, date=time, score=scores[1] * 100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    plot(loss, acc, ev, k, time, uuid)


def plot(loss, acc, ev, k, time, uuid):
    plt.plot(loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('{date:%Y-%m-%d %H:%M:%S}-loss-{uuid}.png'.format(uuid=uuid, date=time))
    plt.clf()
    plt.plot(acc)
    plt.plot(k)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Batch')
    plt.legend(['train', 'top_2'], loc='upper left')
    plt.savefig('{date:%Y-%m-%d %H:%M:%S}-acc-{uuid}.png'.format(uuid=uuid, date=time))
    plt.clf()
    plt.plot(ev[:, 1])
    plt.plot(ev[:, 2])
    plt.title('Model Evaluation')
    plt.ylabel('Accuracy')
    plt.xlabel('Batch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('{date:%Y-%m-%d %H:%M:%S}-eval-{uuid}.png'.format(uuid=uuid, date=time))


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
             'rf_data/training_data_chunk_10.pkl',
             'rf_data/training_data_chunk_11.pkl',
             'rf_data/training_data_chunk_12.pkl',
             'rf_data/training_data_chunk_13.pkl',
             ]
    train_names = [
        'rf_data/training_data_chunk_14.pkl',
    ]
    # files = ['rf_data/training_data_chunk_0.pkl', 'rf_data/training_data_chunk_1.pkl']
    if len(sys.argv) > 1:
        uuid = sys.argv[1]
    else:
        uuid = 'model'
    # train_lstm(files, train_names, 512, 1, 1, uuid=uuid, evaluate=False, train_count=100000)
    # train_conv(files, train_names, 512, 1, 1, uuid=uuid, evaluate=False, train_count=100000)
    train_model(files, train_names, 512, 1, 4, uuid=uuid, load=True)

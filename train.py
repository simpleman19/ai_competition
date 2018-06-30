import matplotlib
matplotlib.use('Agg')
from model_sub import compile_model, load_data
from load_data import shuffle_in_place
import numpy
import math
import sys
import datetime
import gc
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

numpy.random.seed(12)


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self) :
        for f in self.files:
            f.flush()


def train_model(filenames, train_names, batch_size, epochs, file_iterations, loader, train_count=None, uuid=None, load=False, model_file=None, save=False):
    # Variable initialization
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    session = None
    model, scaler = compile_model()
    loss = []
    acc = []
    ev = []
    k = []
    class_rep = []
    train_shuffled_data_flat = None
    train_shuffled_one_hot = None
    temp_data, temp_one_hot = None, None
    shuffled_data_flat, shuffled_one_hot = None, None
    training_file_name = 'training.temp'
    loss_array_fname = 'loss.temp'
    acc_array_fname = 'acc.temp'
    ev_array_fname = 'ev.temp'
    k_array_fname = 'k.temp'
    class_array_fname = 'class.temp'
    e_start = 0
    e_end = epochs
    f_start = 0
    f_end = file_iterations

    # Determine how many vals to use
    if train_count is None:
        num_of_vals = 288000
    else:
        num_of_vals = train_count

    # Get time for timestamp
    time = datetime.datetime.now()
    # Seed scores
    scores = (0, 0, 0)

    # Check if should load a existing model to continue training
    if os.path.isfile(training_file_name):
        with open(training_file_name, 'r') as f:
            tmp = f.readline()
            e_start, e_end, f_start, f_end, model_file_tmp, uuid = tmp.split(',')
        loss = numpy.load(loss_array_fname + '.npy').tolist()
        acc = numpy.load(acc_array_fname + '.npy').tolist()
        ev = numpy.load(ev_array_fname + '.npy').tolist()
        k = numpy.load(k_array_fname + '.npy').tolist()
        class_rep = numpy.load(class_array_fname + 'npy').tolist()
        model.load_weights(model_file_tmp)
    elif model_file is not None:  # If not continueing but loading model
        model.load_weights(model_file)

    # If preload all data flag is set else load data at each iteration
    if load:
        count = 0
        print('Loading on startup...')
        for f in filenames:
            print('Loading: ' + f)
            sys.stdout.flush()
            temp_data, temp_one_hot, labels = loader(f, scaler)
            if train_count is not None:
                temp_data = temp_data[:train_count, :]
                temp_one_hot = temp_one_hot[:train_count, :]
            if shuffled_data_flat is None:
                shape = list(temp_data.shape)
                shape[0] = shape[0] * len(filenames)
                shape = tuple(shape)
                shape_one_hot = list(temp_one_hot.shape)
                shape_one_hot[0] = shape_one_hot[0] * len(filenames)
                shape_one_hot = tuple(shape_one_hot)
                shuffled_data_flat = numpy.zeros(shape, dtype=numpy.float16)
                shuffled_one_hot = numpy.zeros(shape_one_hot, dtype=numpy.float16)
            for i in range(temp_data.shape[0]):
                shuffled_data_flat[i + count * num_of_vals] = numpy.asarray(temp_data[i], dtype=numpy.float16)
                shuffled_one_hot[i + count * num_of_vals] = numpy.asarray(temp_one_hot[i], dtype=numpy.float16)
            count += 1
            gc.collect()
        filenames = ['Loaded_On_Startup']

    # Always preload training data
    for f in train_names:
        print('Loading: ' + f)
        sys.stdout.flush()
        if train_shuffled_data_flat is None:
            train_shuffled_data_flat, train_shuffled_one_hot, _ = loader(f, scaler)
        else:
            temp_data, temp_one_hot, labels = loader(f, scaler)
            train_shuffled_data_flat = numpy.concatenate((train_shuffled_data_flat, temp_data), axis=0)
            train_shuffled_one_hot = numpy.concatenate((train_shuffled_one_hot, temp_one_hot), axis=0)

    # If only using part of the training data
    if train_count is not None:
        train_shuffled_data_flat = train_shuffled_data_flat[:train_count, :]
        train_shuffled_one_hot = train_shuffled_one_hot[:train_count, :]

    # Free some memory
    del temp_data, temp_one_hot

    # Start iterations, loop through all files x times
    for f in range(int(f_start), int(f_end)):
        print('-- File Iteration {} --'.format(f + 1))
        # loop through files if not preloaded, else will only run this loop once
        for file in filenames:
            print('-- New File {} --'.format(file))
            if not load:
                shuffled_data_flat, shuffled_one_hot = loader(file, scaler)
            if load:
                train_count = len(shuffled_one_hot)
            batches = int(math.floor(train_count / batch_size))
            # Epochs
            for e in range(int(e_start), int(e_end)):
                print('Epoch {}/{}'.format(e+1, epochs))
                sys.stdout.flush()
                # Batches
                for i in range(0, batches - 1):
                    metrics = model.train_on_batch(shuffled_data_flat[i*batch_size:(i+1)*batch_size, :],
                                                   shuffled_one_hot[i*batch_size:(i+1)*batch_size])
                    print_metrics((i+1)*batch_size, train_count, model, metrics, scores, e+1, f+1)
                    loss.append(metrics[0])
                    acc.append(metrics[1])
                    k.append(metrics[2])

                # Calc Metrics
                metrics = model.train_on_batch(shuffled_data_flat[(batches-1) * batch_size:train_count, :],
                                               shuffled_one_hot[(batches-1) * batch_size:train_count])
                print_metrics(train_count, train_count, model, metrics, scores, e+1, f+1)
                loss.append(metrics[0])
                acc.append(metrics[1])
                k.append(metrics[2])
                scores = model.evaluate(train_shuffled_data_flat, train_shuffled_one_hot)
                report = classification_report(numpy.argmax(train_shuffled_one_hot, axis=1), model.predict_classes(train_shuffled_data_flat), target_names=list(labels))
                print(report)
                class_rep.append(report)
                ev.append(list(scores))
                print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))

                # Save progress to file incase interrupted
                if save:
                    model_file = save_progress(**locals())
                print("Shuffling array")
                shuffle_in_place(train_shuffled_data_flat, train_shuffled_one_hot)
            # Restart epochs
            e_start = 0
    # Get Final scores and save off all data needed
    scores = model.evaluate(train_shuffled_data_flat, train_shuffled_one_hot)
    model.save('{date:%Y-%m-%d_%H:%M:%S}-{uuid}-{score:1.4f}.h5'.format(uuid=uuid, date=time, score=scores[1] * 100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    # Print class specific data
    for cr in class_rep:
        print(cr)
    # Remove train file to prove done
    os.remove(training_file_name)
    # Plot everything to files
    plot(loss, acc, numpy.asarray(ev), k, time, uuid)
    sys.stdout.flush()


def print_metrics(step, total, model, metrics, scores, e, f):
    print(('{:6d} / {:6d} - {:5s} {:1.4f} - {:5s} {:1.4f} - {:5s} {:1.4f}' +
           ' - {:7s} {:1.4f} - {:7s} {:1.4f} - {:8s} {:1.4f}' +
           ' - epoch {} - iter {}').format(step,
                                           total,
                                           model.metrics_names[0], metrics[0],
                                           model.metrics_names[1], metrics[1],
                                           model.metrics_names[2], metrics[2],
                                           'lst los', scores[0],
                                           'lst acc', scores[1],
                                           'lst top_2', scores[2],
                                           e, f))
    sys.stdout.flush()


def save_progress(**l):
    if l['model_file'] is not None:
        os.remove(l['model_file'])
    model_file = '{date:%Y-%m-%d_%H:%M:%S}-{uuid}-{score:1.4f}.h5'.format(uuid=l['uuid'], date=l['time'],
                                                                          score=l['scores'][1] * 100)
    l['model'].save(model_file)
    with open(l['training_file_name'], 'w') as file:
        file.write("{},{},{},{},{},{}".format(l['e'] + 1, l['e_end'], l['f'], l['f_end'], model_file, l['uuid']))
    numpy.save(l['loss_array_fname'], l['loss'])
    numpy.save(l['acc_array_fname'], l['acc'])
    numpy.save(l['ev_array_fname'], l['ev'])
    numpy.save(l['k_array_fname'], l['k'])
    numpy.save(l['class_array_fname'], l['class_rep'])
    print('Current progress saved')
    return model_file


def plot(loss, acc, ev, k, time, uuid):
    plt.plot(loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('{date:%Y-%m-%d_%H:%M:%S}-loss-{uuid}.png'.format(uuid=uuid, date=time))
    plt.clf()
    plt.plot(acc)
    plt.plot(k)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Batch')
    plt.legend(['train', 'top_2'], loc='upper left')
    plt.savefig('{date:%Y-%m-%d_%H:%M:%S}-acc-{uuid}.png'.format(uuid=uuid, date=time))
    plt.clf()
    plt.plot(ev[:, 1])
    plt.plot(ev[:, 2])
    plt.title('Model Evaluation')
    plt.ylabel('Accuracy')
    plt.xlabel('Batch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('{date:%Y-%m-%d_%H:%M:%S}-eval-{uuid}.png'.format(uuid=uuid, date=time))


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
    train_count = None
    iters = 8
    save = True

    # Modifications to test on laptop
    if os.uname()[1] == 'laptop':
        files = ['rf_data/training_data_chunk_0.pkl', 'rf_data/training_data_chunk_1.pkl']
        train_count = 10000
        iters = 2
        save = False

    # get uuid
    uuid = None
    training_file_name = 'training.temp'
    model_file = None
    if os.path.isfile(training_file_name):
        with open(training_file_name, 'r') as f:
            tmp = f.readline()
            e_start, e_end, f_start, f_end, model_file_tmp, uuid = tmp.split(',')
    elif len(sys.argv) > 1 and uuid is None:
        uuid = sys.argv[1]
    elif uuid is None:
        uuid = 'model'
    if len(sys.argv) > 2:
        model_file = sys.argv[2]

    # Check for existing log and if not create it
    if not os.path.isfile(uuid + '.log'):
        with open(uuid + '.log', 'w') as f:
            f.write('Starting training')

    # Train with log and tee output
    with open(uuid + '.log', 'a') as f:
        sys.stdout = Tee(sys.stdout, f)
        train_model(files, train_names, 512, 1, iters, load_data, uuid=uuid, load=True, model_file=model_file, train_count=train_count, save=save)

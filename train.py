from model1 import compile_model
import numpy
import math
import datetime
from load_data import load_data

numpy.random.seed(7)


def train_model(filenames, batch_size, epochs, train, samples=None):
    model = compile_model()
    test_data = None
    test_labels = None
    for file in filenames:
        print('-- New File -- {}'.format(file))
        shuffled_data, shuffled_one_hot, shuffled_labels = load_data(file)
        if samples is None:
            samples = len(shuffled_one_hot)
        train_count = int(math.floor(samples * train))
        batches = int(math.floor(train_count / batch_size))
        for e in range(0, epochs):
            print('Epoch {}/{}'.format(e, epochs))
            for i in range(0, batches - 1):
                loss = model.train_on_batch(shuffled_data[i*batch_size:(i+1)*batch_size, 0, :],
                                     shuffled_one_hot[i*batch_size:(i+1)*batch_size])
                print('{:6d} / {:6d} - {:5s} {:1.4f} - {:5s} {:1.4f}'.format((i+1)*batch_size,
                                                                             train_count,
                                                                             model.metrics_names[0], loss[0],
                                                                             model.metrics_names[1], loss[1]))
            loss = model.train_on_batch(shuffled_data[(batches-1) * batch_size:train_count, 0, :],
                                        shuffled_one_hot[(batches-1) * batch_size:train_count])
            print('{:6d} / {:6d} - {:5s} {:1.4f} - {:5s} {:1.4f}'.format(train_count,
                                                                        train_count,
                                                                        model.metrics_names[0], loss[0],
                                                                        model.metrics_names[1], loss[1]))
        if test_data is None:
            test_data = shuffled_data[train_count:samples, 0, :]
            test_labels = shuffled_one_hot[train_count:samples]
        else:
            test_data = numpy.concatenate([test_data, shuffled_data[train_count:samples, 0, :]])
            test_labels = numpy.concatenate([test_labels, shuffled_one_hot[train_count:samples]])

    scores = model.evaluate(test_data, test_labels)
    model.save('model1-{date:%Y-%m-%d %H:%M:%S}-{score}.h5'.format(date=datetime.datetime.now(), score=scores[1] * 100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


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
    train_model(files, 5000, 6, .8)

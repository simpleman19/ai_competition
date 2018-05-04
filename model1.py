from keras.models import Sequential
from keras.layers import Dense
import numpy
import pickle
import math
import datetime
from load_data import load_data

numpy.random.seed(7)


def train_model(filenames, batch, epochs, train, samples=None):
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    init_epoch = 0

    test_data = None
    test_labels = None
    for file in filenames:
        shuffled_data, shuffled_one_hot, shuffled_labels = load_data(file)
        if samples is None:
            samples = len(shuffled_one_hot)
        train_count = math.floor(samples * train)
        model.fit(shuffled_data[:train_count, 0, :], shuffled_one_hot[:train_count], batch_size=batch, epochs=init_epoch+epochs, initial_epoch=init_epoch)
        init_epoch += epochs
        if test_data is None:
            test_data = shuffled_data[train_count:samples, 0, :]
            test_labels = shuffled_one_hot[train_count:samples]
        else:
            test_data = numpy.concatenate([test_data, shuffled_data[train_count:samples, 0, :]])
            test_labels = numpy.concatenate([test_labels, shuffled_one_hot[train_count:samples]])

    scores = model.evaluate(test_data, test_labels)
    model.save('model-{date:%Y-%m-%d %H:%M:%S}-{score}.h5'.format(date=datetime.datetime.now(), score=scores[1] * 100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


if __name__ == '__main__':
    train_model(['rf_data/training_data_chunk_0.pkl',
                 'rf_data/training_data_chunk_1.pkl',
                 'rf_data/training_data_chunk_2.pkl',
                 'rf_data/training_data_chunk_3.pkl',
                 'rf_data/training_data_chunk_4.pkl',
                 'rf_data/training_data_chunk_5.pkl',
                 'rf_data/training_data_chunk_6.pkl',
                 'rf_data/training_data_chunk_7.pkl'
                 ], 5000, 4, .8)

import matplotlib
matplotlib.use('Agg')
from model2 import compile_model, load_data
import numpy
import math
import sys
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt


def test(model_file, test_data, test_labels, count=None):
    model, scaler = compile_model()
    model.load_weights(model_file)
    if scaler is not None:
        test_data = scaler.fit_transform(test_data)
    if count is None:
        scores = model.evaluate(test_data, test_labels)
    else:
        scores = model.evaluate(test_data[:count, :], test_labels[:count, :])
    print(scores)


def predict(model_file, test_data, test_labels, count=None):
    model, scaler = compile_model()
    model.load_weights(model_file)
    if scaler is not None:
        test_data = scaler.fit_transform(test_data)
    if count is None:
        prediction = model.predict_classes(test_data)
    else:
        prediction = model.predict_classes(test_data[:count])
    numpy.savetxt('predictions.csv', prediction, delimiter=',')
    numpy.savetxt('actuals.csv', test_labels, delimiter=',')


if __name__ == '__main__':
    shuffled_data_flat, shuffled_one_hot, types = load_data('rf_data/training_data_chunk_14.pkl')
    predict("archive/2018-05-28_02:56:46-1f297724-2c05-4d34-ba37-f1db193b1512-62.9962.h5",
            shuffled_data_flat, shuffled_one_hot)

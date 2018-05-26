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


if __name__ == '__main__':
    shuffled_data_flat, shuffled_one_hot = load_data('rf_data/training_data_chunk_14.pkl')
    test("archive/2018-05-15 16:44:52-6b8a95c1-a168-4e78-8c6e-2ef91603f973-48.1910.h5",
         shuffled_data_flat, shuffled_one_hot, 10000)

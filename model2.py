from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU, BatchNormalization
from keras.backend import relu
from keras.optimizers import Adam
from keras.metrics import top_k_categorical_accuracy
from sklearn import preprocessing
import numpy
from load_data import load_data


def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def relu_max(x):
    return relu(x, max_value=20)


def compile_model():
    scaler = None
    model = Sequential()
    model.add(Dense(2048,
                    input_dim=2048,
                    activation='linear'
                    ))
    model.add(PReLU())
    model.add(Dropout(.1))
    model.add(Dense(24576, activation='linear'))
    model.add(PReLU())
    model.add(Dropout(.1))
    model.add(Dense(12288, activation='linear'))
    model.add(PReLU())
    model.add(Dropout(.1))
    model.add(Dense(24, activation='softmax'))
    adam = Adam(lr=.0005, beta_1=.9, beta_2=.98, decay=0, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', top_2])
    # scaler = Scaler(preprocessing.Normalizer())
    return model, scaler


class Scaler:
    def __init__(self, scaler):
        self.scaler = scaler

    def fit_transform(self, array):
        temp = self.scaler.fit_transform(array[:, :1024])
        return numpy.concatenate((array, temp), axis=1)


if __name__ == '__main__':
    compile_model()

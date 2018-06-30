from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU, BatchNormalization
from keras.backend import relu
from keras.optimizers import Adam
from keras.metrics import top_k_categorical_accuracy
from sklearn import preprocessing
import numpy
from load_data import load_data_sub as load_data


def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def compile_model():
    scaler = None
    model = Sequential()
    model.add(Dense(2048,
                    input_dim=2048,
                    activation='linear'
                    ))
    model.add(PReLU())
    model.add(Dropout(.1))
    model.add(Dense(12288, activation='linear'))
    model.add(PReLU())
    model.add(Dropout(.1))
    model.add(Dense(8192, activation='sigmoid'))
    model.add(Dense(6, activation='softmax'))
    adam = Adam(lr=.00025, beta_1=.9, beta_2=.9, decay=0, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', top_2])
    return model, scaler


class Scaler:
    def __init__(self, scaler):
        self.scaler = scaler

    def fit_transform(self, array):
        temp = self.scaler.fit_transform(array[:, :1024])
        return numpy.concatenate((array, temp), axis=1)


if __name__ == '__main__':
    compile_model()

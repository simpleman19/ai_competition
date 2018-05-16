from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU
from keras.backend import relu
from keras.optimizers import Adam, SGD
from keras.metrics import top_k_categorical_accuracy
from sklearn import preprocessing


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
    model.add(Dense(8192, activation='linear'))
    model.add(PReLU())
    model.add(Dense(8192, activation='linear'))
    model.add(PReLU())
    model.add(Dense(4096, activation='linear'))
    model.add(PReLU())
    model.add(Dense(24, activation='softmax'))
    sgd = SGD(lr=0.01, momentum=0.2, decay=0.001, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', top_2])
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return model, scaler


if __name__ == '__main__':
    compile_model()

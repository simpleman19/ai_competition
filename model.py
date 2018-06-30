from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU
from keras.backend import relu
from keras.optimizers import Adam, SGD
from keras.metrics import top_k_categorical_accuracy
from sklearn import preprocessing
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
    model.add(Dense(12288, activation='linear'))
    model.add(PReLU())
    model.add(Dropout(.1))
    model.add(Dense(12288, activation='linear'))
    model.add(PReLU())
    model.add(Dropout(.1))
    model.add(Dense(24, activation='softmax'))
    adam = Adam(lr=.0005, beta_1=.9, beta_2=.95, decay=0, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', top_2])
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return model, scaler


if __name__ == '__main__':
    compile_model()
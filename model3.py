from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Dropout, LSTM, PReLU, ELU
from keras.optimizers import Adam, SGD
from keras.backend import relu
from keras.metrics import top_k_categorical_accuracy
from sklearn import preprocessing
from keras import regularizers
from load_data import load_data_lstm as load_data


def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def relu_max(x):
    return relu(x, max_value=20)


def compile_model():
    model = Sequential()
    model.add(LSTM(256, input_shape=(1024, 2), activation='tanh'))
    model.add(Dense(128, activation='linear'))
    model.add(PReLU())
    model.add(Dense(24, activation='softmax'))
    adam = Adam(lr=.0005, beta_1=.9, beta_2=.98, decay=0, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', top_2])
    return model, None


if __name__ == '__main__':
    compile_model()

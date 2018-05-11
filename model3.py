from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam, SGD
from keras import regularizers


def compile_model():
    model = Sequential()
    model.add(LSTM(32, input_shape=(1024, 2)))
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    compile_model()

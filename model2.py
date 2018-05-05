from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers


def compile_model():
    model = Sequential()
    model.add(Dense(1024,
                    input_dim=1024,
                    activation='relu'
                    ))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    compile_model()
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras import regularizers


def compile_model():
    model = Sequential()
    model.add(Dense(2048,
                    input_dim=2048,
                    activation='relu'
                    ))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(24, activation='softmax'))
    sgd = SGD(lr=0.02, momentum=0.1, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    compile_model()

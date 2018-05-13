from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras import regularizers


def compile_model():
    model = Sequential()
    model.add(Dense(2048,
                    input_dim=2048,
                    activation='sigmoid'
                    ))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(512, activation='relu./ad   '))
    model.add(Dense(24, activation='softmax'))
    sgd = SGD(lr=0.02, momentum=0.2, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    compile_model()

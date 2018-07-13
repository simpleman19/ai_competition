import pickle
import numpy as np
from sklearn import preprocessing
import gc
import math


def __load(fname, shuffled=True):
    TAG = "-"
    # load data from files
    with open(fname, 'rb') as f:
        dataCube = pickle.load(f, encoding='latin-1')
        dataCubeKeyIndices = list(zip(*dataCube))

    # get all mod types
    modTypes = np.unique(dataCubeKeyIndices[0])

    # get all SNR values
    snrValues = np.unique(dataCubeKeyIndices[1])

    # create one-hot vectors for each mod type
    oneHotArrays = np.eye(len(modTypes), dtype=int)

    # Count Number of examples
    print(TAG + "Counting Number of Examples in Dataset...")
    number_of_examples = 0
    for modType in modTypes:
        for snrValue in snrValues:
            number_of_examples = number_of_examples + len(dataCube[modType, snrValue])

    print(TAG + 'Number of Examples in Dataset: ' + str(number_of_examples))

    # pre-allocate arrays
    signalData = [None] * number_of_examples
    oneHotLabels = [None] * number_of_examples
    # signalLabels = [None] * number_of_examples

    # for each mod type ... for each snr value ... add to signalData, signalLabels, and create one-Hot vectors
    example_index = 0
    one_hot_index = 0
    instance_shape = None

    with open('loads', 'a') as f:
        line = ",".join(map(str, modTypes))
        f.write(line + "\n")

    for modType in modTypes:
        print(TAG + "[Modulation Dataset] Adding Collects for: " + str(modType))
        for snrValue in snrValues:

            # get data for key,value
            collect = dataCube[modType, snrValue]

            for instance in collect:
                signalData[example_index] = instance
                # signalLabels[example_index] = (modType, snrValue)
                oneHotLabels[example_index] = oneHotArrays[one_hot_index]
                example_index += 1

                if instance_shape is None:
                    instance_shape = np.shape(instance)

        one_hot_index += 1  # keep track of iteration for one hot vector generation
    del dataCube
    del dataCubeKeyIndices
    gc.collect()
    # convert to np.arrays
    print(TAG + "Converting to numpy arrays...")
    signalData = np.asarray(signalData, dtype=np.float32)
    oneHotLabels = np.asarray(oneHotLabels, dtype=np.float32)
    # signalLabels = np.asarray(signalLabels)

    # Shuffle data
    print(TAG + "Shuffling Data...")
    """ signalData_shuffled, signalLabels_shuffled, oneHotLabels_shuffled """
    if shuffled:
        # Randomly shuffle data, use predictable seed
        np.random.seed(221)
        shuffle_indices = np.random.permutation(np.arange(len(oneHotLabels)))
        signalData_shuffled = signalData[shuffle_indices]
        del signalData
        # signalLabels_shuffled = signalLabels[shuffle_indices]
        # del signalLabels
        oneHotLabels_shuffled = oneHotLabels[shuffle_indices]
        del oneHotLabels

        return signalData_shuffled, oneHotLabels_shuffled, modTypes
    else:
        return signalData, oneHotLabels, modTypes


def load_act_data(fname, shuffled=True):
    TAG = "-"
    # load data from files
    with open(fname, 'rb') as f:
        dataCube = pickle.load(f, encoding='latin-1')
        dataCubeKeyIndices = dataCube.keys()

    # Count Number of examples
    print(TAG + "Counting Number of Examples in Dataset...")
    number_of_examples = len(dataCube)

    print(TAG + 'Number of Examples in Dataset: ' + str(number_of_examples))

    # pre-allocate arrays
    signalData = [None] * number_of_examples
    index = [None] * number_of_examples
    # signalLabels = [None] * number_of_examples

    # for each mod type ... for each snr value ... add to signalData, signalLabels, and create one-Hot vectors
    example_index = 0
    instance_shape = None

    for key in dataCubeKeyIndices:
        signalData[example_index] = dataCube[key]
        # signalLabels[example_index] = (modType, snrValue)
        index[example_index] = key
        example_index += 1

        if instance_shape is None:
            instance_shape = np.shape(dataCube[key])

    # convert to np.arrays
    print(TAG + "Converting to numpy arrays...")
    signalData = np.asarray(signalData, dtype=np.float32)
    index = np.asarray(index, dtype=np.float32)

    return dataCube


def load_data(fname, scaler=None, shuffled=True):
    '''  Load dataset from pickled file '''

    signalData_shuffled, oneHotLabels_shuffled, modTypes = __load(fname, shuffled)

    if scaler is not None:
        signalData_shuffled = scaler.fit_transform(signalData_shuffled.reshape(signalData_shuffled.shape[0], 2048))
    else:
        signalData_shuffled = signalData_shuffled.reshape(signalData_shuffled.shape[0], 2048)

    gc.collect()

    return signalData_shuffled, oneHotLabels_shuffled, modTypes


def load_data_lstm(fname, scaler=None, shuffled=True):
    '''  Load dataset from pickled file '''

    signalData_shuffled, oneHotLabels_shuffled, modTypes = __load(fname, shuffled)

    signalData_shuffled_flat = signalData_shuffled.transpose(0, 2, 1)

    return signalData_shuffled_flat, oneHotLabels_shuffled, modTypes


def load_data_conv(fname, scaler=None, shuffled=True):
    '''  Load dataset from pickled file '''

    signal_shuffled, one_hot_shuffled, modTypes = __load(fname, shuffled)

    return signal_shuffled, one_hot_shuffled, modTypes


def load_data_sub(fname, scaler=None, shuffled=True):
    '''  Load dataset from pickled file '''

    signal_shuffled, one_hot_shuffled, modTypes = __load(fname, shuffled)

    if scaler is not None:
        signal_shuffled = scaler.fit_transform(signal_shuffled.reshape(signal_shuffled.shape[0], 2048))
    else:
        signal_shuffled = signal_shuffled.reshape(signal_shuffled.shape[0], 2048)

    x = list(one_hot_shuffled.shape)[0]
    new_one_hot = np.zeros((x, 6))
    for x in range(len(one_hot_shuffled)):
        new_one_hot[x] = subset(one_hot_shuffled[x])

    gc.collect()

    return signal_shuffled, new_one_hot, modTypes


def subset(row):
    return_row = np.array([0, 0, 0, 0, 0, 0])
    # 1 4 9 11 22
    if row[1] == 1:
        return_row[1] = 1
    elif row[4] == 1:
        return_row[2] = 1
    elif row[9] == 1:
        return_row[3] = 1
    elif row[11] == 1:
        return_row[4] = 1
    elif row[22] == 1:
        return_row[5] = 1
    else:
        return_row[0] = 1

    return return_row


def shuffle_in_place(data, one_hots):
    num_to_shuffle = len(data)
    shuffle_args = np.arange(0, num_to_shuffle)
    np.random.shuffle(shuffle_args)
    num_to_shuffle -= 1

    for x in range(int(math.floor(num_to_shuffle/2))):
        # Signal
        temp = data[shuffle_args[x]]
        data[shuffle_args[x]] = data[shuffle_args[num_to_shuffle - x]]
        data[shuffle_args[num_to_shuffle - x]] = temp
        # One Hot
        temp = one_hots[shuffle_args[x]]
        one_hots[shuffle_args[x]] = one_hots[shuffle_args[num_to_shuffle - x]]
        one_hots[shuffle_args[num_to_shuffle - x]] = temp

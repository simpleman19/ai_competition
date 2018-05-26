import pickle
import numpy as np
from sklearn import preprocessing
import gc


def __load(fname):
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
    # Randomly shuffle data, use predictable seed
    np.random.seed(221)
    shuffle_indices = np.random.permutation(np.arange(len(oneHotLabels)))
    signalData_shuffled = signalData[shuffle_indices]
    del signalData
    # signalLabels_shuffled = signalLabels[shuffle_indices]
    # del signalLabels
    oneHotLabels_shuffled = oneHotLabels[shuffle_indices]
    del oneHotLabels

    return signalData_shuffled, oneHotLabels_shuffled


def load_data(fname, scaler=None):
    '''  Load dataset from pickled file '''

    signalData_shuffled, oneHotLabels_shuffled = __load(fname)

    if scaler is not None:
        signalData_shuffled = scaler.fit_transform(signalData_shuffled.reshape(signalData_shuffled.shape[0], 2048))
    else:
        signalData_shuffled = signalData_shuffled.reshape(signalData_shuffled.shape[0], 2048)

    gc.collect()

    return signalData_shuffled, oneHotLabels_shuffled


def load_data_lstm(fname, scaler=None):
    '''  Load dataset from pickled file '''

    signalData_shuffled, oneHotLabels_shuffled = __load(fname)

    signalData_shuffled_flat = signalData_shuffled.transpose(0, 2, 1)

    return signalData_shuffled_flat, oneHotLabels_shuffled


def load_data_conv(fname, scaler=None):
    '''  Load dataset from pickled file '''

    signalData_shuffled, oneHotLabels_shuffled = __load(fname)

    return signalData_shuffled, oneHotLabels_shuffled
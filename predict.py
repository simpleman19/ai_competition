import matplotlib
matplotlib.use('Agg')
from model import compile_model, load_data
from load_data import load_act_data
import threading
import math

output = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
          'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
          'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
          'QAM32', 'QAM64', 'QPSK']

sample = ['QPSK', '16PSK', 'FM_NB', 'AM_DSB', 'BPSK', 'NOISE', 'CPFSK_75KHz', 'QAM16',
          '8PSK', 'GMSK', 'MSK', 'OQPSK', 'FM_WB', 'PI4QPSK', 'QAM64', 'GFSK_75KHz',
          'GFSK_5KHz', 'AM_SSB', 'APSK16_c34', '2FSK_5KHz', 'CPFSK_5KHz', 'QAM32',
          '2FSK_75KHz', 'APSK32_c34']

transform_dict = None


def predict(model_, test_data, name):
    with open(name, 'w') as f:
        line = "Index,"
        line += ",".join(map(str, sample))
        f.write(line + "\n")
        for key in test_data.keys():
            prediction = model_.predict_proba(test_data[key].reshape(1, 2048))
            line = str(key) + ","
            line += ",".join(map(str, prediction[0]))
            line = transform_line(line, output, sample)
            f.write(line + "\n")


def split_dict(dictionary, count):
    ret_arr_dicts = count * [None]
    for i in range(count):
        ret_arr_dicts[i] = {}
    num = len(dictionary)
    for key in dictionary.keys():
        ret_arr_dicts[int(math.floor(key/((num+1)/count)))][key] = dictionary[key]
    return ret_arr_dicts


def multi_threaded():
    threads = 4
    date_flat = load_act_data('rf_data/Test_Set_1_Army_Signal_Challenge.pkl', shuffled=False)
    dicts = split_dict(date_flat, threads)
    thread_pointers = threads * [None]
    for i in range(threads):
        model, _ = compile_model()
        model.load_weights("archive/b5823df9-5f3f-4b14-9922-94194b3c5131-69.0288.h5")
        thread_pointers[i] = threading.Thread(target=predict, args=(model, dicts[i], 'predictions{}.csv'.format(i)))
        print("Starting thread " + str(i))
        thread_pointers[i].start()
    for t in thread_pointers:
        t.join()


def single_threaded():
    date_flat = load_act_data('rf_data/Test_Set_1_Army_Signal_Challenge.pkl', shuffled=False)
    model, _ = compile_model()
    model.load_weights("archive/bdde9b04-bc8d-41f8-acfb-9965936ffbd4-68.2564.h5")
    predict(model, date_flat, 'predictions.csv')


def transform_line(line, from_array, to_array):
    global transform_dict
    if transform_dict is None:
        transform_dict = {}
        for x in range(len(from_array)):
            index = get_index(from_array[x], to_array)
            if index is None:
                Exception("Null Index")
            transform_dict[x] = index
    elif not check_transform(from_array, to_array):
        Exception("Invalid transform array")
    split_line = line.split(',')
    transformed = 24 * [0]
    for key in transform_dict:
        transformed[transform_dict[key]] = split_line[key+1]
    new_line = str(split_line[0]) + ","
    new_line += ",".join(map(str, transformed))
    return new_line


def check_transform(from_array, to_array):
    global transform_dict
    for key in transform_dict:
        if not from_array[key] == to_array[transform_dict[key]]:
            return False
    return True


def get_index(val, array):
    for x in range(len(array)):
        if array[x] == val:
            return x
    return None


if __name__ == '__main__':
    single_threaded()

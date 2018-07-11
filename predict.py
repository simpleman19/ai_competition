import matplotlib
matplotlib.use('Agg')
from model import compile_model, load_data
from load_data import load_act_data
import threading
import math


def predict(model_, test_data, name):
    with open(name, 'w') as f:
        for key in test_data.keys():
            prediction = model_.predict_proba(test_data[key].reshape(1, 2048))
            line = str(key) + ","
            line += ",".join(map(str, prediction[0]))
            f.write(line + "\n")
    # numpy.savetxt('predictions.csv', prediction, delimiter=',', fmt="%1.4f")


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
        model.load_weights("archive/2018-05-28_15:59:39-b5823df9-5f3f-4b14-9922-94194b3c5131-69.0288.h5")
        thread_pointers[i] = threading.Thread(target=predict, args=(model, dicts[i], 'predictions{}.csv'.format(i)))
        print("Starting thread " + str(i))
        thread_pointers[i].start()
    for t in thread_pointers:
        t.join()


def single_threaded():
    date_flat = load_act_data('rf_data/Test_Set_1_Army_Signal_Challenge.pkl', shuffled=False)
    model, _ = compile_model()
    model.load_weights("archive/2018-05-28_15:59:39-b5823df9-5f3f-4b14-9922-94194b3c5131-69.0288.h5")
    predict(model, date_flat, 'predictions.csv')


if __name__ == '__main__':
    single_threaded()

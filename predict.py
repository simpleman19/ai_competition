import matplotlib
matplotlib.use('Agg')
from model import compile_model, load_data
from load_data import load_act_data
import numpy


def predict(model_file, test_data, count=None):
    model, _ = compile_model()
    model.load_weights(model_file)

    with open('predictions.csv', 'w') as f:
        for key in test_data.keys():
            prediction = model.predict_proba(test_data[key].reshape(1, 2048))
            line = str(key) + ","
            for x in prediction[0]:
                line += "{},".format(x)
            line = line[:-1] + "\n"
            f.write(line)
    # numpy.savetxt('predictions.csv', prediction, delimiter=',', fmt="%1.4f")


if __name__ == '__main__':
    date_flat = load_act_data('rf_data/Test_Set_1_Army_Signal_Challenge.pkl', shuffled=False)
    predict("archive/2018-05-28_15:59:39-b5823df9-5f3f-4b14-9922-94194b3c5131-69.0288.h5", date_flat, count=10000)

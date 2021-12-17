import joblib
import xgboost
import os
import numpy as np

curr_path = os.path.dirname(os.path.realpath(__file__))
print('Current Path: ', curr_path)
xgb = joblib.load(r'.\boost_final.pkl.compressed')


def predict(attributes: np.array):
    pred = xgb.predict(attributes)

    print('Flux Value Predicted')

    return pred[0]

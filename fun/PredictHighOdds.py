# https://cloud.google.com/ml-engine/docs/custom-prediction-routines#predictor-class
# https://cloud.google.com/ml-engine/docs/scikit/custom-prediction-routine-scikit-learn

import os
import pickle

import numpy as np
import joblib

class PredictHighOdds(object):

    def predict(self, instances, **kwargs):
        
        return [1 if instances.Diff_odds[i] < 0 else 0 for i in instances.index ]

    def fit(self, instances, true_values, **kwargs): return self
    def get_params(self, deep): return {}

    
    @classmethod
    def from_path(cls, model_dir):
        
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)

        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        return cls(model, preprocessor)
        raise NotImplementedError()
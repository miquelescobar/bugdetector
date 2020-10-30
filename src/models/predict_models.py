import pickle
import numpy as np


tree = pickle.load(open('../../models/tree.sav', 'rb'))
mlp = pickle.load(open('../../models/mlp.sav', 'rb'))
rf = pickle.load(open('../../models/rf.sav', 'rb'))
models = {
    'tree': tree,
    'mlp': mlp,
    'rf': rf
}

def predict(X,
            model: str = 'rf',
            probs=True,
            labels=False):
    """
    """
    prediction = {}
    
    if model == 'all':
        prediction = { m: {} for m in models}
        for model_name, loaded_model in models.items():
            if probs:
                prediction[model_name]['probs'] = loaded_model.predict_proba(X)
            if labels:
                prediction[model_name]['labels'] = loaded_model.predict(X)
    
    else:
        prediction = {model:{}}
        loaded_model = models[model]
        if probs:
            prediction[model]['probs'] = loaded_model.predict_proba(X)
        if labels:
            prediction[model]['labels'] = loaded_model.predict(X)
    
    return prediction




if __name__ == '__main__':
    X = np.array([[True, False, 164, 1099, 17772, 55.6, 1984, 12.2, 12.1, 1.8, 0,
       3796, 85, 42, 9.5, 1490, 1, 208, 11, 0, 0, 1361, 2, 0, 0, 0, 1686,
       40092, 14196, 7501, 0, 760, 510, 1490, 1, 0, 17, 17, 5618, 7501, 0,
       19168, 425880, 4.5, 4.128068091844813, 127, 35, 4, 1760, 4, 0, 0,
       2, 0.0010712894429294898]])
    print(predict(X, model='all', labels=True))

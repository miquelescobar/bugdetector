# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

import itertools
from sklearn import preprocessing, metrics
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

import matplotlib.pyplot as plt


# %%
# Random seed
random_seed = 42
def set_random_seed(seed=42):
    np.random.seed(seed)
set_random_seed(random_seed)


# %% [markdown]
# ## Models definition
print('Defining models...', flush=True)
# %% [markdown]
# ### LightGBM

# %%
import lightgbm as lgb


def train_lightgbm(train, valid, test, target_feature, n_classes):
    
    #train, valid, test = get_data_splits(df)
    
    feature_cols = train.columns.drop(target_feature)
    
    dtrain = lgb.Dataset(train[feature_cols], label=train[target_feature])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid[target_feature])

    param = {'num_leaves': 64, 'objective': 'multiclass', 
             'metric': 'multi_logloss', 'num_class': n_classes, 'seed': random_seed}
    
    print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
                    early_stopping_rounds=10, verbose_eval=False)

    test_pred = bst.predict(test[feature_cols])
    test_pred = np.argmax(test_pred, axis=1)
    
    test_accuracy = metrics.accuracy_score(test[target_feature], test_pred)
    test_f1score = metrics.f1_score(test[target_feature], test_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test f1 score: {test_f1score:.4f}")

    return bst

def train_lgbm_for_test(train, test, target_feature, n_classes):
    
    train, valid = get_train_valid_split(train)
    
    feature_cols = train.columns.drop(target_feature)

    dtrain = lgb.Dataset(train[feature_cols], label=train[target_feature])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid[target_feature])

    param = {'num_leaves': 64, 'objective': 'multiclass', 
             'metric': 'multi_logloss', 'num_class': n_classes, 'seed': random_seed}
    
    #print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
                    early_stopping_rounds=10, verbose_eval=False)
    
    test_pred = bst.predict(test)
    test_pred = np.argmax(test_pred, axis=1)
    test_pred = test_pred+1
    return test_pred

def new_features(df, cat_features):
    interactions = pd.DataFrame(index=df.index)

    # Iterate through each pair of features, combine them into interaction features
    for feature1, feature2 in itertools.combinations(cat_features, 2):
        interaction_feature = feature1+'_'+feature2
        interactions[interaction_feature] = preprocessing.LabelEncoder().fit_transform(df[feature1].apply(str)+'_'+df[feature2].apply(str))
    
    return interactions, list(interactions.columns)

# %% [markdown]
# ### Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

def train_decisiontree(X_train, X_test, y_train, y_test):

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    test_pred = clf.predict(X_test)
    test_pred = np.argmax(test_pred, axis=1)
    
    # test_accuracy = metrics.accuracy_score(y_test, test_pred)
    # test_f1score = metrics.f1_score(y_test, test_pred)
    # print(f"Test accuracy: {test_accuracy:.4f}")
    # print(f"Test f1 score: {test_f1score:.4f}")

    return clf

# %% [markdown]
# ### MLP

# %%
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def train_MLPClassifier(X_train, X_test, y_train, y_test):
       
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)    
    
    #print("Training model!")
    est = MLPClassifier(hidden_layer_sizes=(30, 20, 10, 10 ), activation='relu', solver='adam', alpha=0.0001, 
                       batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
                       power_t=0.5, max_iter=40, shuffle=True, random_state=None, tol=0.0001, 
                       verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                       early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
                       epsilon=1e-08, n_iter_no_change=10)
    
    est.fit(X_train, y_train)
    
    #print(f"Test accuracy score: {est.score(X_test, y_test):.4f}")

    return est

#train, valid = train_test_split(X_y, 0.2)
#est = train_MLPRegressor(train, valid, 'damage_grade')

# %% [markdown]
# ### Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

def train_randomforest(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    test_pred = clf.predict(X_test)
    test_pred = np.argmax(test_pred, axis=1)
    
    # test_accuracy = metrics.accuracy_score(y_test, test_pred)
    # test_f1score = metrics.f1_score(y_test, test_pred)
    # print(f"Test accuracy: {test_accuracy:.4f}")
    # print(f"Test f1 score: {test_f1score:.4f}")

    return clf

# %% [markdown]
# ## Data import and preparation

# %%
DATA_ROOT = '../../data/processed'

print('Processing training dataset...', flush=True)

# %%
df = pd.read_csv(f'{DATA_ROOT}/bugs-multitarget.csv')
df['priority'] = df['priority'].astype(int)
df.head()


# %%
from sklearn.preprocessing import MultiLabelBinarizer

N_CLASSES = 6
common_cols = list(df.columns)
common_cols.remove('priority')


mlb = MultiLabelBinarizer()
df = df.groupby(common_cols)['priority'].apply(set).reset_index()
df = shuffle(df)
df['priority'] = df['priority'].to_numpy()
y_raw = df['priority']
y_raw = y_raw.to_numpy()
y_raw = [np.array(list(x)) for x in y_raw]
y = mlb.fit_transform(df['priority'])
df.drop(['commitHash', 'priority'], axis=1, inplace=True)
X = df.to_numpy()
df.head()


# %%
X.shape


# %%
y.shape

# %% [markdown]
# ### Downsampling

# %%
nobug_idxs = []
all_idxs = []
for i, label in enumerate(y):
  all_idxs.append(i)
  if np.array_equal(label, [1, 0, 0, 0, 0, 0]):
    nobug_idxs.append(i)

print(len(nobug_idxs)/len(y))


# %%
import random

D = 0.75
rm_idxs = random.sample(nobug_idxs, int(len(nobug_idxs)*D))
use_idxs = np.array(list(set(all_idxs) - set(rm_idxs)))
X = X[use_idxs]
y = y[use_idxs]


# %%
X.shape


# %%
y.shape

# %% [markdown]
# ## Training & Evaluation
print('Training models...', flush=True)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% [markdown]
# ### Decision Tree

# %%
tree = train_decisiontree(X_train, X_test, y_train, y_test)


# %%
yPred = tree.predict(X_test)


# %%
yPred.sum(axis=0)


# %%
y_test.sum(axis=0)

# %% [markdown]
# ### MLP

# %%
mlp = train_MLPClassifier(X_train, X_test, y_train, y_test)


# %%
yPred = mlp.predict(X_test)


# %%
yPred.sum(axis=0)

# %% [markdown]
# ### Random Forest

# %%
rf = train_randomforest(X_train, X_test, y_train, y_test)


# %%
yPred = rf.predict(X_test)


# %%
yPred.sum(axis=0)

# %% [markdown]
# ## Results

# %%
def plot_confusion_matrix(clf, X, y):
    predY = clf.predict(X)
    cm = multilabel_confusion_matrix(y, predY)
    cms = []
    for i, m in enumerate(cm):
      cms.append(m / m.sum(axis=1)[:, np.newaxis])
      plt.subplot(len(cm)/2, 2, i+1)
      plt.tight_layout()
      plt.imshow(m / m.sum(axis=1)[:, np.newaxis])
      plt.title(f'Priority {i}')
    return cms

def get_accuracy(clf, X, y):
    accuracy = clf.score(X, y)
    print(accuracy)
    return accuracy

def get_bug_classification_accuracy(clf, X, y):
  yPreds = clf.predict(X)
  good = 0
  n = 0
  for i, yPred in enumerate(yPreds):
    for j, label in enumerate(yPred):
      if j > 0:
        n += 1
        if label == y[i, j]:
          good += 1
  acc = good / n
  #print(acc)
  return acc

# %% [markdown]
# ### Decision Tree

# %%
plot_confusion_matrix(tree, X_test, y_test)


# %%
get_accuracy(tree, X_test, y_test)
get_bug_classification_accuracy(tree, X_test, y_test)

# %% [markdown]
# ### MLP

# %%
plot_confusion_matrix(mlp, X_test, y_test)


# %%
get_accuracy(mlp, X_test, y_test)
get_bug_classification_accuracy(mlp, X_test, y_test)

# %% [markdown]
# ### Random Forest

# %%
plot_confusion_matrix(rf, X_test, y_test)


# %%
get_accuracy(rf, X_test, y_test)
get_bug_classification_accuracy(rf, X_test, y_test)


# %%


# %% [markdown]
# ## Save Models

# %%
import pickle
print('Storing trained models into ../../models/', flush=True)


# %%
pickle.dump(tree, open('../../models/tree.sav', 'wb'))
pickle.dump(rf, open('../../models/rf.sav', 'wb'))
pickle.dump(mlp, open('../../models/mlp.sav', 'wb'))


# %%




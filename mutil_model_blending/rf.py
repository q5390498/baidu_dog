from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import h5py
import numpy as np
from sklearn.utils import shuffle

X_train = []
X_val = []
X_test = []
y_val = []

'''"gap_ResNet50.h5", "gap_VGG16.h5","'''
for filename in [ "gap_Xception.h5", "gap_ResNet50.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_val.append(np.array(h['val']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['train_label'])
        y_val = np.array([h['val_label']])

X_train = np.concatenate(X_train, axis=1)
X_val = np.concatenate(X_val, axis=1)
X_test = np.concatenate(X_test, axis=1)

y_train = y_train.reshape((-1, 1))
y_val = y_val.reshape((-1, 1))
X_train = np.vstack((X_train,X_val))
print np.shape(y_train),np.shape(y_val)
y_train = np.vstack((y_train, y_val))
X_train, y_train = shuffle(X_train, y_train)
print np.shape(X_val), np.shape(y_val)
X_train, y_train = shuffle(X_train, y_train)
print np.shape(X_val), np.shape(y_val)
y_val = y_val.reshape((-1, 1))
print np.shape(X_val), np.shape(y_val)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
clf = XGBClassifier()
clf.fit(X_train, y_train)
print clf.score(X_val, y_val)
#a = np.sum(clf.predict(X_val) == y_val)
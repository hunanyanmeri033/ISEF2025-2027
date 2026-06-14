# diagnostic_preprocess.py
import pandas as pd
import numpy as np
from rdkit import Chem
import deepchem as dc
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
csv_path = "/home/merih/ISEF2025-2026/data/raw/HIV.csv"
df = pd.read_csv(csv_path)
#df['smiles'] = pd.to_numeric(df['smiles'], errors='coerce')
X_raw_series = df["smiles"]
Y_raw_series = df["HIV_active"]


# ... (Assume X_raw_series, Y_raw_series are defined and cleaned into lists) ...
valid_smiles = []
valid_labels = []
print("before filtering")
# (populate lists as before) ...
for sm, lab in zip(X_raw_series, Y_raw_series):
    if sm is None:
        print(sm+"removed")
        print("hello")
        continue
     # ensure SMILES is a string
    if not isinstance(sm, str):
        #print(sm+"removed")
        print("int")
        continue
 
    if sm[0].isdigit():
        continue
    
    mol = Chem.MolFromSmiles(str(sm))
    if mol is None:
        print("hello")
        continue

    valid_smiles.append(str(sm))
    valid_labels.append(str(lab))
#convert to fingerprints
valid_X = valid_smiles
valid_X = np.array(valid_X, dtype= str)
valid_labels = np.array(valid_labels, dtype = str)
#valid_X = valid_X.reshape(-1,1)
#valid_labels = valid_labels.reshape(-1,1)
featurizer = dc.feat.CircularFingerprint(size=2048)
valid_X = featurizer.featurize(valid_X)
print(valid_X.shape)
print(valid_labels.shape)
dataset = dc.data.NumpyDataset(valid_X, valid_labels, ids=valid_smiles)

scaffoldsplitter = dc.splits.ScaffoldSplitter()
train, valid, test = scaffoldsplitter.train_valid_test_split(dataset)

#unique, counts = np.unique(train.y, return_counts=True)
#print(dict(zip(unique, counts)))
#unique, counts = np.unique(trainy, return_counts=True)


#convert to type int
train.y[:] = train.y.astype(np.int8)
valid.y[:] = valid.y.astype(np.int8)
test.y[:] = test.y.astype(np.int8)
print("unique types ----------------------------------------------")
print(np.unique(train.y))
#convert to dataframe
traindf = train.to_dataframe()
validdf = valid.to_dataframe()
testdf = test.to_dataframe()
print("dtype stuff")
print(traindf['y'].dtype)
print("value counts")
print(traindf['y'].value_counts())
print("value counts")
print(validdf['y'].value_counts())
print(testdf['y'].value_counts())

#print(Xtrainsampled.columns)

#print(X_train.shape)
#print(np.unique(y_train))

#filtering out 0s and 1s
positive_train = traindf[traindf['y'] == '1']
negative_train = traindf[traindf['y'] == '0']
positive_valid = validdf[validdf['y'] == '1']
negative_valid = validdf[validdf['y'] == '0']
positive_test = testdf[testdf['y'] == '1']
negative_test = testdf[testdf['y'] == '0']
print(positive_train.columns)
print(positive_train['y'].value_counts())
print("dtype stuff")
print("------------------------------------------")
print(positive_train)
#random sampling from scaffold sampling
pos_train = positive_train.sample(n=50, random_state=42)
neg_train = negative_train.sample(n=50, random_state=42)
print(pos_train)
pos_valid=positive_valid.sample(n=50, random_state=42)
neg_valid=negative_valid.sample(n=50, random_state=42)

pos_test = positive_test.sample(n=50, random_state=42)
neg_test = negative_test.sample(n=50, random_state=42)

#adding neg_test below pos_test
Xtestsampled = pd.concat([pos_test, neg_test], ignore_index=True)
Xtrainsampled = pd.concat([pos_train, neg_train], ignore_index=True)
Xvalsampled = pd.concat([pos_valid, neg_valid], ignore_index=True)
ytestsampled = ([1] *50) + ([0] * 50)
ytrainsampled = ([1] *50) + ([0] *50)
yvalsampled = ([1]*50) + ([0] *50)
print("columns before dropping")
print(Xtrainsampled.columns)
#drop the ids column
Xtrainsampled = Xtrainsampled.drop(columns=['ids'])
Xtestsampled = Xtestsampled.drop(columns=['ids'])
Xvalsampled = Xvalsampled.drop(columns=['ids'])
Xtrainsampled = Xtrainsampled.drop(columns=['y'])
Xtestsampled = Xtestsampled.drop(columns=['y'])
Xvalsampled = Xvalsampled.drop(columns=['y'])
Xtrainsampled = Xtrainsampled.drop(columns=['w'])
Xtestsampled = Xtestsampled.drop(columns=['w'])
Xvalsampled = Xvalsampled.drop(columns=['w'])
print("columns")
print(Xtrainsampled.columns)
print(Xtrainsampled)
print(Xtrainsampled.describe)
print(Xtrainsampled.shape)
#min max scale for angle encoding
scalar = MinMaxScaler(feature_range=(-np.pi, np.pi)) #learns min and max and then applies it to scale the data to min and max
X_train_scaled = scalar.fit_transform(Xtrainsampled)
X_val_scaled = scalar.transform(Xvalsampled)
X_test_scaled = scalar.transform(Xtestsampled)
print("after scaling")
print(X_train_scaled)
#print(X_train_scaled.describe)
#print(X_train_scaled.shape)
#principal component analysis
pca = PCA(n_components=6)

X_trainq = pca.fit_transform(X_train_scaled)
X_valq  = pca.transform(X_val_scaled)
X_testq  = pca.transform(X_test_scaled)
print(X_trainq)
#transformer = Normalizer(norm='l2').fit(X_trainq)
#X_trainq = transformer.transform(X_trainq)
#transformer = Normalizer(norm='l2').fit(X_testq)
#X_testq = transformer.transform(X_testq)

X_train = X_trainq
X_test = X_testq
X_val = X_valq
y_train = ytrainsampled
y_test = ytestsampled
y_val = yvalsampled

#regular data for classical model
print("train")
print(X_train)
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_val_tf = tf.convert_to_tensor(X_val, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype = tf.float32)
y_val_tf = tf.convert_to_tensor(y_val, dtype = tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype = tf.float32)

X_trainq_tf = tf.convert_to_tensor(X_trainq, dtype = tf.float32)
X_valq_tf = tf.convert_to_tensor(X_valq, dtype = tf.float32)
X_testq_tf = tf.convert_to_tensor(X_testq, dtype = tf.float32)
y_trainq_tf = tf.convert_to_tensor(ytrainsampled, dtype = tf.float32)
y_valq_tf = tf.convert_to_tensor(yvalsampled, dtype = tf.float32)
y_testq_tf = tf.convert_to_tensor(ytestsampled, dtype = tf.float32)
print(y_train)

print(y_train_tf)

np.save("../data/processed/X_train_tf.npy", X_train_tf)
np.save("../data/processed/y_train_tf.npy", y_train_tf)
np.save("../data/processed/X_val_tf.npy", X_val_tf)
np.save("../data/processed/y_val_tf.npy", y_val_tf)
np.save("../data/processed/X_test_tf.npy", X_test_tf)
np.save("../data/processed/y_test_tf.npy", y_test_tf)

np.save("../data/processed/X_trainq.npy", X_trainq_tf)
np.save("../data/processed/X_valq.npy", X_valq_tf)
np.save("../data/processed/X_testq.npy", X_testq_tf)
np.save("../data/processed/y_trainq_tf.npy", y_trainq_tf)
np.save("../data/processed/y_valq_tf.npy", y_valq_tf)
np.save("../data/processed/y_testq_tf.npy", y_testq_tf)

print("--------------------DONE------------------------")


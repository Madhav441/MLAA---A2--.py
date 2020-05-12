#!/usr/bin/env python
# coding: utf-8

# In[153]:


import sklearn
from sklearn import datasets
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from functools import partial
import tensorflow as tf


# In[154]:


mist_data = fetch_openml('mnist_784', version=1, return_X_y=True)
olivetti_faces = sklearn.datasets.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)


# In[155]:


index_list = []
while len(index_list) <5000:
    index_list.append(random.randint(0,70000))
    index_list = list(set(index_list))


# In[156]:


Sample_mist_data = [mist_data[0][index_list], mist_data[1][index_list]]


# ### Generating Artificial Datasets

# In[157]:


swiss_roll_dataset = []
while len(swiss_roll_dataset) < 5000:
    Pi = np.random.uniform(0,1)
    Qi = np.random.uniform(0,1)
    Ti = ((3*(3.14)/2))*(1+(2*Pi))
    Xi = [Ti*math.cos(Ti),Ti*math.sin(Ti),30*(Qi)]
    swiss_roll_dataset.append(Xi)


# In[158]:


broken_swiss_roll_dataset=[]
while len(broken_swiss_roll_dataset) < 5000:
    #Pi =  np.random.uniform(0.4,0.6)
    Pi =  np.random.uniform(0,2/5) if random.randint(0,1) is 1 else np.random.uniform(3/5,1)
    Qi =  np.random.uniform(0,1) if random.randint(0,1) is 1 else np.random.uniform(0,1)
    Ti = ((3*(3.14)/2))*(1+(2*Pi))
    Xi = [Ti*math.cos(Ti),Ti*math.sin(Ti),30*(Qi)]
    broken_swiss_roll_dataset.append(Xi)


# In[159]:


helix_dataset = []
while len(helix_dataset) < 5000:
    Pi = np.random.uniform(0,1)
    Qi = np.random.uniform(0,1)
    Ti = ((3*(3.14)/2))*(1+(2*Pi))
    Xi = [(2+math.cos(8*Ti))*math.cos(Ti),(2+math.cos(8*Ti))*math.sin(Ti), math.sin(8*Ti)]
    helix_dataset.append(Xi)


# ### Generating Visualisations for Artificial Datasets

# In[211]:


X1 = np.array(swiss_roll_dataset)
Y1 = np.array(broken_swiss_roll_dataset)
Z1 = np.array(helix_dataset)
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X1)
label = ward.labels_


# In[161]:


#https://docs.w3cub.com/scikit_learn/auto_examples/cluster/plot_ward_structured_vs_unstructured/#sphx-glr-auto-examples-cluster-plot-ward-structured-vs-unstructured-py
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X1)
label = ward.labels_
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -75)
for l in np.unique(label):
    ax.scatter(X1[label == l, 0], X1[label == l, 2], X1[label == l, 1],
               color=plt.cm.jet(np.float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('swiss roll')


# In[162]:


#def functionname(args):
#    stufff
#    return eg. labels


# In[163]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Y1)
label = ward.labels_
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(Y1[label == l, 0], Y1[label == l, 2], Y1[label == l, 1],
               color=plt.cm.jet(np.float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('broken swiss roll')


# In[164]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Z1)
label = ward.labels_
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -75)
for l in np.unique(label):
    ax.scatter(Z1[label == l, 0], Z1[label == l, 1], Z1[label == l, 2],
               color=plt.cm.jet(np.float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('Helix')


# ## Train Test Split of Datasets

# ### PCA

# ### Natural Datasets

# #### Olivetti Faces

# In[209]:


X = olivetti_faces.data
y = olivetti_faces.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pca = PCA(n_components = 8)
PCAtrain = pca.fit_transform(X_train)
PCAtest = pca.transform(X_test)


# In[210]:


# does not match ORL set from paper, dimensions of images are 64x64 whereas ORL set in paper is 112x92
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(PCAtrain,y_train)

print("Error of PCA- olivetti faces = ", str(1 - neigh.score(PCAtest, y_test)))


# #### MNIST Data

# In[207]:


print(Sample_mist_data[0].shape)
X = Sample_mist_data[0]
y = Sample_mist_data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
pca= PCA(n_components = 20)
PCAtrain = pca.fit_transform(X_train)
PCAtest = pca.transform(X_test)
print(PCAtrain.shape)
print(PCAtest.shape)


# In[208]:


neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(PCAtrain,y_train)

print("Error of PCA- MNIST data = ", str(1 - neigh.score(PCAtest, y_test)))


# ### Artificial Datasets

# #### swiss roll

# In[212]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)
pca= PCA(n_components = 2) 
PCAtrain = pca.fit_transform(X_train)
PCAtest = pca.transform(X_test)
print(PCAtrain.shape)
print(PCAtest.shape)

neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(PCAtrain,y_train)

print("Error of PCA- swiss roll data = ", str(1 - neigh.score(PCAtest, y_test)))


# #### broken swiss roll

# In[205]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Y1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(Y1, y, test_size=0.33, random_state=42)
pca= PCA(n_components = 2)
PCAtrain = pca.fit_transform(X_train)
PCAtest = pca.transform(X_test)
print(PCAtrain.shape)
print(PCAtest.shape)

neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(PCAtrain,y_train)

print("Error of PCA- Broken Swiss data = ", str(1 - neigh.score(PCAtest, y_test)))


# #### helix dataset

# In[204]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Z1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(Z1, y, test_size=0.33, random_state=42)
pca= PCA(n_components = 1)
PCAtrain = pca.fit_transform(X_train)
PCAtest = pca.transform(X_test)
print(PCAtrain.shape)
print(PCAtest.shape)

neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(PCAtrain,y_train)

print("Error of PCA- Helix data = ", str(1 - neigh.score(PCAtest, y_test)))


# ### KPCA

# ### Natural Datasets

# #### Olivetti faces

# In[172]:


X = olivetti_faces.data
y = olivetti_faces.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
transformer = KernelPCA(n_components=8, kernel='poly', degree = 5)
KPCAtrain = transformer.fit_transform(X_train)
KPCAtest = transformer.transform(X_test)


# In[173]:


neigh = KNeighborsClassifier(n_neighbors =1)
neigh.fit(KPCAtrain,y_train)
KNeighborsClassifier(...)
print("Error of KPCA- olivetti faces = ", str(1 - neigh.score(KPCAtest, y_test)))


# #### MNIST Data

# In[174]:


X = Sample_mist_data[0]
y = Sample_mist_data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
transformer = KernelPCA(n_components=20, kernel='poly', degree = 5)
KPCAtrain = transformer.fit_transform(X_train)
KPCAtest = transformer.transform(X_test)


# In[175]:


neigh = KNeighborsClassifier(n_neighbors =1)
neigh.fit(KPCAtrain,y_train)
KNeighborsClassifier(...)
print("Error of KPCA NMIST = ", str(1 - neigh.score(KPCAtest, y_test)))


# ### Artificial Datasets

# #### Swiss Roll

# In[213]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)

transformer = KernelPCA(n_components=2, kernel='poly', degree = 5)
KPCAtrain = transformer.fit_transform(X_train)
KPCAtest = transformer.transform(X_test)

neigh = KNeighborsClassifier(n_neighbors =1)
neigh.fit(KPCAtrain,y_train)
KNeighborsClassifier(...)
print("Error of KPCA Swiss Roll = ", str(1 - neigh.score(KPCAtest, y_test)))


# #### Broken Swiss Roll

# In[177]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Y1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(Y1, y, test_size=0.33, random_state=42)

transformer = KernelPCA(n_components=2, kernel='poly', degree = 5)
KPCAtrain = transformer.fit_transform(X_train)
KPCAtest = transformer.transform(X_test)

neigh = KNeighborsClassifier(n_neighbors =1)
neigh.fit(KPCAtrain,y_train)
KNeighborsClassifier(...)
print("Error of KPCA Broken Swiss Roll = ", str(1 - neigh.score(KPCAtest, y_test)))


# #### Helix Dataset

# In[178]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Z1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(Z1, y, test_size=0.33, random_state=42)

transformer = KernelPCA(n_components=1, kernel='poly', degree = 5)
KPCAtrain = transformer.fit_transform(X_train)
KPCAtest = transformer.transform(X_test)

neigh = KNeighborsClassifier(n_neighbors =1)
neigh.fit(KPCAtrain,y_train)
KNeighborsClassifier(...)
print("Error of KPCA Helix = ", str(1 - neigh.score(KPCAtest, y_test)))


# ### LLE

# ### Natural Datasets

# #### Olivetti Faces

# In[179]:


X = olivetti_faces.data
y = olivetti_faces.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
embedding = LocallyLinearEmbedding(n_components=8)
LLEtrain = embedding.fit_transform(X_train)
LLEtest = embedding.transform(X_test)


# In[180]:


neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(LLEtrain,y_train)

print("Error of LLE- Olivetti Faces = ", str(1 - neigh.score(LLEtest, y_test)))


# #### NMIST Data

# In[181]:


X = Sample_mist_data[0]
y = Sample_mist_data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
embedding = LocallyLinearEmbedding(n_components=20)
LLEtrain = embedding.fit_transform(X_train)
LLEtest = embedding.transform(X_test)


# In[182]:


neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(LLEtrain,y_train)

print("Error of LLE- MNIST data = ", str(1 - neigh.score(LLEtest, y_test)))


# ### Artificial DataSets

# #### Swiss Roll

# In[183]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)

embedding = LocallyLinearEmbedding(n_components=2)
LLEtrain = embedding.fit_transform(X_train)
LLEtest = embedding.transform(X_test)

neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(LLEtrain,y_train)

print("Error of LLE- Swiss Roll data = ", str(1 - neigh.score(LLEtest, y_test)))


# #### Broken Swiss Roll

# In[184]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Y1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(Y1, y, test_size=0.33, random_state=42)

embedding = LocallyLinearEmbedding(n_components=2)
LLEtrain = embedding.fit_transform(X_train)
LLEtest = embedding.transform(X_test)

neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(LLEtrain,y_train)

print("Error of LLE- Broken Swiss Roll data = ", str(1 - neigh.score(LLEtest, y_test)))


# #### Helix Dataset

# In[185]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Z1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(Z1, y, test_size=0.33, random_state=42)

embedding = LocallyLinearEmbedding(n_components=1)
LLEtrain = embedding.fit_transform(X_train)
LLEtest = embedding.transform(X_test)

neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(LLEtrain,y_train)

print("Error of LLE- Broken Swiss Roll data = ", str(1 - neigh.score(LLEtest, y_test)))


# ### AUTOENCODER

# ### Natural Dataset

# #### NMIST Data

# In[186]:


X = Sample_mist_data[0]
y = Sample_mist_data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

n_inputs = 28 * 28 # for MNIST
n_hidden1 = 200
n_hidden2 = 20 # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.sigmoid,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2) # codings
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# In[187]:


n_epochs = 1000
codings = hidden2 # the output of the hidden layer provides the codings
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        training_op.run(feed_dict={X: X_train}) # no labels (unsupervised)
    codings_val_test  = codings.eval(feed_dict={X: X_test })
    codings_val_train = codings.eval(feed_dict={X: X_train})  
    


# In[188]:


neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(codings_val_train, y_train)

print("Error of Autoencoder - MNIST data = ", str(1 - neigh.score(codings_val_test, y_test)))


# #### Olivetti Faces

# In[189]:


# Ref: "Hands-On Machine Learning with SciKit-Learn and Tensorflow Ch. 15 


X = olivetti_faces.data
y = olivetti_faces.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


n_inputs = 4096 # for MNIST
n_hidden1 = 400
n_hidden2 = 8 # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.sigmoid,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2) # codings
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# In[190]:


n_epochs = 1000
codings = hidden2 # the output of the hidden layer provides the codings
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        training_op.run(feed_dict = {X: X_train}) # no labels (unsupervised)
    codings_val_test  = codings.eval(feed_dict = {X: X_test })
    codings_val_train = codings.eval(feed_dict = {X: X_train})  


# In[191]:


neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(codings_val_train, y_train)

print("Error of Autoencoder - Olivetti Faces = ", str( 1 - neigh.score(codings_val_test, y_test)))


# ### Artificial Dataset

# #### Swiss Roll

# In[193]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)

n_inputs = 3 # for MNIST
n_hidden1 = 2
n_hidden2 = 2 # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.sigmoid,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2) # codings
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


n_epochs = 1000
codings = hidden2 # the output of the hidden layer provides the codings
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        training_op.run(feed_dict = {X: X_train}) # no labels (unsupervised)
    codings_val_test  = codings.eval(feed_dict = {X: X_test })
    codings_val_train = codings.eval(feed_dict = {X: X_train})  
    
    neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(codings_val_train, y_train)

print("Error of Autoencoder - Swiss Roll = ", str( 1 - neigh.score(codings_val_test, y_test)))


# #### Broken Swiss roll

# In[194]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Y1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(Y1, y, test_size=0.33, random_state=42)
n_inputs = 3# for MNIST
n_hidden1 = 2
n_hidden2 = 2 # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.sigmoid,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2) # codings
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 1000
codings = hidden2 # the output of the hidden layer provides the codings
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        training_op.run(feed_dict = {X: X_train}) # no labels (unsupervised)
    codings_val_test  = codings.eval(feed_dict = {X: X_test })
    codings_val_train = codings.eval(feed_dict = {X: X_train})  
    
    neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(codings_val_train, y_train)

print("Error of Autoencoder - Broken Swiss Roll = ", str( 1 - neigh.score(codings_val_test, y_test)))


# #### Helix Dataset

# In[195]:


ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Z1)
label = ward.labels_
y = label
X_train, X_test, y_train, y_test = train_test_split(Z1, y, test_size=0.33, random_state=42)

n_inputs = 3 # for MNIST
n_hidden1 = 2
n_hidden2 = 1 # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.sigmoid,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2) # codings
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 1000
codings = hidden2 # the output of the hidden layer provides the codings
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        training_op.run(feed_dict = {X: X_train}) # no labels (unsupervised)
    codings_val_test  = codings.eval(feed_dict = {X: X_test })
    codings_val_train = codings.eval(feed_dict = {X: X_train})  
    
    neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(codings_val_train, y_train)

print("Error of Autoencoder - Helix = ", str( 1 - neigh.score(codings_val_test, y_test)))


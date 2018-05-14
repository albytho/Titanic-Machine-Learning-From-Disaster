# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##############################################################################
# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset[['Embarked']] = dataset[['Embarked']].fillna(value='C')
dataset[['Age']] = dataset[['Age']].fillna(value=25)

X_train = dataset.iloc[:, [2,4,5,6,7,9,11]].values
y_train = dataset.iloc[:, 1].values


testset = pd.read_csv('test.csv')
testset[['Embarked']] = testset[['Embarked']].fillna(value='C')
testset[['Age']] = testset[['Age']].fillna(value=25)
testset[['Fare']] = testset[['Fare']].fillna(value=15)

X_test = testset.iloc[:, [1,3,4,5,6,8,10]].values
X_test_ids = testset.iloc[:, [0]].values
##############################################################################


##############################################################################
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Convert Gender string into actual number
labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder_X_1.fit_transform(X_test[:, 1])

#Convert Depart string into actual numbers
labelencoder_X_6 = LabelEncoder()
X_train[:, 6] = labelencoder_X_6.fit_transform(X_train[:, 6])
X_test[:, 6] = labelencoder_X_6.fit_transform(X_test[:, 6])


# Add dummy variables to Depart column.  This is better than having a column
# that consists of 0,1,2,etc
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

#Remove one of the country dummy variables so that we will have 2
#This makes sense, if the other two varibles are 0, then the model can
#assume that the third one would be 1.  Typically, you should have m-1 
#categories
X_train = X_train[:,1:]
X_test = X_test[:,1:]
##############################################################################

##############################################################################
# Feature Scaling
# Normalize all the input data
from sklearn.preprocessing import StandardScaler

#You only need to do transform after doing fit_transform once
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
##############################################################################

##############################################################################
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
#8+1/2 = 5 hidden layer nodes
#'uniform' tells the layer to initialize the weights randomly and for them
#to be close to 0
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 8))

#Add a second hidden layer
#This already knows how many inputs to expect, so you don't need input_dim
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

#Add the output layer
#output_dim is 1 since we only have one output
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 16, nb_epoch = 200)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Convert floating points to binaries
index = 0
for row in y_pred:
    if row[0] < 0.5:
        y_pred[index] = 0
    else:
        y_pred[index] = 1
    index = index + 1

for row in y_pred:
    print(int(row[0]))
    
#Evaluating and runing the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, nb_epoch = 200)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

#Tuning the ANN
#The arguments that we are goint to tune will be put seperatly into the gridSearch object
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer, droprate):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dropout(p = droprate))
    classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = droprate))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25],
              'nb_epoch': [100],
              'droprate': [0.0,0.1],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



##############################################################################
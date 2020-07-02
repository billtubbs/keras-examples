from keras.models import Sequential
from keras.layers import Dense

import numpy
import time

# fix random seed for reproducability
numpy.random.seed(7)

# Use Pima Indians onset of diabetes dataset
# from the UCI Machine Learning repository

# This is a binary classification problem:
# (onset of diabetes as 1 or not as 0)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

duration = time.time() - start_time
print("model trained in %d secs." % duration)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
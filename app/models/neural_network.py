import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class SymptomDiseaseModel:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=input_shape, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_input):
        return self.model.predict(X_input)

    def save_model(self, filepath='saved_model.h5'):
        self.model.save(filepath)

    def load_model(self, filepath='saved_model.h5'):
        self.model = tf.keras.models.load_model(filepath)

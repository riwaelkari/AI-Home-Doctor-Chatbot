import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class SymptomDiseaseModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_shape=(132,)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(41, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  

    def train(self, X_train, y_train, epochs=3, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_input):
        return self.model.predict(X_input)

    def save_model(self, filepath='models/saved_model.h5'):
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Loads a trained model from a file.
        """
        self.model = tf.keras.models.load_model(filepath)

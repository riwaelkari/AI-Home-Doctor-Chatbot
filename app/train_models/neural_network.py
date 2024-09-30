import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
import numpy as np


class SymptomDiseaseModel:

    def __init__(self, X_train):
        model = Sequential()

        input_shape = X_train.shape[1]


        # Input layer
        model.add(Input(shape=(input_shape,)))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())  # Batch Normalization
        model.add(Dropout(0.5))  # Dropout layer



        # Hidden layers
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))


        # Define number of output classes
        num_classes = 41
        print(f"Number of classes: {num_classes}")


        # Output layer
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Assign the model to the instance attribute
        self.model = model  # <-- This line assigns the model to the instance
  




    def train(self, X_train, y_train, epochs=2, batch_size=32, validation_split=0.2):
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        

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
  
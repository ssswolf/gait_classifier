import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import time
import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

class ModelTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, model_type='dnn'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        self.scaler = None
        self.model = None

    def normalize_data(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        scaler_file = f"assets/normalizer_{timestamp}.pkl"
        os.makedirs("assets", exist_ok=True)
        joblib.dump(self.scaler, scaler_file)
        print(f'\nSaving Scaler to {scaler_file}\n')

    def train(self):
        print('\nTraining Model...')
        self.normalize_data()

        if self.model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=False)
            self.model.fit(self.X_train, self.y_train)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
        elif self.model_type == 'dnn':
            self.model = self.build_dnn_model()
            self.model.summary()
            self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=32, validation_data=(self.X_val, self.y_val))
        else:
            raise ValueError("Model type not recognized!")
        
        return self.model

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, batch_size=32, verbose=0)
        print("\nTest Loss:     ", test_loss)
        print("Test Accuracy: ", test_acc)
        return test_acc

    def build_dnn_model(self):
        num_classes = len(np.unique(self.y_train))

        encoder_input = keras.Input(shape=(self.X_train.shape[1],))
        encoder_input_1 = keras.layers.Dense(64, activation="tanh")(encoder_input)
        encoder_input_2 = keras.layers.Dense(16, activation="tanh")(encoder_input_1)
        encoder_output = keras.layers.Dense(3, activation="tanh")(encoder_input_2)  

        decoder_input_1 = keras.layers.Dense(16, activation="tanh")(encoder_output)
        decoder_input_2 = keras.layers.Dense(64, activation="tanh")(decoder_input_1)
        decoder_output = keras.layers.Dense(num_classes, activation="softmax")(decoder_input_2)  

        model = keras.Model(encoder_input, decoder_output)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

        return model
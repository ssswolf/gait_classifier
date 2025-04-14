import warnings
warnings.filterwarnings("ignore")

import time
import os
import joblib
from sklearn.model_selection import train_test_split
from models.model_trainer import ModelTrainer


class TrainingSession:
    def __init__(self, data_loader, model_type):
        self.data_loader = data_loader
        self.model_type = model_type
        self.model_trainer = None  
        self.model = None          
    
    def run(self):
        X_ft, X_ts, X_mag, y = self.data_loader.load_data()
        X_train, X_temp, y_train, y_temp = train_test_split(X_ft, y, test_size=0.2, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
        self.model_trainer = ModelTrainer(X_train, y_train, X_val, y_val, X_test, y_test, model_type=self.model_type)
        self.model = self.model_trainer.train()
        self.model_trainer.evaluate()
        self.save_model()

    def save_model(self):
        save_loc = None
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("assets", exist_ok=True)
        if self.model_type == 'dnn':
            self.model.save(f"assets/dnn_model_{timestamp}.h5")
            save_loc = f"assets/dnn_model_{timestamp}.h5"
        elif self.model_type == 'svm':
            joblib.dump(self.model, f"assets/svm_model_{timestamp}.pkl")
            save_loc = f"assets/svm_model_{timestamp}.pkl"
        elif self.model_type == 'rf':
            joblib.dump(self.model, f"assets/rf_model_{timestamp}.pkl")
            save_loc = f"assets/rf_model_{timestamp}.pkl"
        else:
            print("Unknown model type. Model not saved.")
        if save_loc is not None:
            print(f"\nSaving Model to {save_loc}")
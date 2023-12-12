import tensorflow as tf
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import  seaborn as sns
import numpy as np
import pandas as pd
import flwr as fl
import time
np.random.seed(42)
tf.random.set_seed(42)
# data loading this data is considered as data of client
data = pd.read_csv('test_USA.csv')
data = data[data['No. Spans'].isin([57,6])]
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# splitting the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# =========converting to array===================
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# =========reshaping=============================
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# ===========scaling================
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
label_scaler = MinMaxScaler()
label_scaler.fit(y_train)
y_train = label_scaler.transform(y_train)
y_test = label_scaler.transform(y_test)
# model initialization like initializing its parameters only
model = tf.keras.Sequential([
    tf.keras.layers.Dense(305, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_squared_error'])


# =================================Starting Federated Learning==================

# here model which was previously initialized will train on user data

class ONTClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        t0 = time.time()
        history = model.fit(X_train, y_train, epochs=100, batch_size=X_train.shape[0], validation_data=(X_test, y_test), verbose=2)
        print("Training-Time for Client-1: ", time.time() - t0)
        return model.get_weights(), len(X_train), {"Train_loss": history.history['loss'][-1]}

    def evaluate(self, parameters, config):
        # model.set_weights(parameters)
        t0 = time.time()
        loss, accuracy = model.evaluate(X_test, y_test)
        print("EVALUATION-TIME for Client-2: ", time.time() - t0)
        # =============PLOTTING=======================
        predictions = model.predict(X_test)
        mean_y_val=y_test.mean()
        mean_predictions=predictions.mean()
        sns.kdeplot(y_test, palette=['red'], fill=True, label='ORIGINAL-VALUES',legend=False)
        sns.kdeplot(predictions, palette=['blue'], fill=True, label='PREDICTED-VALUES',legend=False)
        plt.axvline(mean_y_val, color='black', linewidth=2, label='Mean ORIGINAL')
        plt.axvline(mean_predictions, color='black', linestyle='dashed', linewidth=2, label='Mean PREDICTED')
        plt.xlabel('GSNR_1')
        plt.ylabel('Probability Density')
        plt.title('CLIENT-2')
        plt.legend()
        plt.show()
        residuals = y_test - predictions
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=residuals)
        plt.title('CLIENT-2')
        plt.xlabel('Residuals')
        plt.show()
        return loss, len(X_test), {"accuracy": accuracy}


# start flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                             client=ONTClient())

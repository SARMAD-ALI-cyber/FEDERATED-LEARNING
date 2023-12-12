import time
import seaborn as sns
import flwr as fl
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
np.random.seed(42)
tf.random.set_seed(42)


# Client 5 is used when we have to give each client 1 span
# Configuration dictionary
# See this documentation for clearance I have used custom config dict
# this is an easy way to control hyperparameters of clients, we access this dict on client side fit function
# there is also on_evaluate_config_fn argument in strategy like for fit it is done in same manner.
# https://flower.dev/docs/framework/how-to-configure-clients.html

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    data = pd.read_csv('test_USA.csv')
    data = data[data['No. Spans'].isin([11,23])]
    x_val = data.iloc[:, :-1]
    y_val = data.iloc[:, -1]
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    y_val = y_val.reshape(-1, 1)
    # ===========scaling================
    scaler = StandardScaler()
    scaler.fit(x_val)
    x_val = scaler.transform(x_val)
    label_scaler = MinMaxScaler()
    label_scaler.fit(y_val)
    y_val = label_scaler.transform(y_val)

    # The `evaluate` function will be called after every round
    def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        t0 = time.time()
        loss, accuracy = model.evaluate(x_val, y_val)
        print("Evaluation-Time for 9000 examples: ", time.time() - t0)
        predictions = model.predict(x_val)
        mean_y_val=y_val.mean()
        mean_predictions=predictions.mean()
        sns.kdeplot(y_val, palette=['red'], fill=True, label='ORIGINAL-VALUES',legend=False)
        sns.kdeplot(predictions, palette=['blue'], fill=True, label='PREDICTED-VALUES',legend=False)
        plt.axvline(mean_y_val, color='black', linewidth=2, label='Mean ORIGINAL')
        plt.axvline(mean_predictions, color='black', linestyle='dashed', linewidth=2, label='Mean PREDICTED')
        plt.xlabel('GSNR_1')
        plt.ylabel('Probability Density')
        plt.title('GLOBAL-MODEL')
        plt.legend()
        plt.show()
        residuals = y_val - predictions
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=residuals)
        plt.title('GLOBAL MODEL')
        plt.xlabel('Residuals')
        plt.show()
        return loss, {"accuracy": accuracy}

    return evaluate


# model initialization like initializing its parameters only
model = tf.keras.Sequential([
    tf.keras.layers.Dense(305, activation='relu', input_shape=(305,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_squared_error'])


def fit_config(server_round: 2):
    config = {
        "current_round": server_round,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "local_epochs": 40,
    }
    return config


# Aggregation strategy
strategy = fl.server.strategy.FedAvg(
    min_available_clients=4,
    min_evaluate_clients=4,
    min_fit_clients=4,
    evaluate_fn=get_evaluate_fn(model),
    on_fit_config_fn=fit_config
)

# star flower server
server = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)

# 0.0.0.0 is used by server to listen to any ip address 8080 is a port number
# here num_rounds means how many times you want to repeat a step
# like if I want to train for 40 epoch and num_round is 3 it wil train 3 times on 40 epochs

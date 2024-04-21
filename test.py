# Example dataset
import sqlite3

import numpy as np
import pandas as pd
from pygad import pygad

# connection = sqlite3.connect('elements.db')
#
# # SQL query to select all columns from the table
# query = "SELECT * FROM discriminations"
#
# df = pd.read_sql_query(query, connection)

# %%
import numpy
import pygad.nn
import pandas
from sklearn.preprocessing import StandardScaler

data = numpy.array(pandas.read_csv("fish.csv"))
xscaler = StandardScaler()
data_inputs = xscaler.fit_transform(numpy.asarray(data[:, 2:], dtype=numpy.float32))

y_scaler = StandardScaler()
data_outputs = y_scaler.fit_transform(data[:, 1].reshape(-1, 1)).flatten()

num_inputs = data_inputs.shape[1]
num_outputs = 1
HL1_neurons = 2

input_layer = pygad.nn.InputLayer(num_inputs)
hidden_layer1 = pygad.nn.DenseLayer(num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="None")
output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="None")

pygad.nn.train(num_epochs=100,
               last_layer=output_layer,
               data_inputs=data_inputs,
               data_outputs=data_outputs,
               learning_rate=0.001,
               problem_type="regression")

predictions = pygad.nn.predict(last_layer=output_layer,
                               data_inputs=data_inputs,
                               problem_type="regression")

abs_error = numpy.mean(numpy.abs(predictions - data_outputs))
print(f"Absolute error : {abs_error}.")

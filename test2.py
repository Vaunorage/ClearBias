# %%
import numpy as np
import torch
import pygad.torchga
import pygad

import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler

connection = sqlite3.connect('elements.db')

# SQL query to select all columns from the table
query = "SELECT * FROM discriminations"

df = pd.read_sql_query(query, connection)

# %%
xscaler = StandardScaler()
yscaler = StandardScaler()

data_inputs = xscaler.fit_transform(df[['Gender', 'Income', 'Expenditure', 'Age']].to_numpy())
data_inputs = torch.tensor(data_inputs).to(torch.float32)

data_outputs = yscaler.fit_transform(df[['outcome']].to_numpy())
data_outputs = torch.tensor(data_outputs).to(torch.float32)


# %%

def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    solution = solution.astype('float32')
    predictions = pygad.torchga.predict(model=model, solution=solution, data=data_inputs)
    mse = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

    return 1.0 / mse

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness = {ga_instance.best_solution()[1]}")

# Adjusted model architecture
model = torch.nn.Sequential(
    torch.nn.Linear(4, 10),  # More neurons in the first layer
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),  # Additional hidden layer
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)   # Assuming a single output
)
print(model)

torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=50)  # Increased population size
loss_function = torch.nn.MSELoss()  # Changed to MSE

ga_instance = pygad.GA(
    num_generations=700,  # Increased number of generations
    num_parents_mating=10,  # Increased mating pool size
    initial_population=torch_ga.population_weights,
    fitness_func=fitness_func,
    on_generation=on_generation
)

ga_instance.run()
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

predictions = pygad.torchga.predict(model=model, solution=solution, data=data_inputs)
print(f"Predictions : \n{np.concatenate([predictions.detach().numpy(), data_outputs], axis=1)}")

mse_error = loss_function(predictions, data_outputs)
print(f"Mean Squared Error : {mse_error.detach().numpy()}")

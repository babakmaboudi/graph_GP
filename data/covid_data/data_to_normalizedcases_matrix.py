from datasets import covid_adjacencies_graph_dataset

graph, temp_data = covid_adjacencies_graph_dataset()

dates = sorted(temp_data.keys())
states = sorted(temp_data[dates[0]].keys())

import torch

cases_matrix = torch.tensor([
     [temp_data[date][state]["cases_normalized"] for date in dates]
     for state in states
])
print(cases_matrix.shape)  # should be (51, number_of_dates)

print("Shape:", cases_matrix.shape)
print("Min:", cases_matrix.min())
print("Max:", cases_matrix.max())

import pickle

with open("y_cases_normalized.pkl", "wb") as f:
    pickle.dump(cases_matrix, f)
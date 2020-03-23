# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %%
import numpy as np

# %%
#                           Forward Propagation
# =============================================================================

input_data = np.array([2, 3])

weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]),
           'output': np.array([2, -1]),
           }

node0_value = (input_data * weights['node_0']).sum()

node1_value = (input_data * weights['node_1']).sum()

hidden_layer = np.array([node0_value, node1_value])

# predictions
output = (hidden_layer * weights['output']).sum()

# %%
#                   Rectified Linear Activation Function
# =============================================================================
"""
An "activation function" is a function applied at each node. It converts
the node's input into some output.

The rectified linear activation function (called ReLU) has been shown to
lead to very high-performance networks. This function takes a single number
as an input, returning 0 if the input is negative, and the input if the input
is positive
"""

# inputs are: num of accounts, and num of children
input_data = np.array([3, 5])

weights = {
    'node_0': np.array([6, 4]),
    'node_1': np.array([2, -3]),
    'output': np.array([1, 5])
}


def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)

    # Return the value just calculated
    return output


# DOT PRODUCT between input_data and weights
# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)
# same output
np.dot(input_data, weights['node_0'])

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)


# %%
#           Applying the network to many observations/rows of data
# =============================================================================
# Define predict_with_network()
def predict_with_network(input_data_row, weights):
    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    # Return model output
    return (model_output)


input_data = [np.array([3, 5]), np.array([1, -1]), np.array([0, 0]), np.array([8, 4])]
# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)

# %%

# =============================================================================


# %%

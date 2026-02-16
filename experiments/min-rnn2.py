"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import torch

# Check if PyTorch is running on macOS and if the M1 chip is available
if torch._C._get_tracing_state() and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Data Input/Output
data = open('input.txt', 'r').read() # The file should be a simple plain text file
characters = list(set(data))
data_size, vocabulary_size = len(data), len(characters)
print('Data has %d characters, %d unique.' % (data_size, vocabulary_size))
character_to_index_mapping = { character:index for index, character in enumerate(characters) }
index_to_character_mapping = { index:character for index, character in enumerate(characters) }

# Hyperparameters
hidden_layer_neurons = 100 # Size of hidden layer of neurons
sequence_length = 25 # Number of steps to unroll the RNN for
learning_rate = 1e-1

# Model Parameters
input_to_hidden_layer_weights = torch.randn(hidden_layer_neurons, vocabulary_size, device=device)*0.01 # Input to hidden
hidden_to_hidden_layer_weights = torch.randn(hidden_layer_neurons, hidden_layer_neurons, device=device)*0.01 # Hidden to hidden
hidden_to_output_layer_weights = torch.randn(vocabulary_size, hidden_layer_neurons, device=device)*0.01 # Hidden to output
hidden_layer_bias = torch.zeros(hidden_layer_neurons, 1, device=device) # Hidden bias
output_layer_bias = torch.zeros(vocabulary_size, 1, device=device) # Output bias

def calculate_loss(inputs, targets, previous_hidden_state):
  """
  Inputs and targets are both lists of integers.
  previous_hidden_state is Hx1 array of initial hidden state
  Returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = torch.clone(previous_hidden_state)
  loss = 0
  # Forward pass
  for t in range(len(inputs)):
    xs[t] = torch.zeros(vocabulary_size,1, device=device) # Encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = torch.tanh(torch.mm(input_to_hidden_layer_weights, xs[t]) + torch.mm(hidden_to_hidden_layer_weights, hs[t-1]) + hidden_layer_bias) # Hidden state
    ys[t] = torch.mm(hidden_to_output_layer_weights, hs[t]) + output_layer_bias # Unnormalized log probabilities for next characters
    ps[t] = torch.exp(ys[t]) / torch.sum(torch.exp(ys[t])) # Probabilities for next characters
    loss += -torch.log(ps[t][targets[t],0]) # Softmax (cross-entropy loss)
  # Backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = torch.zeros_like(input_to_hidden_layer_weights), torch.zeros_like(hidden_to_hidden_layer_weights), torch.zeros_like(hidden_to_output_layer_weights)
  dbh, dby = torch.zeros_like(hidden_layer_bias), torch.zeros_like(output_layer_bias)
  dhnext = torch.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = torch.clone(ps[t])
    dy[targets[t]] -= 1 # Backprop into y
    dWhy += torch.mm(dy, hs[t].T)
    dby += dy
    dh = torch.mm(hidden_to_output_layer_weights.T, dy) + dhnext # Backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # Backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += torch.mm(dhraw, xs[t].T)
    dWhh += torch.mm(dhraw, hs[t-1].T)
    dhnext = torch.mm(hidden_to_hidden_layer_weights.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    torch.clamp(dparam, -5, 5, out=dparam) # Clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def generate_sample(previous_hidden_state, seed_index, sequence_length):
  """ 
  Sample a sequence of integers from the model 
  previous_hidden_state is memory state, seed_index is seed letter for first time step
  """
  x = torch.zeros(vocabulary_size, 1, device=device)
  x[seed_index] = 1
  indices = []
  for t in range(sequence_length):
    previous_hidden_state = torch.tanh(torch.mm(input_to_hidden_layer_weights, x) + torch.mm(hidden_to_hidden_layer_weights, previous_hidden_state) + hidden_layer_bias)
    y = torch.mm(hidden_to_output_layer_weights, previous_hidden_state) + output_layer_bias
    p = torch.exp(y) / torch.sum(torch.exp(y))
    index = torch.multinomial(p.view(-1), num_samples=1)
    x = torch.zeros(vocabulary_size, 1, device=device)
    x[index] = 1
    indices.append(index)
  return indices

iteration, pointer = 0, 0
memory_input_to_hidden, memory_hidden_to_hidden, memory_hidden_to_output = torch.zeros_like(input_to_hidden_layer_weights), torch.zeros_like(hidden_to_hidden_layer_weights), torch.zeros_like(hidden_to_output_layer_weights)
memory_hidden_bias, memory_output_bias = torch.zeros_like(hidden_layer_bias), torch.zeros_like(output_layer_bias) # Memory variables for Adagrad
smooth_loss = -torch.log(torch.tensor(1.0/vocabulary_size, device=device))*sequence_length # Loss at iteration 0
while True:
  # Prepare inputs (we're sweeping from left to right in steps sequence_length long)
  if pointer+sequence_length+1 >= len(data) or iteration == 0: 
    previous_hidden_state = torch.zeros(hidden_layer_neurons,1, device=device) # Reset RNN memory
    pointer = 0 # Go from start of data
  inputs = [character_to_index_mapping[ch] for ch in data[pointer:pointer+sequence_length]]
  targets = [character_to_index_mapping[ch] for ch in data[pointer+1:pointer+sequence_length+1]]

  # Sample from the model now and then
  if iteration % 100 == 0:
    sample_indices = generate_sample(previous_hidden_state, inputs[0], 200)
    txt = ''.join(index_to_character_mapping[index.item()] for index in sample_indices)
    print('----\n %s \n----' % (txt, ))

  # Forward sequence_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, previous_hidden_state = calculate_loss(inputs, targets, previous_hidden_state)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if iteration % 100 == 0: print('Iteration %d, Loss: %f' % (iteration, smooth_loss)) # Print progress
  
  # Perform parameter update with Adagrad
  for param, dparam, mem in zip([input_to_hidden_layer_weights, hidden_to_hidden_layer_weights, hidden_to_output_layer_weights, hidden_layer_bias, output_layer_bias], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [memory_input_to_hidden, memory_hidden_to_hidden, memory_hidden_to_output, memory_hidden_bias, memory_output_bias]):
    mem += dparam * dparam
    param += -learning_rate * dparam / torch.sqrt(mem + 1e-8) # Adagrad update

  pointer += sequence_length # Move data pointer
  iteration += 1 # Iteration counter 


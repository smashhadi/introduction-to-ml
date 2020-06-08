"""
Assignment to compare Linear units, GRU and LSTMs performance 
Following similar structure to "Classifying Names with a Character-Level RNN" tutorial on PyTorch

Compare the accuracy of the encoder when varying the type of hidden units: linear units, gated recurrent
units (GRUs) and long short term memory (LSTM) units. For linear hidden units, just run the script of the
tutorial as it is. For GRUs and LSTMs, modify the code of the tutorial.
"""

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import time
import math
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
  

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_LSTM, self).__init__()

        self.hidden_size = hidden_size
#        self.output_size = output_size

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        #combined = torch.cat((input, hidden), 1)
        hidden, c = self.lstm(input, (hidden, cell))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
      
    def initCell(self):
        return Variable(torch.ones(1, self.hidden_size))

n_hidden = 128
rnn_m = RNN_LSTM(n_letters, n_hidden, n_categories)
      
def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

  

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor
  
  
criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

  
#LSTM
def train_lstm(category_tensor, line_tensor):
    hidden = rnn_m.initHidden()
    cell = rnn_m.initCell()

    rnn_m.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn_m(line_tensor[i], hidden, cell)

    loss_m = criterion(output, category_tensor)
    loss_m.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn_m.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss_m.data[0]
  
  
n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0

all_losses_LSTM = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

       
#LSTM units
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train_lstm(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses_LSTM.append(current_loss / plot_every)
        current_loss = 0
  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()

plt.plot(all_losses_LSTM)

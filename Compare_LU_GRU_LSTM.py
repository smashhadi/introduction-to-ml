"""
PyTorch based assignment.
Compare Linear units, GRU and LSTMs performance
Following "Classifying Names with a Character-Level RNN" tutorial on PyTorch
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
category_lines_train = {}
category_lines_test = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    total = len(lines)
    train_len = int(len(lines) * 0.8)
    all_lines = random.sample(range(0,total), total)
    train_lines = set(all_lines[0:train_len])
    test_lines = set(all_lines[train_len:])
    category_lines_train[category] = [lines[i] for i in train_lines]
    category_lines_test[category] = [lines[i] for i in test_lines]
    

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
  


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
  
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_GRU, self).__init__()

        self.hidden_size = hidden_size
       
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #combined = torch.cat((input, hidden), 1)
        hidden = self.gru(input, hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
      
rnn_g = RNN_GRU(n_letters, n_hidden, n_categories)

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_LSTM, self).__init__()

        self.hidden_size = hidden_size

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

rnn_m = RNN_LSTM(n_letters, n_hidden, n_categories)
 
  
def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines_train[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor
  
  
def randomTestExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines_test[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor
  
criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

  #GRU
def train_gru(category_tensor, line_tensor):
    hidden = rnn_g.initHidden()

    rnn_g.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn_g(line_tensor[i], hidden)

    loss_g = criterion(output, category_tensor)
    loss_g.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn_g.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss_g.data[0]

  
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
  
  
# Just return an output given a line
def evaluate_lin(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output
  
def evaluate_gru(line_tensor):
    hidden = rnn_g.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn_g(line_tensor[i], hidden)

    return output
  
def evaluate_lstm(line_tensor):
    cell = rnn_m.initCell()
    hidden = rnn_m.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn_m(line_tensor[i], hidden, cell)

    return output

  
n_iters = 80000
n_iters1 = 20000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss_l = 0
current_loss_g = 0
current_loss_lm = 0
all_losses_linear = []
all_losses_GRU = []
all_losses_LSTM = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

#Linear units
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output1, loss1 = train(category_tensor, line_tensor)
    output2, loss2 = train_gru(category_tensor, line_tensor)
    output3, loss3 = train_lstm(category_tensor, line_tensor)
 
    
    if iter % plot_every == 0:
        for i in range(1, n_iters1 + 1):
          category, line, category_tensor, line_tensor = randomTestExample()
          output_l = evaluate_lin(line_tensor)
          testloss_l = criterion(output_l, category_tensor)
          current_loss_l += testloss_l.data[0]
          
          output_g = evaluate_gru(line_tensor)
          testloss_g = criterion(output_g, category_tensor)
          current_loss_g += testloss_g.data[0]
    
          output_lm = evaluate_lstm(line_tensor)
          testloss_lm = criterion(output_lm, category_tensor)
          current_loss_lm += testloss_lm.data[0]
          
        all_losses_linear.append(current_loss_l / n_iters1)#plot_every)
        current_loss_l = 0
        all_losses_GRU.append(current_loss_g / n_iters1)#plot_every)
        current_loss_g = 0
        all_losses_LSTM.append(current_loss_lm / n_iters1)#plot_every)
        current_loss_lm = 0
        
    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
  
    
plt.figure()
plt.plot(all_losses_linear, 'r')
plt.plot(all_losses_GRU, 'g')
plt.plot(all_losses_LSTM, 'b')
plt.xlabel("Num of Iterations")
plt.ylabel("Loss")
plt.title("Test (Validation) Loss")
plt.legend(['Linear', 'GRU', 'LSTM'], loc='upper right')

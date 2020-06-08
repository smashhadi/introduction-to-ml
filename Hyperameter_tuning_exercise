"""
PyTorch based assignment
Following ”Generating names with character-level RNN” tutorial on PyTorch
Hyperparameter tuning
"""

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines_train = {}
category_lines_test = {}
all_categories = []
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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
      
      
class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size

        self.i2h2 = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o2 = nn.Linear(input_size + hidden_size, output_size)
        self.o2o2 = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h2(input_combined)
        output = self.i2o2(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o2(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, category):
        pad = self.hidden_size-18
        temp = Variable(torch.zeros(1, pad))
        padded_cat = torch.cat((category, temp), 1)
        return padded_cat
      
      
class RNN3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN3, self).__init__()
        self.hidden_size = hidden_size

        self.i2h3 = nn.Linear(n_categories + hidden_size, hidden_size)
        self.i2o3 = nn.Linear(n_categories + hidden_size, output_size)
        self.o2o3 = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, hidden):
        input_combined = torch.cat((category, hidden), 1)
        hidden = self.i2h3(input_combined)
        output = self.i2o3(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o3(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
  
      
class RNN4(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN4, self).__init__()
        self.hidden_size = hidden_size

        self.i2h4 = nn.Linear(hidden_size, hidden_size)
        self.i2o4 = nn.Linear(hidden_size, output_size)
        self.o2o4 = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, hidden):
        #input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h4(hidden)
        output = self.i2o4(hidden)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o4(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, category):
        pad = self.hidden_size-18
        temp = Variable(torch.zeros(1, pad))
        padded_cat = torch.cat((category, temp), 1)
        return padded_cat
      
# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines_train[category])
    return category, line
  
  
def randomTestPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines_test[category])
    return category, line
  
# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)
  
# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = Variable(categoryTensor(category))
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))
    return category_tensor, input_line_tensor, target_line_tensor

  
def randomTestExample():
    category, line = randomTestPair()
    category_tensor = Variable(categoryTensor(category))
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))
    return category_tensor, input_line_tensor, target_line_tensor
  
  
  
criterion = nn.NLLLoss()

learning_rate = 0.0005

#Case 1 - Original code
def train(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0] / input_line_tensor.size()[0]

  
#Case 2: hidden unit + previous character
def train2(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn2.initHidden(category_tensor)

    rnn2.zero_grad()

    loss2 = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn2(input_line_tensor[i], hidden)
        loss2 += criterion(output, target_line_tensor[i])

    loss2.backward()

    for p in rnn2.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss2.data[0] / input_line_tensor.size()[0]  

  
#Case 3 - category + hidden unit
def train3(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn3.initHidden()

    rnn3.zero_grad()

    loss3 = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn3(category_tensor, hidden)
        loss3 += criterion(output, target_line_tensor[i])

    loss3.backward()

    for p in rnn3.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss3.data[0] / input_line_tensor.size()[0]

  
#Case 4: hidden unit
def train4(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn4.initHidden(category_tensor)

    rnn4.zero_grad()

    loss4 = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn4(hidden)
        loss4 += criterion(output, target_line_tensor[i])

    loss4.backward()

    for p in rnn4.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss4.data[0] / input_line_tensor.size()[0]
  
  
#Testing functions
def test(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()
    testloss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        testloss += criterion(output, target_line_tensor[i])
    
    return output, testloss.data[0] / input_line_tensor.size()[0]  

  
def test2(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn2.initHidden(category_tensor)
    testloss2 = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn2(input_line_tensor[i], hidden)
        testloss2 += criterion(output, target_line_tensor[i])
    
    return output, testloss2.data[0] / input_line_tensor.size()[0]
  
  
def test3(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn3.initHidden()
    testloss3 = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn3(category_tensor, hidden)
        testloss3 += criterion(output, target_line_tensor[i])
    
    return output, testloss3.data[0] / input_line_tensor.size()[0]
  
  
def test4(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn4.initHidden(category_tensor)
    testloss4 = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn4(hidden)
        testloss4 += criterion(output, target_line_tensor[i])
    
    return output, testloss4.data[0] / input_line_tensor.size()[0]
  
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
  
  
rnn = RNN(n_letters, 128, n_letters)
rnn2 = RNN2(n_letters, 128, n_letters)
rnn3 = RNN3(n_letters, 128, n_letters)
rnn4 = RNN4(n_letters, 128, n_letters)

n_iters = 80000
n_iters1 = 20000
#print_every = 5000
plot_every = 5000
all_losses = []
total_loss = 0 # Reset every plot_every iters
all_losses_test1 = []
total_loss_test1 = 0
all_losses_test2 = []
total_loss_test2 = 0
all_losses_test3 = []
total_loss_test3 = 0
all_losses_test4 = []
total_loss_test4 = 0


start = time.time()

for iter in range(1, n_iters + 1):
    category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
    output1, loss1 = train(category_tensor, input_line_tensor, target_line_tensor)
    output2, loss2 = train2(category_tensor, input_line_tensor, target_line_tensor)
    output3, loss3 = train3(category_tensor, input_line_tensor, target_line_tensor)
    output4, loss4 = train4(category_tensor, input_line_tensor, target_line_tensor)


    if iter % plot_every == 0:
        for iter in range(1, n_iters1 + 1):
            category_tensort, input_line_tensort, target_line_tensort = randomTestExample()
            output_t1, testloss1 = test(category_tensort, input_line_tensort, target_line_tensort)
            total_loss_test1 += testloss1
            output_t2, testloss2 = test2(category_tensort, input_line_tensort, target_line_tensort)
            total_loss_test2 += testloss2
            output_t3, testloss3 = test3(category_tensort, input_line_tensort, target_line_tensort)
            total_loss_test3 += testloss3
            outputt4, testloss4 = test4(category_tensort, input_line_tensort, target_line_tensort)
            total_loss_test4 += testloss4
          
        all_losses_test1.append(total_loss_test1 / n_iters1)
        total_loss_test1 = 0
        all_losses_test2.append(total_loss_test2 / n_iters1)
        total_loss_test2 = 0
        all_losses_test3.append(total_loss_test3 / n_iters1)
        total_loss_test3 = 0
        all_losses_test4.append(total_loss_test4 / n_iters1)
        total_loss_test4 = 0

        

plt.figure()
plt.plot(all_losses_test1, 'm')
plt.plot(all_losses_test2, 'g')
plt.plot(all_losses_test3, 'b')
plt.plot(all_losses_test4, 'r')
plt.xlabel("Num of Iterations")
plt.ylabel("Loss")
plt.title("Test (Validation) Loss")
plt.legend(['Case 1', 'Case 2', 'Case 3', 'Case 4'], loc='upper right')

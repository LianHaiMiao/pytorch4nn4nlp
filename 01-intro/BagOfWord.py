from collections import defaultdict
import time
import random
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


# this is a class to contorl all the hyper parameters
class Config(object):
    def __init__(self):
        self.lr = 1e-3
        self.epoch_num = 10
        self.train_path = "../data/classes/train.txt"
        self.test_path = "../data/classes/test.txt"
        
config = Config()


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i)) # we can add the index automatically in this way
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      tag, words = line.lower().strip().split(" ||| ")
      yield ([w2i[x] for x in words.split(" ")], t2i[tag])
# the dataset just like ([id1, id2, id3, ...], tag1)
# [id1, id2, id3, ...] present the sentence, tag1 present the semantic



# Read in the data
train = list(read_dataset(config.train_path))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset(config.test_path))
nwords = len(w2i)
ntags = len(t2i)




# build the model
class BOW(nn.Module):
    def __init__(self, vocab_size, tag_size):
        super(BOW, self).__init__()
        self.embed = nn.Embedding(vocab_size, tag_size)
        self.bow_bias = Parameter(torch.Tensor(tag_size))

    def forward(self, x):
        embeds = self.embed(x) # b*w*t   in this case, b = batch = 1
        word_score = torch.sum(embeds, 1) # b*w*t -> bxt
        scores = word_score.add_(self.bow_bias)
        out = F.log_softmax(scores)
        return out

# create model
bow = BOW(nwords, ntags)

# optim and loss
optimizer = optim.Adam(bow.parameters(), lr=config.lr)
loss_fn = torch.nn.NLLLoss() # loss(bow(input), target) and last layer of model is LogSoftmax


# 开始训练
for epoch in range(config.epoch_num):
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for words, tag in train:
        words = Variable(torch.LongTensor(words)).view(1, -1)
        tag = Variable(torch.LongTensor([tag]))
        bow.zero_grad()
        my_loss = loss_fn(bow(words), tag)
        train_loss += my_loss.data[0]
        my_loss.backward()
        optimizer.step()
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (epoch, train_loss/len(train), time.time()-start))
    
    # Perform testing
    test_correct = 0.0
    for words, tag in dev:
        words = Variable(torch.LongTensor(words)).view(1, -1)
        scores = bow(words).data.numpy()
        predict = np.argmax(scores)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (epoch, test_correct/len(dev)))
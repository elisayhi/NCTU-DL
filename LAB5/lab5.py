# -*- coding: utf-8 -*-
#from __future__ import unicode_literals, print_function, division
import os
import re
import string
import random
import torch
import pickle
import unicodedata
import torch.nn as nn

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from io import open
from argparse import ArgumentParser
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


MAX_LENGTH = 20
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        #self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            #self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        #else:
            #self.word2count[word] += 1

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/train.txt', encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split(' ')] for l in lines]

    # Reverse pairs, make Lang instances
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, condition):
    input_tensor = tensorFromSentence(input_lang, pair[condition[0]])
    target_tensor = tensorFromSentence(output_lang, pair[condition[1]])
    return (input_tensor, target_tensor)

######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        if input_lang.n_words == 28 and output_lang.n_words == 28:
            break
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=32):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        mean = self.mean(hidden)
        logvar = self.logvar(hidden)
        return output, hidden, mean, logvar

    def initHidden(self, cond):
        #return torch.zeros(1, 1, self.hidden_size, device=device)  # VAE
        # CVAE
        zero = torch.zeros(1, 1, self.hidden_size-8, device=device)
        condition = torch.tensor([[ onehot(cond[0])+onehot(cond[1]) ]], device=device).float()
        return torch.cat((zero, condition), 2)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, latent_size=40, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.latent2decoder = nn.Linear(latent_size, hidden_size)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor, hidden, encoder_outputs, cond, is_head):
        if is_head:
            cond = onehot(cond[0]) + onehot(cond[1])
            hidden = torch.cat((hidden, torch.tensor([[cond]], device=device).float()), 2)
            hidden = self.latent2decoder(hidden)
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, cond=None):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def loss_func(pred, true, criterion, mean, logv):
    loss = criterion(pred, true)

    # KL dirvergeance
    #print(logv, mean.pow(2), logv.exp())
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    return loss, KL_loss

def reparameter(en_mean, en_logv):
    std = torch.exp(0.5 * en_logv)
    z = torch.autograd.Variable(torch.randn_like(std), volatile=False)
    z = z * std + en_mean
    z = torch.tensor(z).to(device).float()
    return z


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, condition, KL_weight=0, max_length=MAX_LENGTH):
    """
    condition: list, tense of input and output word
    """
    encoder_hidden = encoder.initHidden(condition)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # encoder
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    KL_losses = []
    for ei in range(input_length):
        encoder_output, encoder_hidden, en_mean, en_logvar = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # decoder
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # add condition
    decoder_hidden = reparameter(en_mean, en_logvar)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #print(f'di: {di}')
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, condition, di==0)
            N_loss, KL_loss = loss_func(decoder_output, target_tensor[di], criterion, en_mean, en_logvar)
            loss += (N_loss + KL_weight * KL_loss)
            KL_losses.append(KL_loss)
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, condition,  di==0)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            N_loss, KL_loss = loss_func(decoder_output, target_tensor[di], criterion, en_mean, en_logvar)
            loss += (N_loss + KL_weight * KL_loss)
            KL_losses.append(KL_loss)
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    #plot_losses = []
    print_loss_total = 0  # Reset every print_every
    #plot_loss_total = 0  # Reset every plot_every
    kl_losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    conditions = [[random.randint(0, 3) for _ in range(2)] for __ in range(n_iters)]
    training_pairs = [tensorsFromPair(random.choice(pairs), conditions[i]) for i in range(n_iters)]
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    for it in tqdm(range(1, n_iters + 1)):
    #for it in range(1, n_iters + 1):
        training_pair = training_pairs[it - 1]
        #print(training_pair)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        kl_weight = 1 / (5000-(it%5000)) if it>20000 else 0
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, conditions[it-1], KL_weight=kl_weight)#0.001*(it//3000))
        print_loss_total += loss
        #plot_loss_total += loss

        kl_losses.append(evaluateAll(encoder, decoder, it))
        if it % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, it / n_iters),
                                         it, it / n_iters * 100, print_loss_avg))
        if it % 1000:
            with open('result/kl_loss.pk', 'wb') as f:
                pickle.dump(kl_losses, f)

        if it == n_iters:
            with open('result/last_model.pk', 'wb') as f:
                pickle.dump({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, f)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, sentence, cond, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(cond)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden, en_mean, en_logvar = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = reparameter(en_mean, en_logvar)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, cond, di==0)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def onehot(x):
    """
    transfer a int (0-3) to 4-bit one hot
    """
    ret = [int(x==i) for i in range(4)]
    return ret

######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluateAll(encoder, decoder, it):#, n=10):
    ts_pairs, ts_conditions = [], []
    with open('data/test.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        lines = np.array([i.split(' ') for i in lines])
    ts_pairs = lines[:, (0,1)]
    ts_conditions = [i.split(',') for i in lines[:, 2]]
    ts_conditions = np.array(ts_conditions, dtype=int)
    correct = 0
    bleu_score = 0
    for pair, cond in zip(ts_pairs, ts_conditions):
        #print('input:', pair[0])
        #print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], cond)
        output_sentence = ''.join(output_words[:-1])
        #print('output', output_sentence)
        bleu_score += calc_bleu(output_sentence, pair[1])
        correct += int(output_sentence==pair[1])
    acc = correct/len(ts_pairs)
    if acc >= 0.7:
        with open(f'result/encoder_{acc}.pk', 'wb') as f:
            pickle.dump(encoder.state_dict(), f)
        with open(f'result/decoder_{acc}.pk', 'wb') as f:
            pickle.dump(decoder.state_dict(), f)
    return bleu_score / len(ts_pairs)


def calc_bleu(output, ref):
    cc = SmoothingFunction()
    return sentence_bleu([ref], output, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method1)

######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

hidden_size = 256
input_lang, output_lang, pairs = prepareData('train', 'test', True)
print(random.choice(pairs))

def main(n_iters):
    with open('result/lang.pk', 'wb') as f:
        pickle.dump(input_lang, f)

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    #print(input_lang.n_words, output_lang.n_words)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # reload model
    with open('result/model3/encoder_0.9.pk', 'rb') as f:
        encoder_weight = pickle.load(f)
    with open('result/model3/decoder_0.9.pk', 'rb') as f:
        decoder_weight = pickle.load(f)
    encoder1.load_state_dict(encoder_weight)
    attn_decoder1.load_state_dict(decoder_weight)


    trainIters(encoder1, attn_decoder1, n_iters, print_every=5000)
    print(f'bleu score: {evaluateAll(encoder1, attn_decoder1)}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', dest='n_iters', default=20000, type=int)
    args = parser.parse_args()
    main(args.n_iters)

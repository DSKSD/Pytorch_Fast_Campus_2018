import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from copy import deepcopy
import random
flatten = lambda l: [item for sublist in l for item in sublist]


def bAbI_data_load(path,vocab=None):
    print("Start to data loading...")
    try:
        data = open(path).readlines()
    except:
        print("Such a file does not exist at %s".format(path))
        return None
    
    data = [d[:-1] for d in data]
    data_p = []
    fact = []
    qa = []
    try:
        for d in data:
            index = d.split(' ')[0]
            if index == '1':
                fact = []
                qa = []
            if '?' in d:
                temp = d.split('\t')
                q = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']
                a = temp[1].split() + ['</s>']
                stemp = deepcopy(fact)
                data_p.append([stemp, q, a])
            else:
                tokens = d.replace('.', '').split(' ')[1:] + ['</s>']
                fact.append(tokens)
    except:
        print("Please check the data is right")
        return None
    
    if vocab:
        data, vocab = bAbI_build_vocab(data_p,vocab)
    else:
        data, vocab = bAbI_build_vocab(data_p)
    
    return data, vocab

def bAbI_build_vocab(data,vocab=None):
    if vocab==None:
        fact,q,a = list(zip(*data))
        vocab = list(set(flatten(flatten(fact)) + flatten(q) + flatten(a)))
        word2index={'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        for vo in vocab:
            if word2index.get(vo) is None:
                word2index[vo] = len(word2index)
        index2word = {v:k for k, v in word2index.items()}
    else:
        word2index = vocab
        
    for t in data:
        for i,fact in enumerate(t[0]):
            t[0][i] = prepare_sequence(fact, word2index).view(1, -1)

        t[1] = prepare_sequence(t[1], word2index).view(1, -1)
        t[2] = prepare_sequence(t[2], word2index).view(1, -1)
    
    return data, word2index
    
def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(torch.LongTensor(idxs))

def data_loader(train_data,batch_size,shuffle=False):
    if shuffle: random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def pad_to_batch(batch, w_to_ix): # for bAbI dataset
    fact,q,a = list(zip(*batch))
    max_fact = max([len(f) for f in fact])
    max_len = max([f.size(1) for f in flatten(fact)])
    max_q = max([qq.size(1) for qq in q])
    max_a = max([aa.size(1) for aa in a])
    
    facts, fact_masks, q_p, a_p = [], [], [], []
    for i in range(len(batch)):
        fact_p_t = []
        for j in range(len(fact[i])):
            if fact[i][j].size(1) < max_len:
                fact_p_t.append(torch.cat([fact[i][j], Variable(torch.LongTensor([w_to_ix['<pad>']] * (max_len - fact[i][j].size(1)))).view(1, -1)], 1))
            else:
                fact_p_t.append(fact[i][j])

        while len(fact_p_t) < max_fact:
            fact_p_t.append(Variable(torch.LongTensor([w_to_ix['<pad>']] * max_len)).view(1, -1))

        fact_p_t = torch.cat(fact_p_t)
        facts.append(fact_p_t)
        fact_masks.append(torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data))), volatile=False) for t in fact_p_t]).view(fact_p_t.size(0), -1))

        if q[i].size(1) < max_q:
            q_p.append(torch.cat([q[i], Variable(torch.LongTensor([w_to_ix['<pad>']] * (max_q - q[i].size(1)))).view(1, -1)], 1))
        else:
            q_p.append(q[i])

        if a[i].size(1) < max_a:
            a_p.append(torch.cat([a[i], Variable(torch.LongTensor([w_to_ix['<pad>']] * (max_a - a[i].size(1)))).view(1, -1)], 1))
        else:
            a_p.append(a[i])

    questions = torch.cat(q_p)
    answers = torch.cat(a_p)
    question_masks = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data))), volatile=False) for t in questions]).view(questions.size(0), -1)
    
    return facts, fact_masks, questions, question_masks, answers

def pad_to_fact(fact, x_to_ix): # this is for inference
    
    max_x = max([s.size(1) for s in fact])
    x_p = []
    for i in range(len(fact)):
        if fact[i].size(1) < max_x:
            x_p.append(torch.cat([fact[i], Variable(torch.LongTensor([x_to_ix['<pad>']] * (max_x - fact[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(fact[i])
        
    fact = torch.cat(x_p)
    fact_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))) for t in fact]).view(fact.size(0), -1)
    return fact, fact_mask
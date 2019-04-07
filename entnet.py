"""
Recurrent Entity Network implementation
(cf. https://arxiv.org/abs/1612.03969)
"""

import torch
from torch import nn


def init_params(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.001)


class InputEncoder(nn.Module):
    def __init__(self, embedding, embedding_dim, max_len, init_params=None):
        super(InputEncoder, self).__init__()
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.f = nn.Parameter(torch.ones(max_len, embedding_dim))

        if init_params:
            self.apply(init_params)

    def forward(self, x):
        return torch.sum(self.embedding(x) * self.f, dim=1)


class OutputModule(nn.Module):
    def __init__(self, embedding_dim, vocab_dim, activation=nn.PReLU(), init_params=None):
        super(OutputModule, self).__init__()
        self.block_dim = embedding_dim
        self.H = nn.Linear(embedding_dim, embedding_dim)
        self.R = nn.Linear(embedding_dim, vocab_dim)
        self.activation = activation
        self.softmax = nn.LogSoftmax(dim=0)

        if init_params:
            self.apply(init_params)

    def forward(self, q, h):
        y = []
        h = torch.split(h, self.block_dim)

        for qi in q:
            u = torch.zeros(h[0].size())
            for hj in h:
                u += self.softmax(qi * hj) * hj

            yi = self.R(self.activation(qi + self.H(u)))

            y.append(yi)

        return torch.stack(y)


class DynamicMemoryCell(nn.Module):
    def __init__(self, n_blocks, block_dim, keys, activation=nn.PReLU(), init_params=None):
        super(DynamicMemoryCell, self).__init__()
        self.n_blocks = n_blocks
        self.block_dim = block_dim
        self.keys = keys
        self.activation = activation

        self.U = nn.Parameter(torch.ones((block_dim, block_dim)))
        self.V = nn.Parameter(torch.ones((block_dim, block_dim)))
        self.W = nn.Parameter(torch.ones((block_dim, block_dim)))
        self.bias = nn.Parameter(torch.ones(block_dim))
        self.g_activation = nn.Sigmoid()

        if init_params:
            self.apply(init_params)

    def forward(self, s, h):
        h_new = []

        h = torch.split(h, self.block_dim)

        for j, hj in enumerate(h):
            w = self.keys[j]

            # gating function
            g = self.g_activation((s * hj).sum() + (s * w).sum())

            # candidate for memory update
            h_squiggle = self.activation(torch.matmul(self.U, hj)
                                         + torch.matmul(self.V, w)
                                         + torch.matmul(self.W, s)
                                         + self.bias)

            # update hidden state j
            hj_new = hj + g * h_squiggle

            # normalization (forgetting mechanism)
            hj_new = hj_new / torch.norm(hj_new)

            h_new.append(hj_new)

        return torch.cat(h_new)


class EntNet(nn.Module):
    def __init__(self,
                 embedding_dim,
                 max_story_len,
                 max_query_len,
                 vocab,
                 n_blocks,
                 init_params=None,
                 load_embeddings=None):

        super(EntNet, self).__init__()

        # include keys into the vocabulary
        self.vocab = vocab
        self.vocab_dim = self.vocab.num_words + n_blocks
        self.n_blocks = n_blocks
        self.block_dim = embedding_dim

        if load_embeddings:
            self.embedding, embedding_dim = load_embeddings()
        else:
            self.embedding = nn.Embedding(self.vocab_dim, embedding_dim)

        keys = self.embedding(torch.LongTensor([key for key in range(self.vocab_dim - n_blocks, self.vocab_dim)]))
        self.input_encoder = InputEncoder(self.embedding, embedding_dim, max_story_len, init_params=init_params)
        self.query_encoder = InputEncoder(self.embedding, embedding_dim, max_query_len, init_params=init_params)
        self.memory = DynamicMemoryCell(n_blocks, embedding_dim, keys, init_params=init_params)
        self.output_module = OutputModule(embedding_dim, self.vocab_dim, init_params=init_params)

        if init_params:
            self.apply(init_params)

    def forward(self, inputs, query, h0=None):
        x = self.input_encoder(inputs)
        q = self.query_encoder(query)

        h = None
        if h0:
            h = h0
        else:
            h = nn.Parameter(torch.rand(self.n_blocks * self.block_dim))

        for s in x:
            h = self.memory(s, h)

        y = self.output_module(q, h)

        return y, h


def loss_function(outputs, labels):
    loss = torch.nn.NLLLoss()

    return loss(outputs, labels)


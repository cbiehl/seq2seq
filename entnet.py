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
        dim = 0
        if len(x.size()) > 1:
            dim = 1

        return torch.sum(self.embedding(x) * self.f, dim=dim)


class OutputModule(nn.Module):
    def __init__(self, block_dim, n_blocks, vocab_dim, activation=nn.PReLU(), init_params=None):
        super(OutputModule, self).__init__()
        self.block_dim = block_dim
        self.n_blocks = n_blocks
        self.vocab_dim = vocab_dim
        self.H = nn.Linear(block_dim, block_dim)
        self.R = nn.Linear(block_dim, vocab_dim)
        self.activation = activation
        self.softmax = nn.LogSoftmax(dim=0)

        if init_params:
            self.apply(init_params)

    def forward(self, q, h):
        u = torch.zeros(self.block_dim)
        for j in range(self.n_blocks):
            low = j * self.block_dim
            high = j * self.block_dim + self.block_dim

            u += self.softmax(q * h[low:high]) * h[low:high]

        return self.R(self.activation(q + self.H(u)))


class DynamicMemoryCell(nn.Module):
    def __init__(self, n_blocks, block_dim, activation=nn.PReLU(), init_params=None):
        super(DynamicMemoryCell, self).__init__()
        self.n_blocks = n_blocks
        self.block_dim = block_dim
        self.activation = activation

        self.U = nn.Parameter(torch.randn((block_dim, block_dim)))
        self.V = nn.Parameter(torch.randn((block_dim, block_dim)))
        self.W = nn.Parameter(torch.randn((block_dim, block_dim)))
        self.bias = nn.Parameter(torch.ones(block_dim)) * 0.01
        self.g_activation = nn.Sigmoid()

        if init_params:
            self.apply(init_params)

    def forward(self, s, h, keys):
        h_new = []
        for j in range(self.n_blocks):
            low = j * self.block_dim
            high = j * self.block_dim + self.block_dim
            w = keys[j]

            # gating function
            g = self.g_activation(torch.dot(s, h[low:high]) + torch.dot(s, w))

            # candidate for memory update
            h_squiggle = self.activation(torch.matmul(self.U, h[low:high])
                                         + torch.matmul(self.V, w)
                                         + torch.matmul(self.W, s)
                                         + self.bias)

            # update hidden state j
            h_ = h[low:high] + g * h_squiggle

            # normalization (forgetting mechanism)
            h_ = h_ / torch.norm(h_)

            h_new.append(h_)

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

        self.key_idx = torch.LongTensor([key for key in range(self.vocab_dim - n_blocks, self.vocab_dim)])
        self.input_encoder = InputEncoder(self.embedding, embedding_dim, max_story_len, init_params=init_params)
        self.query_encoder = InputEncoder(self.embedding, embedding_dim, max_query_len, init_params=init_params)
        self.memory = DynamicMemoryCell(n_blocks, embedding_dim, init_params=init_params)
        self.output_module = OutputModule(embedding_dim, n_blocks, self.vocab.num_words, init_params=init_params)

        if init_params:
            self.apply(init_params)

    def forward(self, inputs, query):
        y = torch.zeros((len(inputs), self.vocab.num_words))
        keys = self.embedding(self.key_idx)
        for i, story in enumerate(inputs):
            h = torch.flatten(keys)  # torch.zeros(self.n_blocks * self.block_dim)
            q = self.query_encoder(query[i])
            for sent in story:
                s = self.input_encoder(sent)
                h = self.memory(s, h, keys)

            y[i] = self.output_module(q, h)

        return y

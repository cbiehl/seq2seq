#Preprocessing based on memory network implementation by fchollet & keras developers
#(cf. https://github.com/keras-team/keras/blob/master/examples/babi_memnn.py)

import re
import nltk
import numpy as np
from functools import reduce
from tensorflow.python.keras.utils.data_utils import get_file
from graphviz import Digraph
import torch


class Voc(object):
    """
    Taken from pytorch tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#create-formatted-data-file
    """
    def __init__(self, name, language='english'):
        self.PAD_token_idx = 0
        self.SOS_token_idx = 1
        self.EOS_token_idx = 2
        self.num_util_tokens = 3
        self.name = name
        self.language = language
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token_idx: "PAD", self.SOS_token_idx: "SOS", self.EOS_token_idx: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in nltk.word_tokenize(sentence, language=self.language):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_tokens(self, tokens):
        for word in tokens:
            self.add_word(word)

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token_idx: "PAD", self.SOS_token_idx: "SOS", self.EOS_token_idx: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.add_word(word)


def flatten(seqofseq):
    return reduce(lambda x, y: x + y, seqofseq)

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def read_vocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)

    return voc, pairs


def download_babi_dataset(outputpath, datasetpath):
    """
    Download bAbi dataset

    :param outputpath:
    :param datasetpath:
    :return:
    """
    try:
        path = get_file(outputpath, origin=datasetpath)
    except ConnectionResetError:
        print("Error downloading dataset: Connection reset.")

    except:
        print("Error loading dataset.")

    return path


def tokenize(sent, language='english'):
    """
    Tokenize a given sentence (i.e. string) using nltk,
    returns list of tokens

    :param language:
    :param sent:
    :return:
    """
    return nltk.word_tokenize(sent, language)


def parse_stories(lines, only_supporting=False):
    """
    Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences
    that support the answer are kept.

    :param lines:
    :param only_supporting:
    :return:
    """
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)

    return data


def get_stories(f, only_supporting=False, max_length=None, flatten_stories=False):
    """
    Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.

    :param f:
    :param only_supporting:
    :param max_length:
    :return:
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    if flatten_stories:
        data = [(flatten(story), q, answer) for story, q, answer in data
                if not max_length or len(flatten(story)) < max_length]
    else:
        data = [(story, q, answer) for story, q, answer in data
                if not max_length or len(flatten(story)) < max_length]

    return data


def zero_pad(seq, voc, maxlen=None, flat=True):
    # get the length of each sentence
    lengths = None
    if flat:
        lengths = [len(s) for s in seq]
    else:
        lengths = [len(s) for story in seq for s in story]

    pad_token = voc.PAD_token_idx

    if maxlen is None:
        maxlen = max(lengths)

    # create an empty matrix with padding tokens
    padded_x = None
    if flat:
        batch_size = len(seq)
        padded_x = np.ones((batch_size, maxlen)) * pad_token

        # copy over the actual sequences
        for i, x_len in enumerate(lengths):
            sequence = seq[i]
            padded_x[i, 0:x_len] = sequence[:x_len]

    else:
        padded_x = []

        k = 0
        for i, s in enumerate(seq):
            padded_sent = np.ones((len(seq[i]), maxlen)) * pad_token
            for j, sent in enumerate(s):
                sent_len = lengths[k]
                padded_sent[j, 0:sent_len] = seq[i][j][:sent_len]
                k += 1

            padded_x.append(padded_sent)

    return padded_x


def vectorize_stories(data, voc, story_maxlen, query_maxlen, stories_are_flat=True):
    inputs, queries, answers = [], [], []

    for story, query, answer in data:
        inputs.append([[voc.word2index[w] for w in sent] for sent in story])
        queries.append([voc.word2index[w] for w in query])
        answers.append(voc.word2index[answer])

    padded_inputs = zero_pad(inputs, voc, maxlen=story_maxlen, flat=stories_are_flat)
    padded_queries = zero_pad(queries, voc, maxlen=query_maxlen, flat=True)

    return (padded_inputs,
            padded_queries,
            np.array(answers))


"""
    PyTorch Graph Visualization
    Author: Ludovic Trottier
    Date created: November 8, 2017.
    Credits: moskomule (https://discuss.pytorch.org/t/print-autograd-graph/692/15)
"""
def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.

    Args:
        var: output Variable
        params: list of (name, Parameters)
    """

    param_map = {id(v): k for k, v in params}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(
        filename='network',
        format='pdf',
        node_attr=node_attr,
        graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:

            node_id = str(id(var))

            if torch.is_tensor(var):
                node_label = "saved tensor\n{}".format(tuple(var.size()))
                dot.node(node_id, node_label, fillcolor='orange')

            elif hasattr(var, 'variable'):
                variable_name = param_map.get(id(var.variable))
                variable_size = tuple(var.variable.size())
                node_name = "{}\n{}".format(variable_name, variable_size)
                dot.node(node_id, node_name, fillcolor='lightblue')

            else:
                node_label = type(var).__name__.replace('Backward', '')
                dot.node(node_id, node_label)

            seen.add(var)

            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])

            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)

    return dot

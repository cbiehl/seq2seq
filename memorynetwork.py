"""
End-to-End Memory Network implementation using keras & TensorFlow

Based on memory network implementation by fchollet & keras developers
(cf. https://github.com/keras-team/keras/blob/master/examples/babi_memnn.py)

References:
http://arxiv.org/abs/1502.05698
http://arxiv.org/abs/1503.08895
"""

import tensorflow as tf
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import Input, Activation, Dense, Permute, Dropout, Bidirectional
from tensorflow.python.keras.layers import add, dot, concatenate
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model
from functools import reduce
import time
import tarfile
import nltk
import numpy as np


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


def get_stories(f, only_supporting=False, max_length=None):
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
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]

    return data


def vectorize_stories(data):
    inputs, queries, answers = [], [], []

    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])

    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))


class MemoryNetwork(Model):
    def __init__(self, n_memories=1, n_memory_units=64, n_recurrent_units=32, dropout=0.3):
        super(MemoryNetwork, self).__init__(name='MemoryNetwork')

        self.n_recurrent_units = n_recurrent_units
        self.dropout = dropout

        # TODO: multiple memory units
        # self.memories = []

        # input layers
        self.input_sequence = Input((story_maxlen,))
        self.question = Input((query_maxlen,))

        # input encoders
        self.input_encoder_m = Sequential()
        self.input_encoder_m.add(Embedding(input_dim=vocab_size,
                                           output_dim=64))
        self.input_encoder_m.add(Dropout(0.3))

        self.input_encoder_c = Sequential()
        self.input_encoder_c.add(Embedding(input_dim=vocab_size,
                                           output_dim=query_maxlen))
        self.input_encoder_c.add(Dropout(0.3))

        # question encoder
        self.question_encoder = Sequential()
        self.question_encoder.add(Embedding(input_dim=vocab_size,
                                            output_dim=64,
                                            input_length=query_maxlen))
        self.question_encoder.add(Dropout(0.3))

        # encode input sequence and question
        input_encoded_m = self.input_encoder_m(self.input_sequence)
        input_encoded_c = self.input_encoder_c(self.input_sequence)
        question_encoded = self.question_encoder(self.question)

        # compute input memory representation
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match) # (samples, story_maxlen, query_maxlen)

        # compute output memory representation
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        answer = concatenate([response, question_encoded])
        answer = Bidirectional(GRU(32), merge_mode='sum')(answer)
        answer = Dropout(0.3)(answer)
        answer = Dense(vocab_size)(answer)  # (samples, vocab_size)

        # output: distribution over the vocabulary
        self.answer = Activation('softmax')(answer)

        self.model = Model([self.input_sequence, self.question], self.answer)

    def call(self, inputs, training=None, mask=None):
        """
        Predict single answer per input sequence in x

        :param x: input sequences
        :param q: questions
        :return:
        """
        return self.model.predict(inputs)

    def train(self, x_train, q_train, a_train, x_test, q_test, a_test, n_epochs=100, batch_size=32):
        """
        Train the model

        :param x_train: input sequences for training
        :param q_train: questions for training
        :param a_train: answers for training
        :param x_test: input sequences for validation
        :param q_test: questions for validation
        :param a_test: answers for validation
        :param n_epochs: number of training epochs (no early stopping)
        :param batch_size: batch size
        """

        optimizer = RMSprop()
        self.model.compile(optimizer, loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        self.model.fit([x_train, q_train], a_train,
                 batch_size=batch_size,
                 epochs=n_epochs,
                 validation_data=([x_test, q_test], a_test))

    def save(self, path):
        """
        Saves the model to the given HDF5 file with keras

        :param path:
        """
        self.model.save(path, overwrite=True, include_optimizer=True)

    @staticmethod
    def load(self, path):
        """
        Loads a saved model from HDF5 file with keras

        :param path:
        """
        self.model = load_model(path)


# test-run
outputpath = 'babi-tasks-v1-2.tar.gz'
path = download_babi_dataset(outputpath, 'https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    # ...
    'three_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_{}.txt'
}

challenge_type = 'three_supporting_facts_10k' #'two_supporting_facts_10k' #'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
try:
    with tarfile.open(path) as tar:
        train_stories = get_stories(tar.extractfile(challenge.format('train')))
        test_stories = get_stories(tar.extractfile(challenge.format('test')))

except FileNotFoundError:
    print("Error: Dataset file not found.")

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])

vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
x_train, q_train, a_train = vectorize_stories(train_stories)
x_test, q_test, a_test = vectorize_stories(test_stories)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', x_train.shape)
print('inputs_test shape:', x_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', q_train.shape)
print('queries_test shape:', q_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', a_train.shape)
print('answers_test shape:', a_test.shape)
print('-')
print('Compiling...')
memnet = MemoryNetwork()
print('Training...')
memnet.train(x_train, q_train, a_train, x_test, q_test, a_test, n_epochs=100)

memnet.save('dnm_model_' + str(time.time()))

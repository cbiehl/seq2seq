import torch
import numpy as np
import tarfile
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score

from entnet import EntNet, init_params

from data_utils import Voc, download_babi_dataset, get_stories, vectorize_stories, flatten, make_dot

# constants
MODE = "training"  # "training", "inference" or "continue"
FILE_MODEL = "data/tmp/entnet"
FILE_SAVED_VOCAB = "data/tmp/vocab.pickle"
FILE_BUFFER_BATH = "data/tmp/babistories.npz"
N_EPOCHS = 100
BATCH_SIZE = 200
LEARNING_RATE = 0.01
N_MEMORY_BLOCKS = 10
EMBEDDING_DIM = 50


def mask_nll_loss(logits, targets, masks):
    total = masks.sum()
    cross_entropy = -torch.log(torch.gather(logits, 1, targets.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(masks).mean()

    return loss, total.item()


def loss_function(logits, targets, masks=None):
    if masks is None:
        loss_fn = torch.nn.NLLLoss()
        loss = loss_fn(logits, targets)
    else:
        loss = mask_nll_loss(logits, targets, masks)

    return loss


def to_long_tensor(array, require_grad=False):
    return torch.tensor(array, dtype=torch.long, requires_grad=require_grad)


def to_long_tensors(*arraylikes, require_grad=False):
    tensors = []
    for a in arraylikes:
        tensors.append(torch.tensor(a, dtype=torch.long, requires_grad=require_grad))

    return tensors


def train(optimizer, model, inputs, queries, answers, batch_size):
    avg_loss = 0

    n = 0
    if type(inputs) in [list, np.ndarray]:
        n = len(inputs)
    else:
        n = inputs.size()[0]

    permutation = torch.randperm(n).detach()

    for i in range(0, n, batch_size):
        # Zero gradients
        optimizer.zero_grad()

        # Extract batch
        indices = permutation[i:i + batch_size]
        b_inputs, b_queries, b_answers = inputs[indices], queries[indices], answers[indices]

        # Forward pass
        logits = model(b_inputs, b_queries)

        loss = loss_function(logits, b_answers)

        if torch.isnan(loss):
            print("NaNNaNNaNaNaNaN Batman!!!")

        # Perform backpropagation
        loss.backward()

        # Clip gradients in-place
        clip = 50.0
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update model weights
        optimizer.step()

        avg_loss = (avg_loss + loss) / batch_size

    return avg_loss


def test(model, vocab, inputs, queries, answers):
    print("Testing model...")
    logits = model(inputs, queries)
    amax = logits.argmax(dim=1).detach().numpy()
    labels = answers.detach().numpy()

    accuracy = accuracy_score(labels, amax)
    f1 = f1_score(labels, amax, average='micro')
    print("Accuracy: ", accuracy)
    print("F1 (micro avg): ", f1)
    print()

    questions = queries[0:10]
    contexts = inputs[0:10]

    for i, c in enumerate(contexts):
        print("Test example %i" % i)
        context = [vocab.index2word[idx.item()] for sent in c for idx in sent if idx.item() != vocab.PAD_token_idx]
        question = [vocab.index2word[idx.item()] for idx in questions[i] if idx.item() != vocab.PAD_token_idx]
        print("Context:")
        print(' '.join(context))
        print()
        print("Question:")
        print(' '.join(question))
        print()
        print("Predicted answer:")
        print(vocab.index2word[amax[i]])
        print()
        print("Actual answer:")
        print(vocab.index2word[labels[i]])
        print()
        print()


def save(model, optimizer, filepath=FILE_MODEL):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, filepath)
    print("Saved model to %s" % filepath)


def visualize(model, contexts, queries):
    y = model(contexts, queries)

    g = make_dot(y, model.named_parameters())

    g.view()


def main():
    if not os.path.isfile(FILE_BUFFER_BATH):
        # Preprocess data from scratch
        # Load bAbI dataset
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

        challenge_type = 'single_supporting_fact_10k'  # 'three_supporting_facts_10k' 'two_supporting_facts_10k'
        challenge = challenges[challenge_type]

        print('Extracting stories for the challenge:', challenge_type)
        try:
            with tarfile.open(path) as tar:
                train_stories = get_stories(tar.extractfile(challenge.format('train')))
                test_stories = get_stories(tar.extractfile(challenge.format('test')))

        except FileNotFoundError:
            print("Error: Dataset file not found.")

        vocab = Voc("bAbI_vocab")
        for story, q, answer in train_stories + test_stories:
            vocab.add_tokens(flatten(story))
            vocab.add_tokens(q)
            if type(answer) == str:
                vocab.add_word(answer)
            else:
                vocab.add_tokens(answer)

        story_maxlen = max(map(len, (sent for x, _, _ in train_stories + test_stories for sent in x)))
        query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

        print('-')
        print('Vocab size:', vocab.num_words, 'unique words')
        print('Story max length:', story_maxlen, 'words')
        print('Query max length:', query_maxlen, 'words')
        print('Number of training stories:', len(train_stories))
        print('Number of test stories:', len(test_stories))
        print('-')
        print('Here\'s what a "story" tuple looks like (input, query, answer):')
        print(train_stories[0])
        print('-')
        print('Vectorizing the word sequences...')

        x_train, q_train, a_train = vectorize_stories(train_stories, vocab, story_maxlen, query_maxlen, False)
        x_test, q_test, a_test = vectorize_stories(test_stories, vocab, story_maxlen, query_maxlen, False)

        np.savez(FILE_BUFFER_BATH,
                 x_train=x_train, q_train=q_train, a_train=a_train,
                 x_test=x_test, q_test=q_test, a_test=a_test,
                 story_maxlen=story_maxlen, query_maxlen=query_maxlen)

        with open(FILE_SAVED_VOCAB, 'wb') as f:
            pickle.dump(vocab, f)

    else:
        # Load preprocessed data
        data = np.load(FILE_BUFFER_BATH)
        x_train = data['x_train']
        x_test = data['x_test']
        q_train = data['q_train']
        q_test = data['q_test']
        a_train = data['a_train']
        a_test = data['a_test']
        story_maxlen = data['story_maxlen']
        query_maxlen = data['query_maxlen']

        with open(FILE_SAVED_VOCAB, 'rb') as f:
            vocab = pickle.load(f)

    # print('-')
    # print('inputs: integer tensor of shape (samples, max_length)')
    # print('inputs_train shape:', x_train.shape)
    # print('inputs_test shape:', x_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', q_train.shape)
    print('queries_test shape:', q_test.shape)
    print('-')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('answers_train shape:', a_train.shape)
    print('answers_test shape:', a_test.shape)
    print('-')
    print()

    for i in range(len(x_train)):
        x_train[i] = to_long_tensor(x_train[i])

    for i in range(len(x_test)):
        x_test[i] = to_long_tensor(x_test[i])

    q_train, a_train, q_test, a_test = to_long_tensors(q_train, a_train, q_test, a_test)

    model = EntNet(EMBEDDING_DIM, story_maxlen, query_maxlen, vocab, n_blocks=N_MEMORY_BLOCKS, init_params=init_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Model summary:")
    print(model)
    print()

    if MODE == "training":
        # train from scratch
        print("Training model...")
        for i in range(N_EPOCHS):
            loss = train(optimizer, model, x_train, q_train, a_train, batch_size=BATCH_SIZE)

            if i % 10 == 0:
                test(model, vocab, x_test, q_test, a_test)
                save(model, optimizer)

            # divide learning rate by half until epoch 200
            if 1 < i < 200 and i % 25 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2

            print("Episode %i\tLoss: %f" % (i, loss))

        save(model, optimizer)

    elif MODE == "continue":
        # Load model parameters and continue training
        state = torch.load(FILE_MODEL)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

        print("Training model...")
        for i in range(N_EPOCHS):
            loss = train(optimizer, model, x_train, q_train, a_train, batch_size=BATCH_SIZE)

            if i % 10 == 0:
                test(model, vocab, x_test, q_test, a_test)
                save(model, optimizer)

            # divide learning rate by half until epoch 200
            if 1 < i < 200 and i % 25 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2

            print("Episode %i\tLoss: %f" % (i, loss))

        save(model, optimizer)

    else:
        # Load model parameters
        state = torch.load(FILE_MODEL)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    test(model, vocab, x_test, q_test, a_test)


if __name__ == '__main__':
    main()

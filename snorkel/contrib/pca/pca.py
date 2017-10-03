import os
import numpy as np

import warnings

from snorkel.learning.disc_learning import TFNoiseAwareModel
from snorkel.models import Candidate

from utils import candidate_to_tokens, SymbolTable
from six.moves.cPickle import dump, load
from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils

from snorkel.learning.utils import reshape_marginals, LabelBalancer


class PCA(TFNoiseAwareModel):
    name = 'PCA'
    representation = True
    gpu = ['gpu', 'GPU']

    """SVD for relation extraction"""

    def gen_dist_exp(self, length, decay, st, ed):
        ret = []
        for i in range(length):
            if i < st:
                ret.append(decay ** (st - i))
            elif i >= ed:
                ret.append(decay ** (i - ed + 1))
            else:
                ret.append(1.)
        return np.array(ret)

    def gen_dist_linear(self, length, st, ed, window_size):
        ret = []
        for i in range(length):
            if i < st:
                ret.append(max(0., 1. * (window_size - st + i + 1) / window_size))
            elif i >= ed:
                ret.append(max(0., 1. * (window_size + ed - i) / window_size))
            else:
                ret.append(1.)
        return np.array(ret)

    def _preprocess_data_combination(self, candidates):
        """Convert candidate sentences to lookup sequences

        :param candidates: candidates to process
        """

        if len(candidates) > 0 and not isinstance(candidates[0], Candidate):
            return candidates

        data = []
        for candidate in candidates:
            words = candidate_to_tokens(candidate)

            # Word level embeddings
            sent = np.array(words)

            k = self.window_size

            m1_start = candidate[0].get_word_start()
            m1_end = candidate[0].get_word_end() + 1

            if self.kernel == 'exp':
                dist_m1 = self.gen_dist_exp(len(words), self.decay, m1_start, m1_end)
            elif self.kernel == 'linear':
                dist_m1 = self.gen_dist_linear(len(words), m1_start, m1_end, self.window_size)
                dist_m1_sent = self.gen_dist_linear(len(words), m1_start, m1_end, max(m1_start, len(words) - m1_end))

            m1_start = max(m1_start - k, 0)
            m1_end = min(m1_end + k, len(words))

            m2_start = candidate[1].get_word_start()
            m2_end = candidate[1].get_word_end() + 1

            if self.kernel == 'exp':
                dist_m2 = self.gen_dist_exp(len(words), self.decay, m2_start, m2_end)
            elif self.kernel == 'linear':
                dist_m2 = self.gen_dist_linear(len(words), m2_start, m2_end, self.window_size)
                dist_m2_sent = self.gen_dist_linear(len(words), m2_start, m2_end, max(m2_start, len(words) - m2_end))

            m2_start = max(m2_start - k, 0)
            m2_end = min(m2_end + k, len(words))

            if self.kernel == 'exp':
                dist_sent = np.maximum(dist_m1, dist_m2)
            elif self.kernel == 'linear':
                dist_sent = np.maximum(dist_m1_sent, dist_m2_sent)

            m1 = np.array(words[m1_start: m1_end])
            m2 = np.array(words[m2_start: m2_end])
            st = min(candidate[0].get_word_end(), candidate[1].get_word_end())
            ed = max(candidate[0].get_word_start(), candidate[1].get_word_start())
            word_seq = np.array(words[st + 1: ed])

            order = 0 if m1_start < m2_start else 1
            if self.kernel is None:
                data.append((sent, m1, m2, word_seq, order))
            else:
                dm1 = dist_m1[m1_start: m1_end]
                dm2 = dist_m2[m2_start: m2_end]
                dword_seq = dist_sent[st + 1: ed]
                data.append((sent, m1, m2, word_seq, order, dist_sent, dm1, dm2, dword_seq))

        return data

    def _check_max_sentence_length(self, ends, max_len=None):
        """Check that extraction arguments are within @self.max_sentence_length"""
        mx = max_len or self.max_sentence_length
        for i, end in enumerate(ends):
            if end >= mx:
                w = "Candidate {0} has argument past max length for model:"
                info = "[arg ends at index {0}; max len {1}]".format(end, mx)
                warnings.warn('\t'.join([w.format(i), info]))

    def _build_model(self, **model_kwargs):
        pass

    def create_dict(self, splits, word=True, char=True):
        if word: self.word_dict = SymbolTable()
        if char: self.char_dict = SymbolTable()

        for candidates in splits['train']:
            for candidate in candidates:
                words = candidate_to_tokens(candidate)
                if word: map(self.word_dict.get, words)
                if char: map(self.char_dict.get, list(' '.join(words)))

        print "|Train Vocab|    = word:{}, char:{}".format(self.word_dict.s, self.char_dict.s)

        for candidates in splits['test']:
            for candidate in candidates:
                words = candidate_to_tokens(candidate)
                if word: map(self.word_dict.get, words)
                if char: map(self.char_dict.get, list(' '.join(words)))

        print "|Total Vocab|    = word:{}, char:{}".format(self.word_dict.s, self.char_dict.s)

    def load_dict(self):
        # load dict from glove
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
            extend_word = True
        else:
            extend_word = False

        if self.char and not hasattr(self, 'char_dict'):
            self.char_dict = SymbolTable()
            extend_char = True
        else:
            extend_char = False

        if extend_word:
            # Word embeddins
            f = open(self.word_emb_path, 'r')

            l = list()
            for _ in f:
                if len(_.strip().split(' ')) > self.word_emb_dim + 1:
                    l.append(' ')
                else:
                    word = _.strip().split(' ')[0]
                    # Replace placeholder to original word defined by user.
                    for key in self.replace.keys():
                        word = word.replace(key, self.replace[key])
                    l.append(word)
            map(self.word_dict.get, l)
            f.close()

        if self.char and extend_char:
            # Char embeddings
            f = open(self.char_emb_path, 'r')

            l = list()
            for _ in f:
                if len(_.strip().split(' ')) > self.char_emb_dim + 1:
                    l.append(' ')
                else:
                    word = _.strip().split(' ')[0]
                    # Replace placeholder to original word defined by user.
                    for key in self.replace.keys():
                        word = word.replace(key, self.replace[key])
                    l.append(word)
            map(self.char_dict.get, l)
            f.close()

    def load_embeddings(self):
        self.load_dict()
        # Random initial word embeddings
        self.word_emb = np.random.uniform(-0.01, 0.01, (self.word_dict.s, self.word_emb_dim)).astype(np.float)

        if self.char:
            # Random initial char embeddings
            self.char_emb = np.random.uniform(-0.01, 0.01, (self.char_dict.s, self.char_emb_dim)).astype(np.float)

        # Set unknown
        unknown_symbol = 1

        # Word embeddings
        f = open(self.word_emb_path, 'r')
        fmt = "fastText" if self.word_emb_path.split(".")[-1] == "vec" else "txt"

        for i,line in enumerate(f):

            if fmt == "fastText" and i == 0:
                continue

            line = line.strip().split(' ')
            if len(line) > self.word_emb_dim + 1:
                line[0] = ' '
            for key in self.replace.keys():
                line[0] = line[0].replace(key, self.replace[key])
            if self.word_dict.lookup(line[0]) != unknown_symbol:
                self.word_emb[self.word_dict.lookup_strict(line[0])] = np.asarray(
                    [float(_) for _ in line[-self.word_emb_dim:]])
        f.close()

        if self.char:
            # Char embeddings
            f = open(self.char_emb_path, 'r')

            for line in f:
                line = line.strip().split(' ')
                if len(line) > self.char_emb_dim + 1:
                    line[0] = ' '
                for key in self.replace.keys():
                    line[0] = line[0].replace(key, self.replace[key])
                if self.char_dict.lookup(line[0]) != unknown_symbol:
                    self.char_emb[self.char_dict.lookup_strict(line[0])] = np.asarray(
                        [float(_) for _ in line[-self.char_emb_dim:]])
            f.close()

    def gen_char_list(self, x, k):
        ret = []
        for w in x:
            l = ['<s>'] + list(w) + ['</s>']
            for i in range(len(l) - k + 1):
                ret.append(''.join(l[i:i + k]))
        ret = [_.replace('<s>', '') for _ in ret]
        ret = filter(lambda x: x != '', ret)
        ret = filter(lambda x: x != '</s>', ret)
        return ret

    def get_principal_components(self, x, y=None):
        # word level features
        ret1 = torch.zeros(self.r + 1, self.word_emb_dim).double()
        if len(''.join(x)) > 0 and len(x) > self.l:
            f = self.word_dict.lookup
            if y is None:
                x_ = torch.from_numpy(np.array([self.word_emb[_] for _ in map(f, x)]))
            else:
                x_ = torch.from_numpy(np.diag(y).dot(np.array([self.word_emb[_] for _ in map(f, x)])))
            mu = torch.mean(x_, 0, keepdim=True)
            ret1[0, ] = mu / torch.norm(mu)
            if self.r > 0:
                u, s, v = torch.svd(x_ - mu.repeat(x_.size(0), 1))
                k = self.r if v.size(1) > self.l + self.r else v.size(1) - self.l
                ret1[1:k + 1, ] = v.transpose(0, 1)[self.l: self.l + k, ]

        if self.bidirectional:
            # word level features
            x_b = list(reversed(x))
            ret1_b = torch.zeros(self.r + 1, self.word_emb_dim).double()
            if len(''.join(x_b)) > 0 and len(x_b) > self.l:
                f = self.word_dict.lookup
                if y is None:
                    x_ = torch.from_numpy(np.array([self.word_emb[_] for _ in map(f, x_b)]))
                else:
                    y_b = list(reversed(y))
                    x_ = torch.from_numpy(np.diag(y_b).dot(np.array([self.word_emb[_] for _ in map(f, x_b)])))
                mu = torch.mean(x_, 0, keepdim=True)
                ret1_b[0,] = mu / torch.norm(mu)
                if self.r > 0:
                    u, s, v = torch.svd(x_ - mu.repeat(x_.size(0), 1))
                    k = self.r if v.size(1) > self.l + self.r else v.size(1) - self.l
                    ret1_b[1:k + 1, ] = v.transpose(0, 1)[self.l: self.l + k, ]

        if self.char:
            # char level features
            ret2 = torch.zeros(self.r + 1, self.char_emb_dim).double()
            x_c = self.gen_char_list(x, self.char_gram)
            if len(x_c) > 0 and len(x_c) > self.l:
                f = self.char_dict.lookup
                x_ = torch.from_numpy(np.array([self.char_emb[_] for _ in map(f, list(x_c))]))
                mu = torch.mean(x_, 0, keepdim=True)
                ret2[0, ] = mu / torch.norm(mu)
                if self.r > 0:
                    u, s, v = torch.svd(x_ - mu.repeat(x_.size(0), 1))
                    k = self.r if v.size(1) > self.l + self.r else v.size(1) - self.l
                    # k = self.r if v.size(1) > self.r else v.size(1)
                    ret2[1:k + 1, ] = v.transpose(0, 1)[self.l: self.l + k, ]
            if self.bidirectional:
                # char level features
                ret2_b = torch.zeros(self.r + 1, self.char_emb_dim).double()
                x_c_b = x_c[::-1]
                if len(x_c_b) > 0 and len(x_c_b) > self.l:
                    f = self.char_dict.lookup
                    x_ = torch.from_numpy(np.array([self.char_emb[_] for _ in map(f, list(x_c_b))]))
                    mu = torch.mean(x_, 0, keepdim=True)
                    ret2_b[0, ] = mu / torch.norm(mu)
                    if self.r > 0:
                        u, s, v = torch.svd(x_ - mu.repeat(x_.size(0), 1))
                        k = self.r if v.size(1) > self.l + self.r else v.size(1) - self.l
                        # k = self.r if v.size(1) > self.r else v.size(1)
                        ret2_b[1:k + 1, ] = v.transpose(0, 1)[self.l: self.l + k, ]

        if self.char:
            if self.bidirectional:
                return torch.cat((ret1.view(1, -1), ret2.view(1, -1), ret1_b.view(1, -1), ret2_b.view(1, -1)), 1)
            else:
                return torch.cat((ret1.view(1, -1), ret2.view(1, -1)), 1)
        else:
            if self.bidirectional:
                return torch.cat((ret1.view(1, -1), ret1_b.view(1, -1)), 1)
            else:
                return ret1.view(1, -1)

    def gen_feature(self, X):
        if self.kernel is None:
            m1 = self.get_principal_components(X[1])
            m2 = self.get_principal_components(X[2])
            if self.sent_feat:
                sent = self.get_principal_components(X[0])
            if self.cont_feat:
                word_seq = self.get_principal_components(X[3])
        else:
            m1 = self.get_principal_components(X[1], X[6])
            m2 = self.get_principal_components(X[2], X[7])
            if self.sent_feat:
                sent = self.get_principal_components(X[0], X[5])
            if self.cont_feat:
                word_seq = self.get_principal_components(X[3], X[8])

        if self.sent_feat and self.cont_feat:
            feature = torch.cat((m1, m2, sent, word_seq))
        elif self.sent_feat:
            feature = torch.cat((m1, m2, sent))
        elif self.cont_feat:
            feature = torch.cat((m1, m2, word_seq))
        else:
            feature = torch.cat((m1, m2))

        feature = feature.view(1, -1)

        # add indicator feature for asymmetric relation
        if self.asymmetric:
            order = torch.zeros(1, 2).double()
            if X[4] == 0:
                order[0][0] = 1.0
            else:
                order[0][1] = 1.0
            feature = torch.cat((feature, order), 1).view(1, -1)

        return feature

    def build_model(self, input_dim, output_dim):
        # We don't need the softmax layer here since CrossEntropyLoss already
        # uses it internally.
        model = torch.nn.Sequential()
        model.add_module("linear",
                         torch.nn.Linear(input_dim, output_dim, bias=False))
        if self.host_device in self.gpu: model.cuda()
        return model

    def train_model(self, model, loss, optimizer, x_val, y_val):
        if self.host_device in self.gpu:
            x = Variable(x_val, requires_grad=False).cuda()
            y = Variable(y_val, requires_grad=False).cuda()
        else:
            x = Variable(x_val, requires_grad=False)
            y = Variable(y_val, requires_grad=False)

        # Reset gradient
        optimizer.zero_grad()

        # Forward
        fx = model.forward(x)

        if self.host_device in self.gpu:
            output = loss.forward(fx.cuda(), y)
        else:
            output = loss.forward(fx, y)

        # Backward
        output.backward()

        # Update parameters
        optimizer.step()

        return output.data[0]

    def predict(self, model, x_val):
        if self.host_device in self.gpu:
            x = Variable(x_val, requires_grad=False).cuda()
        else:
            x = Variable(x_val, requires_grad=False)
        output = model.forward(x)
        sigmoid = nn.Sigmoid()
        if self.host_device in self.gpu:
            pred = sigmoid(output).data.cpu().numpy()
        else:
            pred = sigmoid(output).data.numpy()
        return pred

    def _init_kwargs(self, **kwargs):

        self.model_kwargs = kwargs

        if kwargs.get('init_pretrained', False):
            self.create_dict(kwargs['init_pretrained'], word=True, char=True)

        # Set if use char embeddings
        self.char = kwargs.get('char', True)

        # Set if use whole sentence feature
        self.sent_feat = kwargs.get('sent_feat', True)

        # Set if use whole context feature
        self.cont_feat = kwargs.get('cont_feat', True)

        # Set bidirectional
        self.bidirectional = kwargs.get('bidirectional', False)

        # Set word embedding dimension
        self.word_emb_dim = kwargs.get('word_emb_dim', None)

        # Set word embedding path
        self.word_emb_path = kwargs.get('word_emb_path', None)

        # Set char gram k
        self.char_gram = kwargs.get('char_gram', 1)

        # Set char embedding dimension
        self.char_emb_dim = kwargs.get('char_emb_dim', None)

        # Set char embedding path
        self.char_emb_path = kwargs.get('char_emb_path', None)

        # Set learning rate
        self.lr = kwargs.get('lr', 1e-3)

        # Set learning epoch
        self.n_epochs = kwargs.get('n_epochs', 100)

        # Set ignore first k principal components
        self.l = kwargs.get('l', 0)

        # Set select top k principal components from the rest
        self.r = kwargs.get('r', 10)

        # Set learning batch size
        self.batch_size = kwargs.get('batch_size', 100)

        # Set rebalance setting
        self.rebalance = kwargs.get('rebalance', False)

        # Set surrounding window size for mention
        self.window_size = kwargs.get('window_size', 3)

        # Set relation type indicator (e.g., symmetric or asymmetric)
        self.asymmetric = kwargs.get('asymmetric', False)

        # Set max sentence length
        self.max_sentence_length = kwargs.get('max_sentence_length', 100)

        # Set kernel
        self.kernel = kwargs.get('kernel', None)

        # Set exp
        if self.kernel == 'exp':
            self.decay = kwargs.get('decay', 1.0)

        # Set host device
        self.host_device = kwargs.get('host_device', 'cpu')

        # Replace placeholders in embedding files
        self.replace = kwargs.get('replace', {})

        print "==============================================="
        print "Number of learning epochs:         ", self.n_epochs
        print "Learning rate:                     ", self.lr
        print "Ignore top l principal components: ", self.l
        print "Select top k principal components: ", self.r
        print "Batch size:                        ", self.batch_size
        print "Rebalance:                         ", self.rebalance
        print "Surrounding window size:           ", self.window_size
        print "Use sentence sequence:             ", self.sent_feat
        print "Use window sequence:               ", self.cont_feat
        print "Bidirectional:                     ", self.bidirectional
        print "Host device:                       ", self.host_device
        print "Use char embeddings:               ", self.char
        print "Char gram:                         ", self.char_gram
        print "Kernel:                            ", self.kernel
        if self.kernel == 'exp':
            print "Exp kernel decay:                  ", self.decay
        print "Word embedding size:               ", self.word_emb_dim
        print "Char embedding size:               ", self.char_emb_dim
        print "Word embedding:                    ", self.word_emb_path
        print "Char embedding:                    ", self.char_emb_path
        print "==============================================="

        assert self.word_emb_path is not None
        if self.char:
            assert self.char_emb_path is not None

        if "init_pretrained" in kwargs:
            del self.model_kwargs["init_pretrained"]

    def train(self, X_train, Y_train, X_dev=None, Y_dev=None, print_freq=5, dev_ckpt=True,
              dev_ckpt_delay=0.75, save_dir='checkpoints', **kwargs):

        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """

        self._init_kwargs(**kwargs)

        verbose = print_freq > 0

        # Set random seed
        torch.manual_seed(self.seed)
        if self.host_device in self.gpu:
            torch.cuda.manual_seed(self.seed)

        np.random.seed(seed=int(self.seed))

        # Set random seed for all numpy operations
        self.rand_state.seed(self.seed)

        # load embeddings from file
        self.load_embeddings()

        print "Done loading embeddings..."

        cardinality = Y_train.shape[1] if len(Y_train.shape) > 1 else 2
        if cardinality != self.cardinality:
            raise ValueError("Training marginals cardinality ({0}) does not"
                             "match model cardinality ({1}).".format(Y_train.shape[1],
                                                                     self.cardinality))
        # Make sure marginals are in correct default format
        Y_train = reshape_marginals(Y_train)
        # Make sure marginals are in [0,1] (v.s e.g. [-1, 1])
        if self.cardinality > 2 and not np.all(Y_train.sum(axis=1) - 1 < 1e-10):
            raise ValueError("Y_train must be row-stochastic (rows sum to 1).")
        if not np.all(Y_train >= 0):
            raise ValueError("Y_train must have values in [0,1].")

        if self.cardinality == 2:
            # This removes unlabeled examples and optionally rebalances
            train_idxs = LabelBalancer(Y_train).get_train_idxs(self.rebalance,
                                                               rand_state=self.rand_state)
        else:
            # In categorical setting, just remove unlabeled
            diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
            train_idxs = np.where(diffs > 1e-6)[0]
        X_train = [X_train[j] for j in train_idxs] if self.representation \
            else X_train[train_idxs, :]
        Y_train = Y_train[train_idxs]

        if verbose:
            st = time()
            print "[%s] n_train= %s" % (self.name, len(X_train))

        X_train = self._preprocess_data_combination(X_train)
        if X_dev is not None:
            X_dev = self._preprocess_data_combination(X_dev)
        Y_train = torch.from_numpy(Y_train).float()

        new_X_train = None
        for i in range(len(X_train)):
            feature = self.gen_feature(X_train[i])
            if new_X_train is None:
                new_X_train = torch.from_numpy(np.zeros((len(X_train), feature.size(1)), dtype=np.float)).float()
            new_X_train[i] = feature
        data_set = data_utils.TensorDataset(new_X_train, Y_train)
        train_loader = data_utils.DataLoader(data_set, batch_size=self.batch_size, shuffle=False)

        n_examples, n_features = new_X_train.size()
        n_classes = 1 if self.cardinality == 2 else None

        self.model = self.build_model(n_features, n_classes)
        loss = nn.MultiLabelSoftMarginLoss(size_average=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        dev_score_opt = 0.0

        for idx in range(self.n_epochs):
            cost = 0.
            for x, y in train_loader:
                cost += self.train_model(self.model, loss, optimizer, x, y.float())
            if verbose and ((idx + 1) % print_freq == 0 or idx + 1 == self.n_epochs):
                msg = "[%s] Epoch %s, Training error: %s" % (self.name, idx + 1, cost / n_examples)
                if X_dev is not None:
                    scores = self.score(X_dev, Y_dev, batch_size=self.batch_size)
                    score = scores if self.cardinality > 2 else scores[-1]
                    score_label = "Acc." if self.cardinality > 2 else "F1"
                    msg += '\tDev {0}={1:.2f}'.format(score_label, 100. * score)
                print msg

                if X_dev is not None and dev_ckpt and idx > dev_ckpt_delay * self.n_epochs and score > dev_score_opt:
                    dev_score_opt = score
                    self.save(save_dir=save_dir, only_param=True)

        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time() - st))

        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and verbose and dev_score_opt > 0:
            self.load(save_dir=save_dir, only_param=True)

    def _marginals_batch(self, X):
        new_X_train = None
        X = self._preprocess_data_combination(X)

        for i in range(len(X)):
            feature = self.gen_feature(X[i])
            if new_X_train is None:
                new_X_train = torch.from_numpy(np.zeros((len(X), feature.size(1))))
            new_X_train[i] = feature

        new_X_train = new_X_train.float()

        return self.predict(self.model, new_X_train)

    def embed(self, X):

        new_X_train = None
        X = self._preprocess_data_combination(X)

        for i in range(len(X)):
            feature = self.gen_feature(X[i])
            if new_X_train is None:
                new_X_train = torch.from_numpy(np.zeros((len(X), feature.size(1))))
            new_X_train[i] = feature

        return new_X_train.float().numpy()

    def save(self, model_name=None, save_dir='checkpoints', verbose=True,
             only_param=False):
        """Save current model."""
        model_name = model_name or self.name

        # Note: Model checkpoints need to be saved in separate directories!
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not only_param:
            # Save model kwargs needed to rebuild model
            with open(os.path.join(model_dir, "model_kwargs.pkl"), 'wb') as f:
                dump(self.model_kwargs, f)

            # Save model dicts needed to rebuild model
            with open(os.path.join(model_dir, "model_dicts.pkl"), 'wb') as f:
                if self.char:
                    dump({'char_dict': self.char_dict, 'word_dict': self.word_dict, 'char_emb': self.char_emb,
                          'word_emb': self.word_emb}, f)
                else:
                    dump({'word_dict': self.word_dict, 'word_emb': self.word_emb}, f)

        torch.save(self.model, os.path.join(model_dir, model_name))

        if verbose:
            print("[{0}] Model saved as <{1}>, only_param={2}".format(self.name, model_name, only_param))

    def load(self, model_name=None, save_dir='checkpoints', verbose=True, only_param=False):
        """Load model from file and rebuild in new graph / session."""
        model_name = model_name or self.name
        model_dir = os.path.join(save_dir, model_name)

        if not only_param:
            # Load model kwargs needed to rebuild model
            with open(os.path.join(model_dir, "model_kwargs.pkl"), 'rb') as f:
                model_kwargs = load(f)
                self._init_kwargs(**model_kwargs)

            # Save model dicts needed to rebuild model
            with open(os.path.join(model_dir, "model_dicts.pkl"), 'rb') as f:
                d = load(f)
                self.word_dict = d['word_dict']
                self.word_emb = d['word_emb']
                if self.char:
                    self.char_dict = d['char_dict']
                    self.char_emb = d['char_emb']

        self.model = torch.load(os.path.join(model_dir, model_name))

        if verbose:
            print("[{0}] Loaded model <{1}>, only_param={2}".format(self.name, model_name, only_param))

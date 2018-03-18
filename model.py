import torch
import torch.nn as nn
from torch.autograd import Variable as Var

BATCH_SIZE = 2 # 64

EMBED_SIZE = 50
HIDDEN_SIZE = 300
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4

LOG_EVERY  = 10
SAVE_EVERY = 100

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class lstm_crf(nn.Module):
    def __init__(self, vocab_size, num_tags):
        super().__init__()

        # architecture
        self.lstm = lstm(vocab_size, num_tags)
        self.crf = crf(num_tags)

    def forward(self, x, y0): # training
        y, lens = self.lstm(x) # [n, T, K]
        mask = x.data.gt(0).float() # [n, T]
        y = y * Var(mask.unsqueeze(-1).expand_as(y))
        score = self.crf.score(y, y0, mask) # [n]
        Z = self.crf.forward(y, mask) # [n]
        return -(score - Z) # negative log likelihood

    def decode(self, x): # prediction
        result = []
        y, lens = self.lstm(x)
        for i in range(len(lens)):
            if lens[i] > 1:
                best_path = self.crf.decode(y[i][:lens[i]])
            else:
                best_path = []
            result.append(best_path)
        return result

class crf(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags

        # matrix of transition scores to i from j
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000. # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000. # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000. # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000. # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0.
        self.trans.data[PAD_IDX, PAD_IDX] = 0.

    def score(self, y, y0, mask): # numerator / first term
        # for predicted tags and current transition matrix
        score = Var(Tensor(BATCH_SIZE).fill_(0.))
        y0 = torch.cat([LongTensor(BATCH_SIZE, 1).fill_(SOS_IDX), y0], 1) # [n, 1+T]
        for t in range(y.size(1)): # iterate through sequence
            mask_t = Var(mask[:, t])
            emit = torch.cat([y[i, t, y0[i, t + 1]] for i in range(BATCH_SIZE)])
            trans = torch.cat([self.trans[seq[t + 1], seq[t]] for seq in y0]) * mask_t
            score = score + emit + trans # [n]
        return score

    def forward(self, y, mask): # partition function Z / second term
        # over all possible tags and transitions
        # initialize forward variables (alpha: score) in log space
        # (would be clearer to add emit after first log_sum_exp)
        score = Tensor(BATCH_SIZE, self.num_tags).fill_(-10000.) # [n, K]
        score[:, SOS_IDX] = 0.
        score = Var(score)
        for t in range(y.size(1)): # iterate through sequence
            mask_t = Var(mask[:, t].unsqueeze(-1).expand_as(score)) # [n] -> [n, K]
            score_t = score.unsqueeze(1).expand(-1, *self.trans.size()) # [n, K] -> [n, K, K]
            emit = y[:, t].unsqueeze(-1).expand_as(score_t) # [n, K] -> [n, K, K]
            trans = self.trans.unsqueeze(0).expand_as(score_t) # [K, K] -> [n, K, K]
            score_t = log_sum_exp(score_t + emit + trans) # [n, K]
            score = score_t * mask_t + score * (1 - mask_t) # [n, K]
        score = log_sum_exp(score) # [n]
        return score

    def decode(self, y): # Viterbi decoding
        # initialize viterbi variables (delta: score) 
        # in log space, along with backpointers (psi: bptr)
        score = Tensor(self.num_tags).fill_(-10000.) # [K]
        score[SOS_IDX] = 0.
        score = Var(score)
        bptr = []

        for t in range(len(y)): # iterate through sequence
            z = score.unsqueeze(0).expand_as(self.trans) + self.trans # [K, K]
            score_t, bptr_t = torch.max(z, 1) # max over prev tags: [K], [K]
            score = score_t + y[t] # [K]
            bptr.append(bptr_t.data.tolist())
        best_tag = argmax(score) # for EOS
        best_score = score[best_tag] # for seq

        # back-tracking thru btpr: [T, K]
        best_path = [best_tag]
        for bptr_t in reversed(bptr):
            best_tag = bptr_t[best_tag]
            best_path.append(best_tag)
        best_path = reversed(best_path[:-1]) # skip EOS

        return best_path

class lstm(nn.Module):
    def __init__(self, vocab_size, num_tags):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = BIDIRECTIONAL
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags) # LSTM output to tag

    def init_hidden(self): # initialize hidden states
        h = Var(zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS)) # hidden states
        c = Var(zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS)) # cell states
        return (h, c)

    def forward(self, x):
        self.hidden = self.init_hidden()
        self.lens = [len_unpadded(seq) for seq in x]
        embed = self.embed(x) # [n, T, E]
        embed = nn.utils.rnn.pack_padded_sequence(embed, self.lens, batch_first=True)
        hiddens, _ = self.lstm(embed, self.hidden) # [n, T, H]
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)
        y = self.out(hiddens) # [n, T, K]
        return y, self.lens

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def len_unpadded(x): # get unpadded sequence length
    return next((i for i, w in enumerate(x) if scalar(w) == 0), len(x))

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x): # for 1D tensor
    return scalar(torch.max(x, 0)[1])

def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))

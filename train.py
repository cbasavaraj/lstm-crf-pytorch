import sys
import re
import time

from os.path import isfile

from model import *
from utils import *

DATA_PATH = "./data/"
CKPT_PATH = "./model/"

def load_data():
    data = []
    batch_x = []
    batch_y = []
    batch_len = 0 # maximum sequence length in mini-batch
    print("loading data...")
    word_to_idx = load_word_to_idx(DATA_PATH + sys.argv[2])
    tag_to_idx = load_tag_to_idx(DATA_PATH + sys.argv[3])
    fo = open(DATA_PATH + sys.argv[1], "r")
    for line in fo:
        line = line.strip()
        words = [int(i) for i in line.split(" ")]
        seq_len = words.pop()
        if len(batch_x) == 0: # the first line has the maximum sequence length
            batch_len = seq_len
        pad = [PAD_IDX] * (batch_len - seq_len)
        batch_x.append(words[:seq_len] + [EOS_IDX] + pad)
        batch_y.append(words[seq_len:] + [EOS_IDX] + pad)
        if len(batch_x) == BATCH_SIZE:
            data.append((Var(LongTensor(batch_x)), LongTensor(batch_y))) # append a mini-batch
            batch_x = []
            batch_y = []
    fo.close()
    print("batch size: %d" % BATCH_SIZE)
    print("number of mini-batches: %d" % (len(data)))
    return data, word_to_idx, tag_to_idx

def train():
    num_epochs = int(sys.argv[4])
    data, word_to_idx, tag_to_idx = load_data()
    print(word_to_idx)
    print(tag_to_idx)
    #print(data)
    model = lstm_crf(len(word_to_idx), len(tag_to_idx))
    if CUDA:
        model = model.cuda()
    print(model)
    optim = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    if len(sys.argv) == 6 and isfile(sys.argv[5]):
        epoch = load_checkpoint(sys.argv[5], model)
        filename = re.sub("\.epoch[0-9]+$", "", sys.argv[5])
    else:
        epoch = 0
        filename = "model.ckpt"
    print("training model...")
    for i in range(epoch + 1, epoch + num_epochs + 1):
        loss_epoch = 0
        tic = time.time()
        for x, y in data:
            model.zero_grad()
            loss = torch.sum(model(x, y)) # forward pass
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss = scalar(loss)
            loss_epoch += loss
        toc = time.time()
        loss_epoch /= len(data) * BATCH_SIZE
        if i % LOG_EVERY == 0:
            log = "epoch = %3d, loss = %5.2f, time = %4.2fs" % (i, loss_epoch, toc - tic)
            print(log)
        if i % SAVE_EVERY == 0 or i == epoch + num_epochs:
            save_checkpoint(CKPT_PATH + filename, model, i, loss_epoch)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit("Usage: %s training_data word_to_idx tag_to_idx num_epoch model" % sys.argv[0])
    print("cuda: %s" % CUDA)
    train()

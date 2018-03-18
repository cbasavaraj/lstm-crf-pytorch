import sys
import re

from model import *
from utils import *

DATA_PATH = "./data/"
CKPT_PATH = "./model/"

def load_model():
    word_to_idx = load_word_to_idx(DATA_PATH + sys.argv[2])
    tag_to_idx = load_tag_to_idx(DATA_PATH + sys.argv[3])
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    model = lstm_crf(len(word_to_idx), len(tag_to_idx))
    if CUDA:
        model = model.cuda()
    print(model)
    load_checkpoint(CKPT_PATH + sys.argv[4], model)
    return model, word_to_idx, tag_to_idx, idx_to_tag

def run_model(model, idx_to_tag, data):
    while len(data) < BATCH_SIZE:
        data.append(("", [EOS_IDX]))
    data.sort(key = lambda x: len(x[1]), reverse=True)
    batch_len = len(data[0][1])
    lines = []
    batch = []
    for line, idxs in data:
        lines.append(line)
        batch.append(idxs + [PAD_IDX] * (batch_len - len(idxs)))
    batch = Var(LongTensor(batch))
    preds = []
    for out in model.decode(batch):
        preds.append([idx_to_tag[idx] for idx in out])
    return [(line, pred[:-1]) for line, pred in zip(lines, preds) if len(line) > 0]

def predict():
    data = []
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    fo = open(DATA_PATH + sys.argv[1])
    print("predicting tags...")
    for line in fo:
        line = re.sub("\s+", " ", line)
        line = re.sub("^ | $", "", line)
        line = line.split(" ")
        data.append((line, [word_to_idx[w] for w in line] + [EOS_IDX]))
        if len(data) == BATCH_SIZE:
            results = run_model(model, idx_to_tag, data)
            for result in results:
                print([(word, tag) for word, tag in zip(result[0], result[1])])
            data = []
    fo.close()
    if len(data):
        results = run_model(model, idx_to_tag, data)
        for result in results:
            print([(word, tag) for word, tag in zip(result[0], result[1])])

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s test_data word_to_idx tag_to_idx model" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()

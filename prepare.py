import sys
import re

from model import SOS, EOS, PAD, SOS_IDX, EOS_IDX, PAD_IDX
from utils import normalize

MIN_LENGTH = 10
MAX_LENGTH = 50

DATA_PATH = "./data/"

def prepare_data(): # word-level
    word_to_idx = {PAD: PAD_IDX, EOS: EOS_IDX}
    tag_to_idx = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
    data = []
    fo = open(DATA_PATH + sys.argv[1])
    for line in fo:
        line = re.sub("\s+", " ", line)
        line = re.sub("^ | $", "", line)
        tokens = line.split(" ")
        if len(tokens) < MIN_LENGTH or len(tokens) > MAX_LENGTH: # length constraints
            continue
        sent = []
        tags = []
        for tkn in tokens:
            word, tag = tkn.split('/')
            word = normalize(word)
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
            sent.append(word_to_idx[word])
            tags.append(tag_to_idx[tag])
        print(sent)
        print(tags)
        data.append(sent + tags)
    data.sort(key = len, reverse = True)
    fo.close()
    return data, word_to_idx, tag_to_idx

def save_data(data):
    fo = open(DATA_PATH + "train.txt", "w")
    for seq in data:
        fo.write("%s %d\n" % (" ".join([str(i) for i in seq]), len(seq) // 2))
    fo.close()

def save_word_to_idx(word_to_idx):
    fo = open(DATA_PATH + "word_to_idx.txt", "w")
    for word, _ in sorted(word_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % word)
    fo.close()

def save_tag_to_idx(tag_to_idx):
    fo = open(DATA_PATH + "tag_to_idx.txt", "w")
    for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tag)
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_corpus" % sys.argv[0])
    data, word_to_idx, tag_to_idx = prepare_data()
    save_data(data)
    save_word_to_idx(word_to_idx)
    save_tag_to_idx(tag_to_idx)

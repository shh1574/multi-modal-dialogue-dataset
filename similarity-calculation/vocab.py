# Create a vocabulary wrapper
import nltk
import pickle
from collections import Counter
import json
import argparse
import os

annotations = {
    'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'coco': ['annotations/captions_train2014.json',
             'annotations/captions_val2014.json'],
    'f8k_precomp': ['train_caps.txt', 'dev_caps.txt'],
    '10crop_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f30k_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f8k': ['dataset_flickr8k.json'],
    'f30k': ['dataset_flickr30k.json'],
}


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab_for_coco_flickr(data_path, jsons, threshold):
    """Build a simple vocabulary wrapper."""
    paths = ['coco_train_caps.txt', 'coco_dev_caps.txt', 'flickr_train_caps.txt', 'flickr_dev_caps.txt']
    counter = Counter()
    for path in paths:
        full_path = os.path.join(data_path, path)
        captions = from_txt(full_path)
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
            counter.update(tokens)

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(data_path, data_name):
    vocab = build_vocab_for_coco_flickr(data_path, jsons=annotations, threshold=4)
    with open('./vocab/coco_flickr_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", './vocab/coco_flickr_vocab.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../context_aware/data')
    parser.add_argument('--data_name', default='coco',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)

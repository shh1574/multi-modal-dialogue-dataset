from __future__ import print_function
import os
import pickle

import torch
import numpy
import shutil
import random

import nltk

import numpy as np
from model import VSRN
from tqdm import tqdm
from vocab import Vocabulary

def prepare_caption(vocab, text_path, text_file):
    captions = []
    captions_with_length = []
    captions = []

    with open(os.path.join(text_path, text_file), 'rb') as f:
        for i, line in enumerate(f):
            tokens = nltk.tokenize.word_tokenize(line.strip().decode('utf-8').lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            captions_with_length.append((torch.Tensor(caption), len(caption), str(i)))

    captions_with_length.sort(key=lambda x: x[1], reverse=True)
    captions, lengths, orders = zip(*captions_with_length)
            
    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return targets, lengths, orders

def calculate_similarities_vsrn(model_path, model_path2, img_dataset, mode='all', split='train', data_path='dataset/'):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']

    # load vocabulary used by the model
    
    vocab_path = os.path.join(data_path, 'vocab')

    vocab = pickle.load(open(os.path.join(vocab_path, '{}_precomp_vocab.pkl'.format(img_dataset)), 'rb'))

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    model = VSRN(opt)
    model.load_state_dict(checkpoint['model'])

    model2 = VSRN(opt2)
    model2.load_state_dict(checkpoint2['model'])

    print('Dataset Loading...')

    # thresholds = [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 10]
    thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1]

    print('Image Loading...')
    images = np.load(os.path.join(data_path, '{}_{}_ims.npy'.format(img_dataset, split)))
    if split == 'dev' or split == 'test':
        images = np.array([images[i] for i in range(len(images)) if i % 5 == 0])

    print('Image shape : ', len(images))
    img_embs = None
    img_embs2 = None

    print('Image Processing...')
    for i in tqdm(range(len(images) // 100)):
        image = torch.Tensor(images[i * 100 : (i + 1) * 100])
        ids = [i for i in range(i * 100, (i + 1) * 100)]

        img_emb = model.forward_emb_img(image)
        img_emb2 = model2.forward_emb_img(image)

        if img_embs is None:
            img_embs = np.zeros((len(images), img_emb.size(1)))
            img_embs2 = np.zeros((len(images), img_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        img_embs2[ids] = img_emb2.data.cpu().numpy().copy()

        del image, img_emb, img_emb2
    del images

    print("Encoded Image shape : ", img_embs.shape)

    dialog_datasets = ['persona', 'empathy', 'dailydialog']
    # dialog_datasets = ['dailydialog']

    for dialog_dataset in dialog_datasets:
        ranges2idx = [[] for _ in range(len(thresholds))]

        write_path = 'output/{}_{}_{}_{}'.format(mode, dialog_dataset, img_dataset, split)
        
        if not os.path.exists(write_path):
            print('Making directory... {}'.format(write_path))
            os.makedirs(write_path)
        else:
            res = input('Do you want to remove the directory : {} '.format(write_path))
            if res == 'y':
                shutil.rmtree(write_path)
                os.makedirs(write_path)
            else:
                return

        text_path = os.path.join(data_path, '{}_stopwords'.format(dialog_dataset))
        text_files = [f for f in os.listdir(text_path) if f.endswith('.txt')]
        # text_files = ['test.txt']
        cnt = [0 for _ in range(len(thresholds))]

        print("Text Processing...")
        for text_file in tqdm(text_files[:100]):
            # print(text_file)
            captions, lengths, orders = prepare_caption(vocab, text_path, text_file)

            cap_embs = model.forward_emb_cap(captions, lengths)
            cap_embs = cap_embs.data.cpu().numpy().copy()

            cap_embs2 = model2.forward_emb_cap(captions, lengths)
            cap_embs2 = cap_embs2.data.cpu().numpy().copy()

            if mode == 'all':
                range_idx, final_cnt = t2i_vsrn_range(img_embs, img_embs2, cap_embs, cap_embs2, text_file, orders)
                for j, c in enumerate(final_cnt):
                    cnt[j] += c
                # print(range_idx)
                for j, info in enumerate(range_idx):
                    if info is not None:
                        ranges2idx[j].append(info)

            elif mode == 'threshold':
                upper_threshold, exist = t2i_threshold(img_embs, cap_embs, img_embs2, cap_embs2)
                if not exist:
                    continue
                with open(os.path.join(write_path, text_file), 'w') as f:
                    for o, u in zip(orders, upper_threshold):
                        if u == '':
                            continue
                        f.write('{} {}\n'.format(o, u))
        
        if mode == 'all':
            for info in ranges2idx:
                random.shuffle(info)

            for threshold, idx in zip(thresholds, ranges2idx):
                if idx is None:
                    continue

                file_name = '{}.txt'.format(threshold)
                with open(os.path.join(write_path, file_name), 'w') as f:
                    for info in idx[:min(1500, len(idx))]:
                        f.write('{} {} {} {}\n'.format(info[0], info[1], info[2], info[3]))


def t2i_vsrn_range(images, images2, captions, captions2, file_name, orders):
    npts = int(captions.shape[0])
    thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1]
    final_cnt = [0 for _ in range(len(thresholds))]
    
    d = numpy.dot(captions, images.T)
    d2 = numpy.dot(captions2, images2.T)
    d = (d + d2) / 2

    range_idx = [[] for _ in range(len(thresholds))]

    for i, order in enumerate(orders):
        tmp_cnt = [0]
        for j in range(len(thresholds)):
            tmp_cnt.append(numpy.count_nonzero(d[i] <= thresholds[j]))

        cnt = []
        for j in range(len(tmp_cnt) - 1):
            cnt.append(tmp_cnt[j + 1] - tmp_cnt[j])

        for j, c in enumerate(cnt):
            final_cnt[j] += c

        start = 0
        argsorted_array = numpy.argsort(d[i])

        for j, c in enumerate(cnt):
            if c != 0:
                rand_idx = int(random.choice(argsorted_array[start:start + c]))
                range_idx[j].append((file_name, order, rand_idx, d[i][rand_idx]))

            start += c

        final_range_idx = [random.choice(idx) if len(idx) != 0 else None for idx in range_idx]

    return final_range_idx, final_cnt


def t2i_threshold(images, captions, images2, captions2, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = int(captions.shape[0])
    threshold = 0.5

    # ranks = numpy.zeros(npts)
    # top1 = numpy.zeros(npts)

    upper_threshold = []

    d = numpy.dot(captions, images.T)
    d2 = numpy.dot(captions2, images2.T)
    d = (d + d2) / 2
    exist = False

    for i in range(npts):
        top_idx = numpy.argsort(d[i])[::-1][:numpy.count_nonzero(d[i] >= threshold)]
        tmp = ''
        for t in top_idx:
            tmp += '{} {} '.format(t, d[i][int(t)])
            exist = True
        upper_threshold.append(tmp)

    return upper_threshold, exist

def main():
    img_datasets = ['coco', 'flickr']
    splits = ['test', 'dev', 'train']
    mode = 'all'

    data_path = 'dataset'

    for img_dataset in img_datasets:
        for split in splits:
            calculate_similarities_vsrn(
                os.path.join(data_path, 'model_{}_1.pth.tar'.format(img_dataset)),
                os.path.join(data_path, 'model_{}_2.pth.tar'.format(img_dataset)),
                img_dataset,
                mode=mode,
                split=split,
                data_path=data_path
            )

if __name__ == "__main__":
    main()



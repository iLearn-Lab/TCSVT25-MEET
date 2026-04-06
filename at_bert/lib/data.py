# encoding:utf-8
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
from transformers import BertTokenizer, BertModel
import random
from pathlib import Path

LIRONG_ROOT = Path(__file__).resolve().parents[2]
BERT_ROOT = LIRONG_ROOT / 'data' / 'bert-base-uncased'

class PrecompDataset_bert(data.Dataset):
    def __init__(self, data_path, data_split, opt):
        loc = data_path + '/'
        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.tokenizer = BertTokenizer.from_pretrained(str(BERT_ROOT))
        # Raw captions
        self.captions = []
        with open(os.path.join(loc, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # num_captions
        self.length = len(self.captions)

        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

    def __getitem__(self, index):
        img_index = index // self.im_div
        image = torch.Tensor(self.images[img_index])
        caption = self.captions[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        target = process_caption_bert(self.tokenizer, caption_tokens, False)

        return image, target, index, img_index

    def __len__(self):
        return self.length

def collate_fn_bert(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, ids, img_ids = zip(*data)

    img_ids = torch.tensor(img_ids)
    ids = torch.tensor(ids)

    # print(img_ids)
    repeat = len(img_ids) - len(torch.unique(img_ids))

    # Sort a data list by caption length
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # images = torch.stack(images, 0)

    img_lengths = [len(image) for image in images]

    # dataset_size * max_lengths (maybe 36) * 2048
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]

    img_lengths = torch.tensor(img_lengths)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # count the length of each captions
    lengths = [len(cap) for cap in captions]

    # pad the redundancy with zero, in order to input BERT model as a batch
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = torch.tensor(lengths)

    # all_images: Batch_size * max_img_lengths * 2048 (the dimension of region-features)
    # targets:  Batch_size * max_cap_lengths

    return all_images, img_lengths, targets, lengths, ids, img_ids, repeat

def process_caption_bert(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        # text -> token (basic_tokenizer.tokenize) -> sub_token (wordpiece_tokenizer.tokenize)
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)

        prob = random.random()

        # first, 20% probability use the augmenation operations
        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token from the BERT-vocab
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> 40% delete the token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    # record the index of sub_token
                    deleted_idx.append(len(output_tokens) - 1)
        # 80% probability keep the token
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    # and first and last notations for BERT model
    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']

    # Convert token to vocabulary indices, torch.float32
    target = tokenizer.convert_tokens_to_ids(output_tokens)

    # convert to the torch-tenfor
    target = torch.Tensor(target)
    return target

#hello data
class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'
        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        # if data_split == 'dev':
        #     self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab
        # print("data.py line48 caption", caption)
        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(
        #     str(caption).lower().encode('utf-8').decode('utf-8'))
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (num, 36, 2048).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 36, 2048).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # dset = PrecompDataset(data_path, data_split, vocab)
    dset = PrecompDataset_bert(data_path, data_split, opt)
    # data_loader = torch.utils.data.DataLoader(dataset=dset,
    #                                           batch_size=batch_size,
    #                                           shuffle=shuffle,
    #                                           pin_memory=False,
    #                                           collate_fn=collate_fn,
    #                                           num_workers=num_workers)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn_bert,
                                              num_workers=num_workers)
    return data_loader,dset.length

def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader,train_len = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader,val_len = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)

    return train_loader, val_loader,train_len,val_len

def get_test_loader_data(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    print(data_name)
    test_loader,test_len = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader,test_len

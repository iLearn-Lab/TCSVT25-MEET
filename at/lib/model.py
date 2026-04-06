"""SGRAF model"""

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from collections import OrderedDict
from .where_cuda import device

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=True)

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] +
                   cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len

class Hash_EncoderImage(nn.Module):

    def __init__(self, embed_size, no_imgnorm=False):
        super(Hash_EncoderImage, self).__init__()
        self.v_global = VisualSA(embed_size, 0.4, 36)

        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

    def forward(self, img_emb):

        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global(img_emb, img_ave)
        img_glo =  img_glo.view(img_glo.shape[0],1,-1)

        return img_glo

class Hash_EncoderText(nn.Module):

    def __init__(self, embed_size, no_txtnorm=False):
        super(Hash_EncoderText, self).__init__()

        self.u_global = TextSA(embed_size, 0.4)
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

    def forward(self, cap_emb):
        cap_ave = torch.mean(cap_emb, 1)
        cap_global = self.u_global(cap_emb, cap_ave)
        cap_global = cap_global.view(cap_global.shape[0],1,-1)
        return cap_global

class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)

        new_global = l2norm(new_global, dim=-1)

        return new_global

class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global

def xattn_score_hash(images, captions,opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = 1
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        t2i_sim = cosine_sim(cap_i_expand, images, dim=2)
        i2t_sim = cosine_sim(images,cap_i_expand, dim=2)

        sim = t2i_sim + i2t_sim
        sim = sim.view(-1,1)

        similarities.append(sim)
    similarities = torch.cat(similarities, 1)
    return similarities

#hash ranking loss
class ranking_loss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ranking_loss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix

        scores = xattn_score_hash(im, s, self.opt)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)

        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5

        if torch.cuda.is_available():
            I = mask.cuda(device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class HQ(object):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)

        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   no_txtnorm=opt.no_txtnorm)

        self.img_hcode = Hash_EncoderImage(opt.embed_size, no_imgnorm=False)
        self.txt_hcode = Hash_EncoderText(opt.embed_size, no_txtnorm=False)
        self.img_proj = nn.Sequential(
            nn.Linear(1024, opt.H),
            nn.Tanh()
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(1024, opt.H),
            nn.Tanh()
        )

        if torch.cuda.is_available():
            self.img_enc.cuda(device)
            self.txt_enc.cuda(device)

            self.img_hcode.cuda(device)
            self.txt_hcode.cuda(device)
            self.img_proj.cuda(device)
            self.txt_proj.cuda(device)
            cudnn.benchmark = True

        # Loss and Optimizer
        self.h_rankingloss = ranking_loss(opt=opt,
                                          margin=opt.margin,
                                          max_violation=opt.max_violation)
        cam_params = list(self.txt_enc.parameters())
        cam_params += list(self.img_enc.parameters())

        hash_params = list(self.img_hcode.parameters())
        hash_params += list(self.txt_hcode.parameters())
        hash_params += list(self.img_proj.parameters())
        hash_params += list(self.txt_proj.parameters())

        self.cam_params = cam_params
        self.hash_params = hash_params

        self.cam_optimizer = torch.optim.Adam(cam_params, lr=opt.learning_rate)
        self.hash_optimizer = torch.optim.Adam(self.hash_params, lr=opt.learning_hash_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.img_hcode.state_dict(),self.txt_hcode.state_dict(),self.img_proj.state_dict(),self.txt_proj.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

        self.img_hcode.load_state_dict(state_dict[2])
        self.txt_hcode.load_state_dict(state_dict[3])
        self.img_proj.load_state_dict(state_dict[4])
        self.txt_proj.load_state_dict(state_dict[5])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()

        self.img_hcode.train()
        self.txt_hcode.train()
        self.img_proj.train()
        self.txt_proj.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()

        self.img_hcode.eval()
        self.txt_hcode.eval()
        self.img_proj.eval()
        self.txt_proj.eval()

    def forward_emb(self, images, captions, lengths):
        # print('(((((((((((((')
        # print(images.shape)
        # print(captions.shape)
        # raise ValueError
        '''
        torch.Size([128, 36, 2048])
        torch.Size([128, 45])
        '''
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda(device)
            captions = captions.cuda(device)

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs,cap_lens = self.txt_enc(captions, lengths)

        img_out = self.img_hcode(img_embs)
        txt_out = self.txt_hcode(cap_embs)
        cp_img_out = img_out.clone().detach()
        cp_txt_out = txt_out.clone().detach()
        cp_img_out = self.img_proj(cp_img_out)
        cp_txt_out = self.txt_proj(cp_txt_out)
        return img_out, txt_out, cp_img_out,cp_txt_out

    def eval_emb(self,images, captions, lengths):

        if torch.cuda.is_available():
            images = images.cuda(device)
            captions = captions.cuda(device)

        with torch.no_grad():

            img_emb = self.img_enc(images)
            cap_emb, cap_lens = self.txt_enc(captions, lengths)
            img_out = self.img_hcode(img_emb)
            txt_out = self.txt_hcode(cap_emb)

            cp_img_out = img_out.clone().detach()
            cp_txt_out = txt_out.clone().detach()
            cp_img_out = self.img_proj(cp_img_out)
            cp_txt_out = self.txt_proj(cp_txt_out)
        return img_out, txt_out, cp_img_out,cp_txt_out

    def forward_loss(self, img_out, txt_out, cp_img_out, cp_txt_out, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        hash_ranking_loss = self.h_rankingloss(img_out, txt_out)
        hash_new_loss = self.h_rankingloss(cp_img_out, cp_txt_out)
        loss = hash_ranking_loss + hash_new_loss
        # print("model.py line730: hash_rankloss", hash_ranking_loss.item())
        self.logger.update('Le', loss.item())
        return loss

    def train_emb(self, images, captions, lengths, ids, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('cam_lr', self.cam_optimizer.param_groups[0]['lr'])
        self.logger.update('hash_lr', self.hash_optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_out, txt_out, cp_img_out, cp_txt_out = self.forward_emb(images, captions, lengths)
        # print('****************')
        # print(images.shape)
        # print(captions.shape)
        # print(img_out.shape)
        # print(txt_out.shape)
        # print(cp_img_out.shape)
        # print(cp_txt_out.shape)
        # if self.Eiters == 3:
        #     raise ValueError('stop')
        '''
****************
torch.Size([128, 36, 2048])
torch.Size([128, 47])
torch.Size([128, 1, 1024])
torch.Size([128, 1, 1024])
torch.Size([128, 1, 64])
torch.Size([128, 1, 64])
****************
torch.Size([128, 36, 2048])
torch.Size([128, 31])
torch.Size([128, 1, 1024])
torch.Size([128, 1, 1024])
torch.Size([128, 1, 64])
torch.Size([128, 1, 64])
        '''

        # measure accuracy and record loss
        self.cam_optimizer.zero_grad()
        self.hash_optimizer.zero_grad()
        loss = self.forward_loss(img_out, txt_out, cp_img_out, cp_txt_out)
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.cam_params, self.grad_clip)
            clip_grad_norm_(self.hash_params, self.grad_clip)
        self.cam_optimizer.step()
        self.hash_optimizer.step()


# encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from .image_caption import get_test_loader
from .data import *
import time
import os
from .model import HQ
from .vse import VSEModel
from .vocab import deserialize_vocab
from .utils import *
import torch
import sys
import cupy as cp
import logging
from transformers import BertTokenizer
logger = logging.getLogger(__name__)
from .where_cuda import device,device_string
import csv
from .data import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from pathlib import Path
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)

AT_BERT_ROOT = Path(__file__).resolve().parents[1]
LIRONG_ROOT = AT_BERT_ROOT.parent
DATA_ROOT = LIRONG_ROOT / 'data'
NPY_ROOT = AT_BERT_ROOT / 'npys'

def encode_data_train(model, data_loader,img_output ,txt_output):

    model.val_start()
    for i, (images, image_lengths, captions, lengths, ids, img_ids, repeat) in enumerate(data_loader):
        # if i %200 == 0:
        #     print("evalution line421 the num of i is",i)

        _,_,img_emb,cap_emb = model.eval_emb(images, captions, img_ids, lengths, image_lengths) #torch.Size([128, 1, 64])

        for i,tempid in enumerate(ids):
            img_output[tempid] = img_emb[i]
            txt_output[tempid] = cap_emb[i]

        del images, captions
    return img_output,txt_output

def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    for i, (images, image_lengths, captions, lengths, ids, img_ids, repeat) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        # compute the embeddings
        img_emb,cap_emb,cp_img_out,cp_txt_out = model.eval_emb(images, captions, img_ids, lengths, image_lengths)

        if img_embs is None:

            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1),img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1),cap_emb.size(2)))
            cp_img_outs = np.zeros((len(data_loader.dataset), cp_img_out.size(1), cp_img_out.size(2)))
            cp_txt_outs = np.zeros((len(data_loader.dataset), cp_txt_out.size(1), cp_txt_out.size(2)))
        # cache embeddings
        temp_img_emb = img_emb.data.cpu().numpy().copy()
        temp_cap_emb = cap_emb.data.cpu().numpy().copy()
        temp_cp_img_out = cp_img_out.data.cpu().numpy().copy()
        temp_cp_txt_out = cp_txt_out.data.cpu().numpy().copy()

        for i,tempid in enumerate(ids):

            img_embs[tempid] = temp_img_emb[i]
            cap_embs[tempid] = temp_cap_emb[i]

            cp_img_outs[tempid] = temp_cp_img_out[i]
            cp_txt_outs[tempid] = temp_cp_txt_out[i]
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
    return img_embs, cap_embs,cp_img_outs,cp_txt_outs

def HammingD(f1, f2, R_rate=0.2):
    '''
    '''
    f1 = f1.cpu().detach().numpy()
    f2 = f2.cpu().detach().numpy()

    A = f1.reshape(f1.shape[0], -1)
    B = f2.reshape(f2.shape[0], -1)

    A = 2*np.int8(A>=0) -1
    B = 2*np.int8(B>=0) -1

    dis = np.dot(A, np.transpose(B))
    ids = np.argsort(-dis, 1)

    if ids.shape[0] < ids.shape[1]:
        npts = ids.shape[0]
        ranks = np.zeros(npts)
        for i in range(npts):

            rank = 1e9

            for index in range(5 * i, 5 * i + 5, 1):

                tmp = np.where(ids[i] == index)[0][0]

                if tmp < rank:
                    rank = tmp
            ranks[i] = rank

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        medr = np.floor(np.median(ranks)) + 1

        meanr = ranks.mean() + 1

        okr = 100.0 * len(np.where(ranks < int(ids.shape[1] *R_rate))[0]) / len(ranks)
        print("evalution.py line182 Hash search---Image2Text %.1f,%.1f,%.1f %.1f %.1f %.1f" % (r1,r5,r10, medr,meanr,okr))
    # Text2image:

    else:
        npts = ids.shape[0]
        ranks = np.zeros(npts)
        for i in range(npts):
            tmp = i // 5
            rank = np.where(ids[i] == tmp)[0][0]
            ranks[i] = rank

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        okr = 100.0 * len(np.where(ranks < int(ids.shape[1] * R_rate))[0]) / len(ranks)
        print("evalution.py line201 Hash search---Text2image %.1f,%.1f,%.1f %.1f %.1f %.1f" % (r1,r5,r10,medr,meanr,okr))

    R_h = int(ids.shape[1] * R_rate)

    ID = ids[:, 0:R_h]
    print("evalution.py line207 Hash_ID.shape",ID.shape)
    return ID

def AQD_i2t(C,img_out, txt_code_q, id_list,q_rate,return_ranks = False):
    '''
    '''

    txt_code_q = cp.asarray(txt_code_q.cpu().numpy())
    img_out= cp.asarray(img_out.cpu().detach().numpy())
    C = cp.asarray(C.cpu().numpy())
    id_list = cp.asarray(id_list)

    #(img_size,1,64)--->(img_size,64)
    img_out = img_out.reshape(img_out.shape[0],-1)
    npts = img_out.shape[0]

    ranks = cp.zeros(npts)
    top1 = cp.zeros(npts)

    newdis = cp.zeros((npts, id_list.shape[1]))
    ans_idxs = cp.zeros((npts, id_list.shape[1]))

    dis = cp.dot(img_out, cp.dot(txt_code_q, C).T)

    for i in range(npts):

        filtered_id = id_list[i]

        newdis[i] = dis[i][filtered_id]

    idxs = cp.argsort(-newdis, 1)

    for i in range(npts):
        filtered_id = id_list[i]

        ans_idxs[i] = filtered_id[idxs[i]]

    for i in range(npts):
        rank = 1e9
        for index in range(5 * i, 5 * i + 5, 1):
            is_include = cp.where(ans_idxs[i] == index)[0]

            if len(is_include) != 0:

                tmp = is_include[0]
                if tmp < rank:
                    rank = tmp
        ranks[i] = rank
        top1[i] = ans_idxs[i][0]

    # Compute metrics
    r1 = 100.0 * len(cp.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(cp.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(cp.where(ranks < 10)[0]) / len(ranks)
    okr = 100.0 * len(cp.where(ranks < int(id_list.shape[1] * q_rate))[0]) / len(ranks)
    print("evalution.py line271 quantization search---Image2Text %.1f,%.1f,%.1f %.1f" % (r1, r5, r10, okr))

    R_h = int(id_list.shape[1]*q_rate)
    ID = ans_idxs[:,:R_h]

    ID = cp.asnumpy(ID).astype(int)
    print("evalution.py line277 Quantization_ID.shape", ID.shape)

    return ID

def AQD_t2i( C,txt_out, img_code_q, id_list,q_rate,return_ranks = False):

    img_code_q = cp.asarray(img_code_q.cpu().numpy())
    txt_out= cp.asarray(txt_out.cpu().detach().numpy())
    txt_out = txt_out.reshape(txt_out.shape[0],-1)
    C = cp.asarray(C.cpu().numpy())
    id_list = cp.asarray(id_list)

    npts = txt_out.shape[0]
    ranks = cp.zeros(npts)
    top1 = cp.zeros(npts)

    newdis = cp.zeros((npts, id_list.shape[1]))
    ans_idxs = cp.zeros((npts, id_list.shape[1]))
    dis = cp.dot(txt_out, cp.dot(img_code_q, C).T)
    for i in range(npts):
        filtered_id = id_list[i]
        newdis[i] = dis[i][filtered_id]
    idxs = cp.argsort(-newdis, 1)

    for i in range(npts):
        filtered_id = id_list[i]
        ans_idxs[i] = filtered_id[idxs[i]]

    for i in range(npts):
        rank = 1e9
        txt_index = i//5
        is_include = cp.where(ans_idxs[i] == txt_index)[0]
        if len(is_include)!=0:
            rank =is_include[0]
        ranks[i] = rank
        top1[i] = ans_idxs[i][0]

    # Compute metrics
    r1 = 100.0 * len(cp.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(cp.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(cp.where(ranks < 10)[0]) / len(ranks)
    okr = 100.0 * len(cp.where(ranks < int(img_code_q.shape[0] * q_rate))[0]) / len(ranks)
    print("evalution.py line322 quantization search---Text2image %.1f,%.1f,%.1f %.1f " % ( r1, r5, r10, okr))

    R_h = int(id_list.shape[1] * q_rate)

    ID = ans_idxs[:, :R_h]
    ID = cp.asnumpy(ID).astype(int)
    print("evalution.py line329 Quantization_ID.shape", ID.shape)
    return ID

def i2t_rerank(sim, K1):
    '''
        take i2t direction as an example
        sim: (img_size,text_size)
    '''

    size_i = sim.shape[0]

    sort_i2t = np.argsort(-sim, 1)

    sort_t2i = np.argsort(-sim, 0)

    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    for i in range(size_i):
        for j in range(K1):

            result_t = sort_i2t[i][j]

            query = sort_t2i[:, result_t]

            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_i2t_re[i] = sort_i2t_re[i][sort]
        address = np.array([])

    sort_i2t[:,:K1] = sort_i2t_re

    return sort_i2t

def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities

K_i2t = 15
K_t2i = 12

def afterhash_i2t(img_embs, cap_embs, id_list):
    npts = img_embs.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    newsim = np.zeros((img_embs.shape[0],cap_embs.shape[0]))
    for i in range(newsim.shape[0]):
        newsim[i] = -10

    for i in range(npts):
        # sys.stdout.write('\r>> afterhash_i2t batch (%d,%d)' % (i // 100, i % 100))
        filtered_id = id_list[i]

        temp_img_emb = img_embs[i]
        temp_cap_emb = cap_embs[filtered_id, :]

        sim = compute_sim(temp_img_emb, temp_cap_emb)

        dis = sim
        dis = dis.reshape(dis.shape[0])

        for j in range(dis.shape[0]):
            newsim[i,filtered_id[j]] = dis[j]
    sys.stdout.write('\n')

    idxs = i2t_rerank(newsim, K1=K_i2t)

    # print(newsim.shape)
    for i in range(npts):
        rank = 1e9
        for index in range(5 * i, 5 * i + 5, 1):
            tmp = np.where(idxs[i] == index)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[i] = rank
        top1[i] = idxs[i][0]
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    print("evalution.py line416 Recall of afterhash Image2Text is %.1f %.1f %.1f" % (r1, r5, r10))

    return (r1, r5, r10, idxs)

def afterhash_t2i(img_embs, cap_embs, id_list):
    npts = cap_embs.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    newsim = np.zeros((cap_embs.shape[0], img_embs.shape[0]))
    for i in range(newsim.shape[0]):
        newsim[i] = -10

    for i in range(npts):
        # sys.stdout.write('\r>> afterhash_t2i batch (%d,%d)' % (i // 100, i % 100))
        filtered_id = id_list[i]
        temp_cap_emb = cap_embs[i]
        temp_img_emb = img_embs[filtered_id, :]

        sim = compute_sim(temp_img_emb, temp_cap_emb)
        dis = sim
        for j in range(dis.shape[0]):
            newsim[i,filtered_id[j]] = dis[j]
    sys.stdout.write('\n')

    idxs = i2t_rerank(newsim, K1=K_t2i)

    for i in range(npts):
        txt_index = i // 5
        rank = np.where(idxs[i] == txt_index)[0][0]
        ranks[i] = rank
        top1[i] = idxs[i][0]
    # for i in range(npts):
    #     idxs = np.argsort(newsim[i])[::-1]
    #     txt_index = i // 5
    #     rank = np.where(idxs == txt_index)[0][0]
    #     ranks[i] = rank
    #     top1[i] = idxs[0]
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    print("evalution.py line503 Recall of afterhash Text2Image is %.1f %.1f %.1f" % (r1, r5, r10))
    return (r1, r5, r10, idxs)

def evalrank_cam(model_path,data_path,id_list_i2t,id_list_t2i,split='test'):
    # load model and options
    map_location = {
                f"cuda:2": device_string,
                f"cuda:3": device_string,
                f"cuda:1": device_string,
                f"cuda:0": device_string
            }
    checkpoint = torch.load(model_path,map_location=map_location)
    opt = checkpoint['opt']
    opt.workers = 5
    
    if data_path is not None:
        opt.data_path = data_path

    bert_path = str(DATA_ROOT / 'bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    model = VSEModel(opt)

    model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()
    print('Loading dataset')
    data_loader = get_test_loader(data_path, opt.data_name,split, tokenizer, opt.batch_size, opt.workers, opt)

    with torch.no_grad():
        time1 = time.time()
        img_embs, cap_embs = encode_data_cam(model, data_loader)
        time2 = time.time()
        print('image, cap',img_embs.shape,cap_embs.shape)
        print('i2t, t2i', id_list_i2t.shape, id_list_t2i.shape)

    img_embs = img_embs[::5]
    
    start = time.time()
    (r1i, r5i, r10i, idxs_i2t) = afterhash_i2t(img_embs, cap_embs, id_list_i2t)
    i2t_end = time.time()
    (r1t, r5t, r10t, idxs_t2i) = afterhash_t2i(img_embs, cap_embs,  id_list_t2i)
    t2i_end = time.time()

    draw_mode = False
    if draw_mode:
        cp_img_out = img_embs
        cp_txt_out = cap_embs
        draw_img = cp_img_out.squeeze();
        draw_txt = cp_txt_out.squeeze();
        # np.save(NPY_ROOT / 'draw_img.npy', draw_img)
        # np.save(NPY_ROOT / 'draw_txt.npy', draw_txt)
        # np.save(NPY_ROOT / 'id_list_i2t.npy', id_list_i2t)
        # np.save(NPY_ROOT / 'id_list_t2i.npy', id_list_t2i)
        # np.save(NPY_ROOT / 'idxs_i2t.npy', idxs_i2t)
        np.save(NPY_ROOT / 'id_list_t2i.npy', id_list_t2i)
        print(id_list_t2i.shape)
        raise ValueError
        # draw_img = draw_img.cpu().numpy() if isinstance(draw_img, torch.Tensor) else draw_img
        # draw_txt = draw_txt.cpu().numpy() if isinstance(draw_txt, torch.Tensor) else draw_txt

        # tsne_img = TSNE(n_components=2, random_state=42).fit_transform(draw_img)
        # tsne_txt = TSNE(n_components=2, random_state=42).fit_transform(draw_txt)
        # colors = ['red' if i in idxs_i2t[0][:10] else 'gray' for i in range(len(tsne_txt))]
        # plt.scatter(tsne_txt[:, 0], tsne_txt[:, 1], c=colors, s=1)
        # plt.title('t-SNE Scatter Plot')
        # plt.xlabel('Component 1')
        # plt.ylabel('Component 2')
        # plt.savefig('tsne_plots_with_clusters.pdf')
        # plt.show()
        raise ValueError
        # scaler = MinMaxScaler()
        # tsne_img = scaler.fit_transform(tsne_img)
        # tsne_txt = scaler.fit_transform(tsne_txt)

        kmeans_img = KMeans(n_clusters=200, random_state=42).fit(tsne_img)
        kmeans_txt = KMeans(n_clusters=20, random_state=42).fit(tsne_txt)

        labels_img = kmeans_img.labels_
        unique_labels_img = np.unique(labels_img)
        num_unique_labels_img = len(unique_labels_img)
        print("Number of unique labels in labels_img:", num_unique_labels_img)
        zero_indices = np.where(labels_img == 0)[0]
        raise ValueError
        right_list = []
        for i in zero_indices:
            right_list.extend(id_list_i2t[i])
        right_list = np.array(list(set(right_list)))
        # print("right_list",right_list.shape)
        # raise ValueError
    
        labels_txt = kmeans_txt.labels_

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(wspace=0.05)

        scatter_img = axes[0].scatter(tsne_img[:, 0], tsne_img[:, 1], c=labels_img, s=4, cmap='tab20b', marker='*')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        colors_txt = np.array(labels_txt, dtype=float)
        for i in range(len(colors_txt)):
            if i not in right_list:
                colors_txt[i] = -1
        cmap = plt.cm.get_cmap('tab20b')
        cmap.set_under('gray')

        # scatter_txt = axes[1].scatter(tsne_txt[:, 0], tsne_txt[:, 1], c=labels_txt, s=1, cmap='tab20b')
        scatter_txt = axes[1].scatter(tsne_txt[:, 0], tsne_txt[:, 1], c=colors_txt, s=1, cmap=cmap, vmin=0)

        axes[1].set_xticks([])
        axes[1].set_yticks([])

        plt.savefig('tsne_plots_with_clusters.pdf', bbox_inches='tight')
        plt.close()

        raise ValueError
    
    print("evalution.py 489 calculate afterhash matching time i2t_need t2i_need:",i2t_end-start, t2i_end - i2t_end)
    rsum = round(round(r1i, 1) + round(r5i, 1) + round(r10i, 1) + round(r1t, 1) + round(r5t, 1) + round(r10t, 1),1)
    return rsum, r1i, r5i, r10i, r1t, r5t, r10t, i2t_end-start, t2i_end - i2t_end, time2-time1

def encode_data_cam(model, data_loader, log_step=10):
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    for i, data_i in enumerate(data_loader):

        # make sure val logger is used
        # images, image_lengths, captions, lengths, ids, img_ids, repeat = data_i
        images, image_lengths, captions, lengths, ids = data_i
        model.logger = val_logger
        if i == 0:
            print("data_i",images.shape, captions.shape)
            # print("captions", captions[0])

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(images, captions, lengths, image_lengths=image_lengths)

        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging.info('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Batch-Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs

def evalrank_test(model_path,data_path,cam_model_path,split='test', fold5=False):
    """
        model path :hash  model path
    """
    map_location = {
                f"cuda:2": device_string,
                f"cuda:3": device_string,
                f"cuda:1": device_string,
                f"cuda:0": device_string
            }
    checkpoint = torch.load(model_path,map_location=map_location)
    hq_opt = checkpoint['opt']
    C_img = checkpoint["C_img"]
    C_txt = checkpoint["C_txt"]

    if data_path is not None:
        hq_opt.data_path = data_path
    # load vocabulary used by the model
    hq_opt.vocab_path = str(DATA_ROOT / "vocab")
    vocab = deserialize_vocab(os.path.join(hq_opt.vocab_path, '%s_vocab.json' % hq_opt.data_name))
    hq_opt.vocab_size = len(vocab)
    hq_opt.H = 64
    hq_model = HQ(hq_opt)

    hq_model.load_state_dict(checkpoint['model'])
    data_loader,data_len = get_test_loader_data(split, hq_opt.data_name, vocab,hq_opt.batch_size, hq_opt.workers, hq_opt)
    print('Gnerating hash and quantization vectors...',data_len)
    _, _, cp_img_out, cp_txt_out = encode_data(hq_model, data_loader, hq_opt.log_step, logging.info)
    
    cp_img_out = np.array([cp_img_out[i] for i in range(0, len(cp_img_out), 5)])

    cp_img_out = torch.from_numpy(cp_img_out).to(device)
    cp_txt_out = torch.from_numpy(cp_txt_out).to(device)

    val_len = cp_txt_out.shape[0]

    realK = 2 ** hq_opt.K
    img_val_q = torch.zeros(val_len // 5, hq_opt.M * realK).to(device)
    txt_val_q = torch.zeros(val_len, hq_opt.M * realK).to(device)

    for i in range(1):
        img_val_q = update_codes_ICM(cp_img_out, img_val_q, C_img, hq_opt.max_iter_update_b, cp_img_out.shape[0], hq_opt.M,
                                     realK)
        txt_val_q = update_codes_ICM(cp_txt_out, txt_val_q, C_txt, hq_opt.max_iter_update_b, cp_txt_out.shape[0], hq_opt.M,
                                     realK)

    hash_start = time.time()
    hash_idx_i2t = HammingD(cp_img_out, cp_txt_out, R_rate=0.3)
    # hash_idx_i2t = HammingD(cp_img_out, cp_txt_out, R_rate=1)
    hash_i2t_end = time.time()
    hash_idx_t2i = HammingD(cp_txt_out, cp_img_out, R_rate=0.6)
    # hash_idx_t2i = HammingD(cp_txt_out, cp_img_out, R_rate=1)
    print("*"*100)
    print(hash_idx_t2i.shape)
    print(hash_idx_i2t.shape)
    # raise ValueError()
    hash_end = time.time()
    print("calculate hashing select time:", hash_i2t_end - hash_start, hash_end - hash_i2t_end)
    print("hash_idx_i2t.shape", hash_idx_i2t.shape, hash_idx_t2i.shape)

    q_start = time.time()
    q_idx_i2t = AQD_i2t(C_txt, cp_img_out, txt_val_q, hash_idx_i2t, q_rate=0.5)
    q_i2t_end = time.time()
    q_idx_t2i = AQD_t2i(C_img, cp_txt_out, img_val_q, hash_idx_t2i, q_rate=0.6)
    q_end = time.time()

    print("calculate Quantization time:", q_i2t_end - q_start, q_end - q_i2t_end)

    rsum, r1i, r5i, r10i, r1t, r5t, r10t, t1, t2, t3=evalrank_cam(cam_model_path,data_path, q_idx_i2t, q_idx_t2i,split)
    print("t2i_rSum %.1f"%(round(r1i,1) + round(r5i,1) + round(r10i,1)))
    print("i2t_rSum %.1f"%(round(r1t,1) + round(r5t,1) + round(r10t,1)))
    print("t = ", t1+t2+hash_end-hash_start+q_end-q_start + t3*0.3)
    print('t1' , t1)
    print('t2', t2)
    print('t3', t3)
    rsum = round(r1i,1) + round(r5i,1) + round(r10i,1) + round(r1t,1) + round(r5t,1) + round(r10t,1)
    print("rsum=%.1f" % rsum)
    return rsum, r1i, r5i, r10i, r1t, r5t, r10t
    # evalrank_clip(q_idx_i2t, q_idx_t2i)


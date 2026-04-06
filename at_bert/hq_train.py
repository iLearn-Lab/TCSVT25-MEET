# encoding:utf-8
"""Training script"""

import os
import time
import shutil

import torch
import numpy

# from data import *
from lib.vocab import Vocabulary, deserialize_vocab
#
from lib.model import HQ
from lib.evalution_AHR import AQD_t2i, AQD_i2t, AverageMeter, LogCollector, encode_data, HammingD,evalrank_cam, encode_data_train
from lib.data import get_loaders, get_test_loader_data

import logging
from lib.utils import *
import argparse
from lib.where_cuda import device
import os.path as osp
from pathlib import Path

AT_BERT_ROOT = Path(__file__).resolve().parent
LIRONG_ROOT = AT_BERT_ROOT.parent
DATA_ROOT = LIRONG_ROOT / "data"
ESA_ROOT = LIRONG_ROOT / "ESA"
MODEL_ROOT = AT_BERT_ROOT / "modelzoos"
CAM_MODEL_PATHS = {
    "f30k_precomp": ESA_ROOT / "f30k_butd_region_bert1" / "model_best.pth",
    "coco_precomp": ESA_ROOT / "coco_butd_region_bert1" / "model_best.pth",
}

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default=str(DATA_ROOT),
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default=str(DATA_ROOT / 'vocab'),
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=12, type=int,
                        help='Number of training epochs.')

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--learning_hash_rate', default=.02, type=float,
                        help='Initial learning_hash rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=1500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--lambda_softmax', default=20., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--focal_type', default="equal",
                        help='equal|prob')

    parser.add_argument('--H', default=64, type=int, help='hash code length')
    parser.add_argument('--K', default=8, type=int, help='each dictionary size')
    parser.add_argument('--M', default=8, type=int, help='number of dictionies')
    parser.add_argument('--max_iter_update_Cb', default=1, type=int)
    parser.add_argument('--max_iter_update_b', default=1, type=int)
    parser.add_argument('--q_lambda', default=0.0001, type=float, help='hyperparameter in the loss function')
    parser.add_argument('--nhead', default=16, type=int, help='the num_head for Transformer encoder')
    parser.add_argument('--dropout', default=0.1, type=float, help='the dropout rate for Transformer encoder')
    parser.add_argument('--residual_weight', default=0.8, type=float, help='the weight of residual operation for pooling')

    opt = parser.parse_args()

    opt.data_path = str(DATA_ROOT)
    opt.data_name = "f30k_precomp"
    opt.vocab_path = str(DATA_ROOT / "vocab")
    # opt.vocab_path = str(DATA_ROOT / "vocab")
    # opt.logger_name = str(AT_BERT_ROOT / "runlogs" / "testmodel_logs")

    # opt.H = 64
    # opt.M = 8
    # opt.K = 8
    hpara = "H" + str(opt.H) + "_M" + str(opt.M) + "_K" + str(opt.K)
    opt.model_name = str(MODEL_ROOT / "try1" / opt.data_name.split('_')[0] / hpara)
    opt.max_violation = True

    K = 2 ** opt.K

    M = opt.M
    # q_dim = opt.embed_size
    q_dim = opt.H

    ####### initiate parameters used in the training ######

    C_img = torch.FloatTensor(M * K, q_dim).uniform_(-1, 1).to(device)
    C_txt = torch.FloatTensor(M * K, q_dim).uniform_(-1, 1).to(device)

    # vocab = deserialize_vocab(os.path.join(
    #     opt.vocab_path, '%s_vocab.json' % opt.data_name))
    # opt.vocab_size = len(vocab)
    # Load data loaders
    train_loader, val_loader,train_len,val_len = get_loaders(opt.data_name, None, opt.batch_size, opt.workers, opt)
    test_loader,test_len = get_test_loader_data("test", opt.data_name, None,
                                  opt.batch_size, opt.workers, opt)

    code_img = torch.zeros(train_len, M * K).to(device)
    code_txt = torch.zeros(train_len, M * K).to(device)

    # dense features
    img_output = torch.zeros(train_len, q_dim).to(device)
    txt_output = torch.zeros(train_len, q_dim).to(device)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # Construct the model
    model = HQ(opt)

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        print("the programming is runing in epoch :",epoch)
        adjust_learning_rate(opt, model.cam_optimizer, epoch)
        adjust_learning_rate(opt, model.hash_optimizer, epoch)
        train(opt, train_loader, model, epoch, val_loader)
        img_output,txt_output = encode_data_train(model, train_loader,img_output ,txt_output)

        C_img,C_txt = initial_centers(img_output, txt_output, opt.M, 2**opt.K, img_output.shape[-1])

        code_img = update_codes_ICM(img_output, code_img, C_img, opt.max_iter_update_b, img_output.shape[0], opt.M,2**opt.K)
        code_txt = update_codes_ICM(txt_output, code_txt, C_txt, opt.max_iter_update_b, txt_output.shape[0], opt.M,2**opt.K)

        # evaluate on validation set

        rsum, r1i, r5i, r10i, r1t, r5t, r10t, hashtime_i2t, hashtime_t2i, quanttime_i2t, quanttime_t2i, afterhashtime_i2t, afterhashtime_t2i = \
            validate(opt, test_loader, model, C_img,C_txt)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.makedirs(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
            'C_img':C_img,
            'C_txt':C_txt,
            # 'hashtime_i2t':hashtime_i2t,
            # 'hashtime_t2i':hashtime_t2i,
            # 'quanttime_i2t':quanttime_i2t,
            # 'quanttime_t2i':quanttime_t2i,
            # 'afterhashtime_i2t':afterhashtime_i2t,
            # 'afterhashtime_t2i':afterhashtime_t2i,
            'r1i': r1i,
            'r5i': r5i,
            'r10i': r10i,
            'r1t': r1t,
            'r5t': r5t,
            'r10t': r10t,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')

    for file in os.listdir(opt.model_name):
        if file != "model_best.pth.tar":
            os.remove(opt.model_name + "/" + file)

def train(opt, train_loader, model, epoch,val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()

    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

def validate(opt, val_loader, model,C_img,C_txt):
    # compute the encoding for all the validation images and captions
    _, _, cp_img_out, cp_txt_out = encode_data(model, val_loader, opt.log_step, logging.info)
    cp_img_out = numpy.array([cp_img_out[i] for i in range(0, len(cp_img_out), 5)])
    cp_img_out = torch.from_numpy(cp_img_out).to(device)
    cp_txt_out = torch.from_numpy(cp_txt_out).to(device)

    val_len = cp_txt_out.shape[0]

    realK = 2**opt.K
    img_val_q = torch.zeros(val_len//5,opt.M*realK).to(device)
    txt_val_q = torch.zeros(val_len,  opt.M * realK).to(device)

    for i in range(1):
        img_val_q = update_codes_ICM(cp_img_out,img_val_q, C_img,opt.max_iter_update_b, cp_img_out.shape[0], opt.M, realK)
        txt_val_q = update_codes_ICM(cp_txt_out, txt_val_q, C_txt, opt.max_iter_update_b, cp_txt_out.shape[0], opt.M, realK)

    hash_start = time.time()
    hash_idx_i2t = HammingD(cp_img_out, cp_txt_out, R_rate=0.3)
    hash_i2t_end = time.time()
    hash_idx_t2i = HammingD(cp_txt_out, cp_img_out, R_rate=0.6)
    hash_end = time.time()
    print("calculate hashing select time:",hash_i2t_end-hash_start, hash_end - hash_i2t_end)

    q_start = time.time()
    q_idx_i2t = AQD_i2t(C_txt, cp_img_out, txt_val_q,hash_idx_i2t, q_rate = 0.5)
    q_i2t_end = time.time()
    q_idx_t2i = AQD_t2i(C_img, cp_txt_out, img_val_q, hash_idx_t2i,q_rate = 0.6)
    q_end = time.time()
    print("calculate Quantization time:", q_i2t_end - q_start,q_end-q_i2t_end)

    MODEL_PATH = str(CAM_MODEL_PATHS.get(opt.data_name, CAM_MODEL_PATHS["f30k_precomp"]))
    DATA_PATH = str(DATA_ROOT)
    if opt.data_name == "f30k_precomp":
        currscore, r1i, r5i, r10i, r1t, r5t, r10t, afterhashtime_i2t, afterhashtime_t2i,_  = evalrank_cam(MODEL_PATH, DATA_PATH,q_idx_i2t, q_idx_t2i, split="test")
    else:
        print('coco')
        currscore, r1i, r5i, r10i, r1t, r5t, r10t, afterhashtime_i2t, afterhashtime_t2i,_ = evalrank_cam(MODEL_PATH,
                                                                                                       DATA_PATH,
                                                                                                       q_idx_i2t,
                                                                                                       q_idx_t2i,
                                                                                                       split="testall")
    hashtime_i2t = hash_i2t_end - hash_start
    hashtime_t2i = hash_end - hash_i2t_end
    quanttime_i2t = q_i2t_end - q_start
    quanttime_t2i = q_end - q_i2t_end
    # currscore = evalrank_cam(val_loader, q_idx_i2t, q_idx_t2i)
    return currscore, r1i, r5i, r10i, r1t, r5t, r10t, hashtime_i2t, hashtime_t2i, quanttime_i2t, quanttime_t2i, afterhashtime_i2t, afterhashtime_t2i

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix +
                                'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    import sys

    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)

    root_dir = osp.abspath(osp.dirname(__file__))
    lib_path = osp.join(root_dir, 'lib')
    add_path(lib_path)

    import _init_paths
    main()

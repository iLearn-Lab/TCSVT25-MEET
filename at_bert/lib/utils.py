# encoding:utf-8
import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as  nn
import cupy as cp
from torch.utils.dlpack import to_dlpack,from_dlpack
from cupy import fromDlpack
import torch.nn.functional as F
from torch.autograd import Variable
'''
quantization functions 
X = sum_i^MC_ib_i  
X is continuous feature(input) 
C is center(output) 
b is binary code(output) 
'''

from .where_cuda import device ,device_id
def initial_centers(img_input, txt_input, M, K, q_dim):
    '''
    initial_centers initial the quantization centers

    args:
        img_input: image continuous feature
        txt_input: text continuous feature
        M: number of dictionaries
        K: number of centers in each dictionary
        q_dim: continuous feature dimension

    '''
    C_init_img = np.zeros([M * K, q_dim])
    C_init_txt = np.zeros([M * K, q_dim])
    print("initilizing Centers")

    img_out = img_input.cpu().detach().numpy()
    # img_out = np.array([img_out[i] for i in range(0, len(img_out), 5)])
    txt_out = txt_input.cpu().detach().numpy()

    img_out = img_out.reshape(img_out.shape[0],-1)
    txt_out = txt_out.reshape(txt_out.shape[0],-1)
    # all_input = np.vstack([img_input.cpu().detach().numpy(), txt_input.cpu().detach().numpy()])
    # all_input = np.vstack([img_out, txt_out])

    for i in range(M):
        kmeans = MiniBatchKMeans(n_clusters=K).fit(img_out[:, int(i * q_dim / M): int((i + 1) * q_dim / M)])
        C_init_img[i * K: (i + 1) * K, int(i * q_dim / M): int((i + 1) * q_dim / M)] = kmeans.cluster_centers_
        # print("img codebook: ", i, " finish")
    C_init_img = C_init_img.astype(np.float32)
    C_init_img = torch.from_numpy(C_init_img).to(device)

    for i in range(M):
        kmeans = MiniBatchKMeans(n_clusters=K).fit(txt_out[:, int(i * q_dim / M): int((i + 1) * q_dim / M)])
        C_init_txt[i * K: (i + 1) * K, int(i * q_dim / M): int((i + 1) * q_dim / M)] = kmeans.cluster_centers_
        # print("txt codebook: ", i, " finish")
    C_init_txt = C_init_txt.astype(np.float32)
    C_init_txt = torch.from_numpy(C_init_txt).to(device)
    return C_init_img,C_init_txt

def update_codes_ICM(output, code, C, max_iter_update_b, N, M, K):

    device_cupy = cp.cuda.Device(device_id)
    device_cupy.use()

    output = output.detach()
    output = fromDlpack(to_dlpack(output.view(output.shape[0], -1)))
    C = fromDlpack(to_dlpack(C))
    code = fromDlpack(to_dlpack(code))

    dim_step = output.shape[-1]//M

    for iterate in range(max_iter_update_b):

        sub_list = [i for i in range(M)]
        random.shuffle(sub_list)
        cnt = 0
        for m in sub_list:
            cnt = cnt+1
            # print("utils.py line126 the num of fine subspaces is ",cnt)

            v = cp.zeros((N, K))
            cntk = 0
            for indicator in range(K):
                cntk = cntk+1
                a = cp.zeros(K)
                a[indicator] = 1
                code[:, m * K: (m + 1) * K] = a

                foo = cp.sum(cp.square(output[:, m * dim_step:(m + 1) * dim_step] - cp.matmul(code[:, m * K:(m + 1) * K],C[m * K:(m + 1) * K, m * dim_step:(m + 1) * dim_step])),axis=1)
                v[:, indicator] = foo

            code[:, m * K: (m + 1) * K] = cp.eye(K)[cp.argmin(v, axis=1).reshape(-1)]

    assert cp.sum(cp.sum(code, 1) == M), "update_code wrong"
    code = from_dlpack(code.toDlpack())
    return code


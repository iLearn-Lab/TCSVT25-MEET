import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as  nn
from utils import *
from where_cuda import device
'''
loss functions used in the project
'''

class quantization_loss(nn.Module):
    """
       Comput quantization loss
    """

    def __init__(self):
        super(quantization_loss, self).__init__()

    def forward(self, C_img, C_txt, output_img, code_img, output_txt, code_txt):
        '''
            quantization_loss quantization loss in paper equation(6)
            args:
                C: uantization centers in the dictionary
                output_img: image continuous feature
                code_img: image binary code
                output_txt: text continuous feature
                code_txt: text binary code
            '''
        output_img = output_img.view(output_img.shape[0],-1)
        output_txt = output_txt.view(output_txt.shape[0],-1)

        img_loss = torch.sum(torch.mul(output_img - torch.matmul(code_img, C_img), output_img - torch.matmul(code_img, C_img)))
        txt_loss = torch.sum(torch.mul(output_txt - torch.matmul(code_txt, C_txt), output_txt - torch.matmul(code_txt, C_txt)))
        q_loss = img_loss + txt_loss
        # print("loss.py line36 img_loss and txt_loss are ",img_loss.item(),txt_loss.item())
        return q_loss


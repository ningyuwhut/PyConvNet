# coding=utf-8

import numpy as np

#x_shape 是 输入矩阵的维度，HF是卷积核的高，WF是卷积核的宽
def im2col_index(x_shape, HF, WF, pad, stride):
    # get input size
    H, W, D, N = x_shape #N是图像个数
    # get output size
    out_h = 0
    out_w = 0
    if type(pad) is int:
        out_h = (H + 2 * pad - HF) / stride + 1 #卷积后的输出矩阵的高
        out_w = (W + 2 * pad - WF) / stride + 1 #卷积后输出矩阵的宽
    else:
        out_h = (H + pad[0] + pad[1] - HF) / stride + 1
        out_w = (W + pad[2] + pad[3] - WF) / stride + 1
    # for row index, compute the first index of the first HF * WF block
    r0 = np.repeat(np.arange(HF), WF) #将np.arange(HF)中的元素重复WF次，即一个卷积核展开成向量的长度
    r0 = np.tile(r0, D)#复制拼接D个r0,D是卷积核的通道数
    #假设HF=WF=2,D=1，那么r0=（0，0，1，1）
    #r0的长度为HF*WF*D
    # then compute the bias of each block
    #假设out_h=out_w=3,那么r_bias=(0,0,0,1,1,1,2,2,2)
    r_bias = stride * np.repeat(np.arange(out_h), out_w)#为什么乘以stide？r_bias是卷积后输出矩阵展开成向量后的偏置
    # then the row index is the r0 + r_bias
    r = r0.reshape(-1, 1) + r_bias.reshape(1, -1) #扩展成 
    #r记录的是将x展开之后每个元素对应的在x中的行的下标

    #r的列是每一块需要进行卷积的单元
    # the same to the col index
    c0 = np.tile(np.arange(WF), HF * D)#假设如上，c0为(0,1,0,1)
    c_bias = stride * np.tile(np.arange(out_w), out_h)#c_bias为（0，1，2，0，1，2，0，1，2）
    c = c0.reshape(-1, 1) + c_bias.reshape(1, -1)
    #c记录的是x展开之后每个元素对应的x中的列的下标

    # then the dimension index
    d = np.repeat(np.arange(D), HF * WF).reshape(-1, 1)

    return (r, c, d)
#转换成每列一个卷积核的形式，列的个数是Hout*Wout*N，一个图片对应Hout*Wout列
#x的维度是（Height,Weight,Depth, Number of Samples(Pic))
#转换后变成(HF*WF*DF,N*Hout*Wout)
def im2col(x, HF, WF, pad, stride):
    # padding
    x_padded = None
    if type(pad) is int:
        x_padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='constant')
    else:
        x_padded = np.pad(x, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0), (0, 0)), mode='constant')
    r, c, d = im2col_index(x.shape, HF, WF, pad, stride)
    cols = x_padded[r, c, d, :]
    cols = cols.reshape(HF * WF * x.shape[2], -1) #reshape 成(HF*WF*DF,N*Hout*Wout)
    return cols

#x_shape是当前层的输入矩阵
#cols是当前层的残差矩阵,维度和im2col的输出矩阵的维度相同
#纬度是（HF*WF*DF, N*Hout*Wout)
def col2im(cols, x_shape, HF, WF, pad, stride):
    # get input size
    H, W, D, N = x_shape
    H_padded = 0
    W_padded = 0
    if type(pad) is int:
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
    else:
        H_padded, W_padded = H + pad[0] + pad[1], W + pad[2] + pad[3]
    x_padded = np.zeros((H_padded, W_padded, D, N), dtype=cols.dtype)#初始化一个全零的矩阵
    r, c, d = im2col_index(x_shape, HF, WF, pad, stride)
    cols_reshaped = cols.reshape((HF * WF * D, -1, N))  
    np.add.at(x_padded, (r, c, d, slice(None)), cols_reshaped)
    #参考:https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html
    if pad == 0:
        return x_padded
    elif type(pad) is int:
        return x_padded[pad:-pad, pad:-pad, :, :]
    else:
        return x_padded[pad[0]:-pad[1], pad[2]:-pad[3], :, :]

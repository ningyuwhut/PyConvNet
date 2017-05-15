# coding=utf-8

#import pylayer

import numpy as np
from im2col import im2col
from im2col import col2im
#x的维度是（Height,Weight,Depth, Number of Samples(Pic))
#输出纬度是(HO,WO,NF,N)
def conv_forward(x, w, b, params):
    # get convolution parameters
    stride = params['stride']
    pad = params['pad']
    # get input size
    H, W, D, N = x.shape
    HF, WF, DF, NF = w.shape
    _, _, DB, NB = b.shape
    # check input size
    assert D == DF, 'dimension does not work'
    assert NF == NB, 'batch size does not work'
    # check params
    assert (H + 2 * pad - HF) % stride == 0, 'pad and stride do not work'
    assert (W + 2 * pad - WF) % stride == 0, 'pad and stride do not work'
    # get output size
    HO = (H + 2 * pad - HF) / stride + 1
    WO = (W + 2 * pad - WF) / stride + 1
    x_col = im2col(x, HF, WF, pad, stride)#转换成一列是一个感受野的形式
    w_col = w.transpose(3, 0, 1, 2).reshape((NF, -1)) #先转置成(NF,HF, WF, DF),再转置成（NF，HF*WF*DF），这样就变成一行是一个卷积核，
    output_col = w_col.dot(x_col) + b.reshape(-1, 1)#卷积操作，结果便是一行是一个卷积核的结果
    #纬度是(NF,N*Hout*Wout)
    output_col = output_col.reshape((NF, HO, WO, N))
    output_col = output_col.transpose(1, 2, 0, 3)#变成(HO,WO,NF,N)
    return output_col

#dout是下一层的残差
#x的维度是（Height,Weight,Depth, Number of Samples(Pic))
#dout的维度和池化层的输入的维度相同，纬度是(Hout, Wout, D, N)

def conv_backward(x, w, b, conv_param, dout):   
    print " in conv_backward"
    HF, WF, DF, NF = w.shape
    print "dout.shape", dout.shape
    x_col = im2col(x, HF, WF, conv_param['pad'], conv_param['stride'])
    #转换后变成(HF*WF*DF,N*Hout*Wout)

    print "x_col.shape", x_col.shape
    w_col = w.transpose(3, 0, 1, 2).reshape((NF, -1))#每一行是一个卷积核
    #w_col的维度是(NF,HF,WF,DF),reshape成(NF,HF*WF*DF)

    db = np.sum(dout, axis=(0, 1, 3))
    dout = dout.transpose(2, 0, 1, 3)#(NF,Hout,Wout, N)
    dout = dout.reshape((w_col.shape[0], x_col.shape[-1]))#(NF,N*Hout*Wout) 
    dx_col = w_col.T.dot(dout)#当前层的残差 , （HF*WF*DF, N*Hout*Wout),和x_col的维度相同
    dw_col = dout.dot(x_col.T) #当前层关于卷积核的梯度

    dx = col2im(dx_col, x.shape, HF, WF, conv_param['pad'], conv_param['stride'])
    dw = dw_col.reshape((dw_col.shape[0], HF, WF, DF))
    dw = dw.transpose(1, 2, 3, 0)

    return [dx, dw, db]
##x的维度是（Height,Weight, NF,Number of Samples(Pic))
def max_pooling_forward(x, pool_params):
    # get max-pooling parameters
    stride = pool_params['stride']
    HF = pool_params['HF']
    WF = pool_params['WF']
    pad = pool_params['pad']
    # get input size
    H, W, D, N = x.shape
    x_reshaped = x.reshape(H, W, 1, -1)#(H,W,1,D*N)
    # get output size
    HO = 0
    WO = 0
    if type(pad) is int:
        HO = (H + 2 * pad - HF) / stride + 1
        WO = (W + 2 * pad - WF) / stride + 1
    else:
        HO = (H + pad[0] + pad[1] - HF) / stride + 1
        WO = (W + pad[2] + pad[3] - WF) / stride + 1
    x_col = im2col(x_reshaped, HF, WF, pad, stride)
    #x_col的维度是(HF*WF,N*Hout*Wout)

    x_col_argmax = np.argmax(x_col, axis=0)#求每个feature_map的最大值所在的下标
    #x_col_argmax的维度是(1,N*Hout*Wout)，每列是一个卷积核的最大值的下标

    x_col_max = x_col[x_col_argmax, np.arange(x_col.shape[1])]
    out = x_col_max.reshape((HO, WO, D, N))
    return out

def max_pooling_backward(x, dout, pool_params):
    print "in max_pooling_backward"
    print "dout.shape", dout.shape
    print "x.shape", x.shape
    H, W, D, N = x.shape
    x_reshaped = x.reshape(H, W, 1, -1)
    x_col = im2col(x_reshaped, pool_params['HF'],
                   pool_params['WF'], pool_params['pad'], pool_params['stride'])
    x_col_argmax = np.argmax(x_col, axis=0)
    dx_col = np.zeros_like(x_col) #和x_col同样纬度的0矩阵
    print " 1 dx_col.shape", dx_col.shape

    dx_col[x_col_argmax, np.arange(x_col.shape[1])] = dout.ravel() #把dout平铺
    print " 2 dx_col.shape", dx_col.shape

    dx_shaped = col2im(dx_col, x_reshaped.shape, pool_params['HF'], pool_params['WF'],
                       pool_params['pad'], stride=pool_params['stride'])
    dx = dx_shaped.reshape(x.shape)
    return [dx]

def relu_forward(x):
    out = np.where(x > 0, x, 0)
    return out

def relu_backward(x, dout):
    dx = np.where(x > 0, dout, 0)
    return [dx]

def softmax_loss_forward(x, y):
    # x is the prediction(C * N), y is the label(1 * N)
    x_reshaped = x.reshape((x.shape[2], x.shape[3]))
    probs = np.exp(x_reshaped - np.max(x_reshaped, axis=0, keepdims=True))
    probs /= np.sum(probs, axis=0, keepdims=True)
    N = x_reshaped.shape[1]
    loss = -np.sum(np.log(probs[y, np.arange(N)])) / N
    return loss

def softmax_loss_backward(x, y):
    x_reshaped = x.reshape((x.shape[2], x.shape[3]))
    probs = np.exp(x_reshaped - np.max(x_reshaped, axis=0, keepdims=True))
    probs /= np.sum(probs, axis=0, keepdims=True)
    dx = probs.copy()
    N = x_reshaped.shape[1]
    dx[y, np.arange(N)] -= 1
    dx /= N
    dx = dx.reshape((1, 1, dx.shape[0], dx.shape[1]))
    return [dx]

def softmax(x):
    x_reshaped = x.reshape((x.shape[2], x.shape[3]))
    probs = np.exp(x_reshaped - np.max(x_reshaped, axis=0, keepdims=True))
    probs /= np.sum(probs, axis=0, keepdims=True)
    return np.argmax(probs, axis=0)

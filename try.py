import sys
import numpy as np

sys.path.append("..")
from papr import calcPAPR
from plot import plotBER, plotCCDF, plotImages
from utils import dec2bin
from modulation import qamdemod

from CMnet import PAPRnetEncoder, PAPRnetDecoder
# import statsmodels.api as sm
import cmath
import math
import scipy.io as scio
from channel import SISOFlatChannel, awgn
from utils import polar2rect, img2bits, bits2img

N = 256

# 加载测试数据
x = np.load('data/modChunks.npz')['data']
xval = x[0:1000]								#取前1000数据组

# 保存数据
data1 = 'data_after_4QAM.mat'
scio.savemat(data1, {'data_after_4QAM':xval})

# 虚部实部分离
xvalrect = np.concatenate( [np.expand_dims(xval.real, 1), np.expand_dims(xval.imag, 1)], axis=1)  

# Encoding
encoder = PAPRnetEncoder(N)
encoder.load_weights('encoder.hdf5')

xenc = encoder.predict(xvalrect, batch_size=256)   # 获得编码输出
xhidd = xenc[:, 0, :] + 1j * xenc[:, 1, :]   

# 保存原始OFDM+paprnet 编码后的的数据
data3 = 'data_paprnet_before_ifft.mat'
scio.savemat(data3, {'data_paprnet_before_ifft':xhidd})



dataFile = 'p_after_fft.mat'
data = scio.loadmat(dataFile)
x = data['p_after_fft']

xenc = np.concatenate( [np.expand_dims(x.real, 1), np.expand_dims(x.imag, 1)], axis=1)

# Decoding
decoder = PAPRnetDecoder(N)
decoder.load_weights('decoder.hdf5')

xdec = decoder.predict(xenc, batch_size=256)
xest = xdec[:, 0, :] + 1j * xdec[:, 1, :]

# 保存原始OFDM+paprnet 解码之后的数据
data4 = 'data_paprnet_after_fft.mat'
scio.savemat(data4, {'data_paprnet_after_fft':xest})













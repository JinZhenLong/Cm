import numpy as np
from CMnet import CMnetEncoder, CMnetDecoder
import cmath
import scipy.io as scio
from time import time
N = 256

x = np.load('data/modChunks.npz')['data']
xval = x[0:1000]

xsig = np.fft.ifft(xval, n=256, axis=1)*cmath.sqrt(N)
encoder = CMnetEncoder(N)
encoder.load_weights('train_model/for_plot/encoder5.hdf5')

xvalrect = np.concatenate( [np.expand_dims(xval.real, 1), np.expand_dims(xval.imag, 1)], axis=1)


xenc = encoder.predict(xvalrect, batch_size=256)

xhidd = xenc[:, 0, :] + 1j * xenc[:, 1, :]


data3 = 'data_paprnet_before_ifft_5.mat'
scio.savemat(data3, {'data_paprnet_before_ifft_5':xhidd})



decoder = CMnetDecoder(N)
decoder.load_weights('train_model/for_plot/decoder5.hdf5')


xdec = decoder.predict(xenc, batch_size=256)

xest = xdec[:, 0, :] + 1j * xdec[:, 1, :]




data4 = 'data_paprnet_after_fft_5.mat'
scio.savemat(data4, {'data_paprnet_after_fft_5':xest})

signal = xsig
data5 = 'data_original_after_fft_5.mat'
scio.savemat(data5, {'data_original_after_fft_5':signal})

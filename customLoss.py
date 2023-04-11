import tensorflow as tf
import keras.backend as K
from keras.backend import mean, var, max, abs, square, sqrt

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def CmLoss(y_ture,y_pred):
    # yPower = K.sqrt(K.sum(K.square(y_ture - y_pred), axis=1))    # pre: yPower = K.sqrt(K.sum(K.square(y_pred), axis=1))
    # yMax = K.max(yPower, axis=-1)
    # yMean = K.mean(yPower, axis=-1)
    # yPAPR = yMax/yMean  # pre: yPAPR = 10 * log10(yMax/yMean)

    yPower = K.sqrt(K.sum(K.square(y_pred), axis=1))
    ymean_1 = K.mean(yPower, axis=1)
    ymean_2 = K.mean(yPower, axis=1)
    ymean_3 = (ymean_2)*(ymean_2) ** 2
    yRcm = K.sqrt(ymean_1 / ymean_3)
    ycm = (yRcm - 1.52) / 1.56

    # return yPAPR
    return ycm

def dataLoss(y_ture ,y_pred):
    yPower = K.sqrt(K.sum(K.square(y_ture - y_pred), axis=1))    # pre: yPower = K.sqrt(K.sum(K.square(y_pred), axis=1))
   # ymean_1 = K.mean(yPower, axis=1)
   # ymean_2 = K.mean(yPower, axis=1)
   # ymean_3 = (ymean_2) ^ 3
   # yRcm = K.sqrt(ymean_1 / ymean_3)
    yMax = K.max(yPower, axis=-1)



    return yMax
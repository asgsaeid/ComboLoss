from keras import backend as K

ce_w = 0.5
ce_d_w = 0.5
e = K.epsilon()
smooth = 1
'''
ce_w values smaller than 0.5 penalize false positives more while values larger than 0.5 penalize false negatives more
ce_d_w is level of contribution of the cross-entropy loss in the total loss.
'''

def Combo_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    d = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    y_pred_f = K.clip(y_pred_f, e, 1.0 - e)
    out = - (ce_w * ((y_true_f * K.log(y_pred_f)) + ((1 - ce_w) * (1.0 - y_true_f) * K.log(1.0 - y_pred_f))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_d_w * weighted_ce) - ((1 - ce_d_w) * d)
    return combo



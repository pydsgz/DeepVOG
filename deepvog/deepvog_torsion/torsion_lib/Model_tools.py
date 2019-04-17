import keras.backend as K
from keras.models import load_model

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0000001) / (K.sum(y_true_f) + K.sum(y_pred_f) +  0.0000001)

def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice

def load_custom_model(model_path):

    model = load_model(model_path, custom_objects = {"dice_coef_multilabel":dice_coef_multilabel})
    return model
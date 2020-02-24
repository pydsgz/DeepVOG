import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform

from tensorflow.python.keras import layers
from keras import backend as K
from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.layers import Conv2DTranspose, Concatenate
from tensorflow.python.keras.models import Model, load_model
#from kito import reduce_keras_model

def encoding_block(X, filter_size, filters_num, layer_num, block_type, stage, s = 1, X_skip=0):
    
    # defining name basis
    conv_name_base = 'conv_' + block_type + str(stage) + '_'
    bn_name_base = 'bn_' + block_type + str(stage)  + '_'
    
    
    for i in np.arange(layer_num)+1:
        # First component of main path 
        X = Conv2D(filters_num, filter_size , strides = (s,s), padding = 'same', name = conv_name_base + 'main_' + str(i), kernel_initializer = glorot_uniform())(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + 'main_' + str(i))(X)
        if i != layer_num:
            X = Activation('relu')(X)


    X = Activation('relu')(X)
    
    # Down sampling layer
    X_downed = Conv2D(filters_num*2, (2, 2), strides = (2,2), padding = 'valid', name = conv_name_base + 'down', kernel_initializer = glorot_uniform())(X)
    X_downed = BatchNormalization(axis = 3, name = bn_name_base + 'down')(X_downed)
    X_downed = Activation('relu')(X_downed)
    return X, X_downed

def decoding_block(X, filter_size, filters_num, layer_num, block_type, stage, s = 1, init_X_joint = 0, X_jump = 0, up_sampling = True):
    
    # defining name basis
    conv_name_base = 'conv_' + block_type + str(stage) + '_'
    bn_name_base = 'bn_' + block_type + str(stage)  + '_'
    

    # Joining X_jump from encoding side with X_uped
    if init_X_joint == 1:
        X_joined_input = X
    else:
    # X_joined_input = Add()([X,X_jump])
        X_joined_input = Concatenate(axis = 3)([X,X_jump])
    
    ##### MAIN PATH #####
    for i in np.arange(layer_num)+1:
        # First component of main path 
        X_joined_input = Conv2D(filters_num, filter_size , strides = (s,s), padding = 'same',
                                name = conv_name_base + 'main_' + str(i), kernel_initializer = glorot_uniform())(X_joined_input)
        X_joined_input = BatchNormalization(axis = 3, name = bn_name_base + 'main_' + str(i))(X_joined_input)
        if i != layer_num:
            X_joined_input = Activation('relu')(X_joined_input)

    X_joined_input = Activation('relu')(X_joined_input)
    
    # Up-sampling layer. At the output layer, up-sampling is disabled and replaced by other stuffs manually
    if up_sampling == True:
        X_uped = Conv2DTranspose(filters_num, (2, 2), strides = (2,2), padding = 'valid',
                                 name = conv_name_base + 'up', kernel_initializer = glorot_uniform())(X_joined_input)
        X_uped = BatchNormalization(axis = 3, name = bn_name_base + 'up')(X_uped)
        X_uped = Activation('relu')(X_uped)
        return X_uped
    else:
        return X_joined_input
    
# FullVnet
# Output layers have 3 channels. The first two channels represent two one-hot vectors (pupil and non-pupil)
# The third layer contains all zeros in all cases (trivial)

def DeepVOG_net(input_shape = (240, 320, 3), filter_size= (3,3)):
    
    X_input = Input(shape=input_shape)#, batch_size=4)
    
    Nh, Nw = input_shape[0], input_shape[1]
    
    # Encoding Stream
    X_jump1, X_out = encoding_block(X = X_input, X_skip = 0, filter_size= filter_size, filters_num= 16,
                                      layer_num= 1, block_type = "down", stage = 1, s = 1)
    X_jump2, X_out = encoding_block(X = X_out, X_skip = X_out, filter_size= filter_size, filters_num= 32,
                                      layer_num= 1, block_type = "down", stage = 2, s = 1)
    X_jump3, X_out = encoding_block(X = X_out, X_skip = X_out, filter_size= filter_size, filters_num= 64,
                                      layer_num= 1, block_type = "down", stage = 3, s = 1)
    X_jump4, X_out = encoding_block(X = X_out, X_skip = X_out, filter_size= filter_size, filters_num= 128,
                                      layer_num= 1, block_type = "down", stage = 4, s = 1)
    
    # Decoding Stream
    X_out = decoding_block(X = X_out, X_jump = 0, filter_size= filter_size, filters_num= 256, 
                                 layer_num= 1, block_type = "up", stage = 1, s = 1, init_X_joint = 1)
    X_out = decoding_block(X = X_out, X_jump = X_jump4, filter_size= filter_size, filters_num= 256, 
                                 layer_num= 1, block_type = "up", stage = 2, s = 1, init_X_joint = 0)
    X_out = decoding_block(X = X_out, X_jump = X_jump3, filter_size= filter_size, filters_num= 128, 
                                 layer_num= 1, block_type = "up", stage = 3, s = 1, init_X_joint = 0)
    X_out = decoding_block(X = X_out, X_jump = X_jump2, filter_size= filter_size, filters_num= 64, 
                                 layer_num= 1, block_type = "up", stage = 4, s = 1, init_X_joint = 0)
    X_out = decoding_block(X = X_out, X_jump = X_jump1, filter_size= filter_size, filters_num= 32, 
                                 layer_num= 1, block_type = "up", stage = 5, s = 1, init_X_joint = 0, up_sampling = False)
    # Output layer operations
    X_out = Conv2D(filters = 3, kernel_size = (1,1) , strides = (1,1), padding = 'valid',
                   name = "conv_out", kernel_initializer = glorot_uniform())(X_out)
    X_out = Activation("softmax")(X_out)
    model = Model(inputs = X_input, outputs = X_out, name='Pupil')
    

    return model

def load_DeepVOG():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    base_dir = os.path.dirname(__file__)
    tf.keras.backend.set_floatx('float16')
    #K.set_floatx('float16')
    model = DeepVOG_net(input_shape = (240, 320, 3), filter_size= (10,10))
    model.load_weights(os.path.join(base_dir, "DeepVOG_weights_quantz_16.h5"))
    #model.save(os.path.join(base_dir, "DeepVOG_weights_quantz_16.h5"))
    
    # Convert the model.
    #import pathlib
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    ##converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]   
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    ##converter.target_spec.supported_types = [tf.float16]
    #tflite_fp16_model = converter.convert()
    #tflite_models_dir = pathlib.Path(os.path.join(base_dir))
    #tflite_model_fp16_file = tflite_models_dir/"full_model_quant_f16.tflite"
    #tflite_model_fp16_file.write_bytes(tflite_fp16_model)
    
    # Load TFLite model and allocate tensors.
    #interpreter = tf.lite.Interpreter(model_path=os.path.join(base_dir, "full_model_quant_f16.tflite"))#"full_model_lite.tflite"))
    #interpreter.allocate_tensors()

    # Get input and output tensors.
    #input_details = interpreter.get_input_details()
    #output_details = interpreter.get_output_details()
    #input_shape = input_details[0]['shape']

    
    #return interpreter
    return model

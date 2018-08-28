import tensorflow as tf#创建copressor class
import keras
import warnings
from keras.models import Model

class BaseCompresser(object):
    def __init__(self, 
                 model=None,
                 layers_name=None,
                 sparsity=None, 
                 metric_name=None,
                 metric_tolerance=None,
                 input_filepath=None,
                 output_filepath=None):

        if not isinstance(model, Model):
            raise ValueError('Currently only Keras model is supported.')
        self.model = model
        model_layers_name = [layer.name for layer in self.model.layers if layer.trainable == True and len(layer.get_weights()) == 2] 
        if isinstance(layers_name, tuple) or isinstance(layers_name, list):
            for layer in layers_name:
                if layer not in model_layers_name:
                    raise ValueError('The model does not have trainable layer: %s, please check layers_name list' % str(layer))
            self.layers_name = layers_name
        elif layers_name==None:
            self.layers_name = model_layers_name[::-1]
        else:
            raise ValueError('Use a list or tuple to specify layers to be compressed.')
        # TODO
        # sort layer_names
        if isinstance(sparsity, dict):
            for layer in sparsity.keys():
                if layer not in model_layers_name:
                    raise ValueError('The model does not have trainable layer: %s, please check sparsity dict' % str(layer))
        elif sparsity is not None:
            raise ValueError('sparsity should be a dict or None type.')
        self.sparsity = sparsity

        if metric_name not in model.metrics_names:
            raise ValueError('The model does not have the metric: %s' % str(metric_name))
        self.metric_name = metric_name #+ '_1'
        self.metric_index = model.metrics_names.index(metric_name)

        if metric_tolerance is None:
            raise ValueError('Spicify a tolerance for metric.')
        self.metric_tolerance = metric_tolerance
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def compress(self, #对外接口，直接调用这个函数，检测参数是不是正确，若正确，def _compres 重写这个方法来实现算法
                 x=None,
                 y=None,
                 batch_size=32,
                 epochs=1,
                 verbose=1,
                 validation_data=None,
                 shuffle=True,
                 class_weight=None,
                 sample_weight=None,
                 steps_per_epoch=None,
                 validation_steps=None):
        if x is None or y is None:#-71 jia
            raise ValueError('train data x and y is needed for model compression.') 
        if validation_data is None:
            raise ValueError('validation_data is needed for model compression.')
        elif not len(validation_data) == 2 or len(validation_data) == 3:
            raise ValueError('When passing validation_data, '
                             'it must contain 2 (x_val, y_val) '
                             'or 3 (x_val, y_val, val_sample_weights) '
                             'items, however it contains %d items' %
                             len(validation_data))

        self._compress(x=x,
                       y=y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle,
                       class_weight=class_weight,
                       sample_weight=sample_weight,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps)

    def _compress(self, 
                 x=None,
                 y=None,
                 batch_size=None,
                 epochs=1,
                 verbose=1,
                 validation_data=None,
                 shuffle=True,
                 class_weight=None,
                 sample_weight=None,
                 steps_per_epoch=None,
                 validation_steps=None):
        raise NotImplementedError
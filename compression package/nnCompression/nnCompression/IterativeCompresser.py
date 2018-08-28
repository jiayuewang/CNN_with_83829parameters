from .compresser import BaseCompresser
import numpy as np
# 初始化变量
class IterativeCompresser(BaseCompresser):
    def __init__(self, 
                 model=None, 
                 layers_name=None, 
                 sparsity=None,
                 metric_name=None,
                 metric_tolerance=None,
                 input_filepath=None,
                 output_filepath=None,
                 extra_sparsity=0.05,
                 decrease_rate=0.01):
        super(IterativeCompresser, self).__init__(model, layers_name, sparsity, metric_name, metric_tolerance, input_filepath, output_filepath)
        self.achieve_sparsity = True
        if self.sparsity is None:
            self.achieve_sparsity = False
        
        self.extra_sparsity = extra_sparsity
        self.decrease_rate = decrease_rate

    def _determine_sparsity(self, 
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
        sparsity_dict = {}    
        for layer_name in self.layers_name:
            sparsity = 0.99
            self.model.load_weights(self.input_filepath)
            layer = self.model.get_layer(name=layer_name)
            self._prune_layer(layer, sparsity)

            metrics_before = self._get_metric(x=validation_data[0],
                                              y=validation_data[1],
                                              batch_size=batch_size,
                                              sample_weight=sample_weight,
                                              steps=validation_steps)
                            
            while metrics_before <= self.metric_tolerance:
                # TODO
                # min or max ?
                self.model.fit(x=x,
                               y=y,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=0,
                               shuffle=shuffle,
                               class_weight=class_weight,
                               sample_weight=sample_weight,
                               steps_per_epoch=steps_per_epoch)
                self._prune_layer(layer, sparsity)
                metrics_after = self._get_metric(x=validation_data[0],
                                                 y=validation_data[1],
                                                 batch_size=batch_size,
                                                 sample_weight=sample_weight,
                                                 steps=validation_steps)
                if metrics_after <= metrics_before:
                    if sparsity > 0:
                        sparsity -= self.decrease_rate
                        if sparsity <= 0:
                            break
                
                metrics_before = metrics_after
            
            sparsity_dict[layer_name] = sparsity
        self.achieve_sparsity = True
        return sparsity_dict

    def _compress(self, # 重写，继承base compressor 创建class后，检测参数对不对，_compressor
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

        if not self.achieve_sparsity:
            self.sparsity = self._determine_sparsity(x=x,#计算sparcity
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
        for layer_name in self.sparsity.keys():
            self.sparsity[layer_name] += self.extra_sparsity
            if self.sparsity[layer_name] >= 1:
                self.sparsity[layer_name] = 0.99
        
        self._prune_network(self.sparsity)
        current_metric = self._get_metric(x=validation_data[0],
                                          y=validation_data[1],
                                          batch_size=batch_size,
                                          sample_weight=sample_weight,
                                          steps=validation_steps)
        while True:
            while True:
                self.model.fit(x=x,
                               y=y,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=0,
                               shuffle=shuffle,
                               class_weight=class_weight,
                               sample_weight=sample_weight,
                               steps_per_epoch=steps_per_epoch) 
                self._prune_network(self.sparsity)
                previous_metric = current_metric
                current_metric = self._get_metric(x=validation_data[0],
                                                  y=validation_data[1],
                                                  batch_size=batch_size,
                                                  sample_weight=sample_weight,
                                                  steps=validation_steps) 
                if current_metric >= previous_metric:
                    break
            if current_metric >= self.metric_tolerance:
                # TODO
                # min or max ?
                break
            else:
                for layer_name in self.sparsity:
                    self.sparsity[layer_name] -= self.decrease_rate
                    if self.sparsity[layer_name] <= 0:
                        self.sparsity[layer_name] = 0
                self.model.load_weights(self.input_filepath)

        if self.output_filepath is not None:
            self.model.save(self.output_filepath)

    def _prune_layer(self, layer, sparsity):
        weights, bias = layer.get_weights()
        flat = np.reshape(weights, [-1])
        if sparsity == 1:
            threshold = np.inf
        elif sparsity == 0.:
            threshold = -np.inf
        else:
            flat_list = sorted(map(abs,flat))
            threshold = flat_list[int(len(flat_list) * sparsity)]
        under_threshold = abs(weights) < threshold
        weights[under_threshold] = 0
        layer.set_weights([weights, bias])
        return None

    def _prune_network(self, sparsity):
        for layer_name in self.layers_name:
            layer = self.model.get_layer(name=layer_name)
            self._prune_layer(layer, sparsity[layer_name])
        return None

    def _get_metric(self,
                    x,
                    y,
                    batch_size,
                    sample_weight,
                    steps):
        metrics_before = self.model.evaluate(x=x, 
                                             y=y, 
                                             batch_size=batch_size, 
                                             verbose=0, 
                                             sample_weight=sample_weight, 
                                             steps=steps)
        if not isinstance(metrics_before, list):
            metrics_before = [metrics_before]
        metrics_before = metrics_before[self.metric_index]
        return metrics_before
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn

from agent.utils.strings import *

class AdaptiveNormalization(base.Layer):
    '''Adaptive Normalization can be used above existing normalization (linear) to 
    linearly scale and shift the input using variable parameters to restore the
    representative capacity of the network.
    It ensures that the transformation inserted in the network can represent the 
    identity transform.

    **Note:**
    - support single axis transformation

    Arguements:
        axis: An `int`, the axis that should be normalized, typically the 
            features axis.
        center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
        scale: If True, multiply by `gamma`. If False, `gamma` is
            not used. When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling can be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight
        trainable: Boolean, if `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: A string, the name of the layer.
    '''

    def __init__(self,
                    axis=-1,
                    center=True,
                    scale=True,
                    beta_initializer=init_ops.zeros_initializer(),
                    gamma_initializer=init_ops.ones_initializer(),
                    trainable=True,
                    name=None,
                    **kwargs):

        super(AdaptiveNormalization, self).__init__(name=name, 
                                                    trainable=trainable, 
                                                    **kwargs)  
        self.axis = axis
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)
        
        if not isinstance(self.axis, int):
            raise TypeError('axis must be int, type given: %s'
                                % type(self.axis))
        
        # Resolve negatives value for axis
        if self.axis < 0:
            self.axis = ndims + self.axis

        if self.axis < 0 or self.axis >= ndims:
            raise ValueError('Invalid axis: %d' % self.axis)

        axis_to_dim = input_shape[self.axis].value
        if axis_to_dim is None:
            raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                                input_shape)

        self.input_spec = base.InputSpec(min_ndim=2, 
                                            ndim=ndims, 
                                            axes={
                                                self.axis: axis_to_dim
                                            })

        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            param_dtype = dtypes.float32
        else:
            param_dtype = self.dtype or dtypes.float32

        param_shape = (axis_to_dim, )

        if self.scale:
            self.gamma = self.add_variable(name=GAMMA,
                                            shape=param_shape,
                                            dtype=param_dtype,
                                            initializer=self.gamma_initializer,
                                            trainable=True)
        else:
            self.gamma = None   

        if self.center:
            self.beta = self.add_variable(name=BETA,
                                            shape=param_shape,
                                            dtype=param_dtype,
                                            initializer=self.beta_initializer,
                                            trainable=True)
        else:   
            self.beta = None

        self.mean = constant_op.constant(0.0,
                                            dtype=dtypes.float32, 
                                            shape=param_shape,
                                            name=ZERO)
        self.variance = constant_op.constant(1.0,
                                            dtype=dtypes.float32, 
                                            shape=param_shape,
                                            name=UNIT)

        self.built = True

    def call(self, inputs):
        input_shape = inputs.get_shape()
        ndims = len(input_shape)

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis] = input_shape[self.axis].value
        def _broadcast(v):
            if (v is not None and
                        self.axis < (ndims - 1)):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        outputs = nn.batch_normalization(inputs,
                                            _broadcast(self.mean),
                                            _broadcast(self.variance),
                                            offset,
                                            scale,
                                            variance_epsilon=0.0,
                                            name=SCALE_SHIFT)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

def adaptive_normalization(inputs,
                            axis=-1,
                            center=True,
                            scale=True,
                            beta_initializer=init_ops.zeros_initializer(),
                            gamma_initializer=init_ops.ones_initializer(),
                            trainable=True,
                            name=None):
    '''Functional interface for the adaptive normalization layer
    '''

    layer = AdaptiveNormalization(axis=axis,
                                    scale=scale,
                                    beta_initializer=beta_initializer,
                                    gamma_initializer=gamma_initializer,
                                    trainable=trainable,
                                    name=name)
    return layer.apply(inputs)

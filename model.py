## model.py
# Store code for the network architecture for the basic Convoca model 
# 

import os

# Limit GPU consumption
DEVICE = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from utils import periodic_padding

## Define a network function in JAX

class CaModule(hk.Module):

    def __init__(self, data: jax.Array, config: dict, bc: str):
        """For now we assume the `data` object is a 3-dimensional
            array
    
            :param data: the X_train data, list of states (shape [B, H, W, 1])
            :param config: Dict, a configuration dictionary
            :param bc: boundary conditions - whether to use "periodic" or
                "constant" (zero padding) 
                For now we will do constant padding.
        """
        super().__init__()
        self.num_classes = config['num_classes']
        self.layer_dims = config['layer_dims']  # this should be a list of ints
        self.data = data

        if bc == 'periodic':
            # Need to add some wraparound layer
            self.data = periodic_padding(data, padding=1)
            self.conv_pad = 'valid'
        else:
            self.conv_pad = 'same'

        self.perception_block = hk.Sequential([
            hk.Conv2D(self.layer_dims[0], kernel_shape=(3,3), padding=self.conv_pad,
                      w_init=hk.initializers.VarianceScaling(scale=2.0), 
                      b_init=hk.initializers.VarianceScaling(scale=2.0)
                     ),
            jax.nn.relu,
            hk.Reshape(output_shape=(-1, self.layer_dims[0])),
        ])

        self.final_block = hk.Sequential([
            hk.Linear(self.num_classes,
                      w_init=hk.initializers.VarianceScaling(scale=2.0),
                      b_init=hk.initializers.VarianceScaling(scale=2.0)
                     ),
            jax.nn.relu     
        ])

    def __call__(self,):

        # The first layer of the network, just acquire the neighborhood
        x = self.perception_block(self.data)

        # Run the data through the internal layers
        for i in range(len(self.layer_dims)):
            _internal_layer = hk.Linear(self.layer_dims[i], 
                                           w_init=hk.initializers.VarianceScaling(scale=2.0),
                                           b_init=hk.initializers.VarianceScaling(scale=2.0)
                                        )
            x = jax.nn.relu(_internal_layer(x))

        # Final layer
        x = self.final_block(x)
        
        return x

def network_fn(data: jax.Array, config: dict, bc="periodic"):
    ca_module = CaModule(data, config, bc)
    return ca_module()

# def network_fn(data: jax.Array, config: Dict, bc="periodic"):
#     """For now we assume the `data` object is a 3-dimensional
#         array

#         :param data: the X_train data, list of states (shape [B, H, W, 1])
#         :param config: Dict, a configuration dictionary
#         :param bc: boundary conditions - whether to use "periodic" or
#             "constant" (zero padding) 
#             For now we will do constant padding.
#     """
#     num_classes = config['num_classes']
#     layer_dims = config['layer_dims']  # this should be a list of ints

#     if bc == 'periodic':
#         # Need to add some wraparound layer
#         data = periodic_padding(data, padding=1)
#         conv_pad = 'valid'
#     else:
#         conv_pad = 'same'

#     # w_init = hk.initializers.VarianceScaling(scale=2.0)
#     # b_init = hk.initializers.VarianceScaling(scale=2.0)

#     # TODO: This is very ugly I think
#     def get_internal_block(n_dim):
#         internal_layer = [
#             hk.Linear(layer_dims[n_dim], 
#                        w_init=hk.initializers.VarianceScaling(scale=2.0),
#                        b_init=hk.initializers.VarianceScaling(scale=2.0)),
#             jax.nn.relu,
#         ]
#         return internal_layer

#     network_lst = \        
#         [
#             hk.Conv2D(layer_dims[0], kernel_shape=(3,3), padding=conv_pad,
#                       w_init=hk.initializers.VarianceScaling(scale=2.0), 
#                       b_init=hk.initializers.VarianceScaling(scale=2.0)
#                      ),
#             jax.nn.relu,
#             hk.Reshape(output_shape=(-1, layer_dims[0])),
#         ] + 
        
#     network = hk.Sequential(
#         [
#             hk.Conv2D(layer_dims[0], kernel_shape=(3,3), padding=conv_pad,
#                       w_init=hk.initializers.VarianceScaling(scale=2.0), 
#                       b_init=hk.initializers.VarianceScaling(scale=2.0)
#                      ),
#             jax.nn.relu,
#             hk.Reshape(output_shape=(-1, layer_dims[0])),
#         ] +
#         # [
#         #     hk.Linear(layer_dims[i], 
#         #                w_init=hk.initializers.VarianceScaling(scale=2.0),
#         #                b_init=hk.initializers.VarianceScaling(scale=2.0)
#         #               ),
#         #     jax.nn.relu,
#         # ]
#         #     hk.Linear(layer_dims[2], 
#         #               w_init=hk.initializers.VarianceScaling(scale=2.0), 
#         #               b_init=hk.initializers.VarianceScaling(scale=2.0)
#         #              ),
#         #     jax.nn.relu,
            
#         #     # hk.Linear(layer_dims[i]) for i in range(1, len(layer_dims)),
#         #     hk.Linear(num_classes,
#         #               w_init=hk.initializers.VarianceScaling(scale=2.0),
#         #               b_init=hk.initializers.VarianceScaling(scale=2.0)
#         #              ),
#         #     jax.nn.relu
#         # ]
#     )
#     # print(network)
#     return network(data)

def logit_to_preds(logits, shape=None):
    """Transform logits to images"""
    labels = jnp.argmax(jax.nn.softmax(logits), axis=-1)
    if shape:
        out = jnp.reshape(labels, shape)
    return out.astype(np.float32)
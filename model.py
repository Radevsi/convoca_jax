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

def network_fn(data: jax.Array, layer_dims, bc="periodic"):
    """For now we assume the `data` object is a 3-dimensional
        array

        :param data: the X_train data, list of states (shape [B, H, W, 1])
        :param layer_dims: list of ints
        :param bc: boundary conditions - whether to use "periodic" or
            "constant" (zero padding) 
            For now we will do constant padding.
    """
    num_classes = 2

    if bc == 'periodic':
        # Need to add some wraparound layer
        data = periodic_padding(data, padding=1)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'

    # w_init = hk.initializers.VarianceScaling(scale=2.0)
    # b_init = hk.initializers.VarianceScaling(scale=2.0)
        
    network = hk.Sequential(
        [
            hk.Conv2D(layer_dims[0], kernel_shape=(3,3), padding=conv_pad,
                      w_init=hk.initializers.VarianceScaling(scale=2.0), 
                      b_init=hk.initializers.VarianceScaling(scale=2.0)
                     ),
            jax.nn.relu,
            hk.Reshape(output_shape=(-1, layer_dims[0])),

             hk.Linear(layer_dims[1], 
                       w_init=hk.initializers.VarianceScaling(scale=2.0),
                       b_init=hk.initializers.VarianceScaling(scale=2.0)
                      ),
            jax.nn.relu,
            hk.Linear(layer_dims[2], 
                      w_init=hk.initializers.VarianceScaling(scale=2.0), 
                      b_init=hk.initializers.VarianceScaling(scale=2.0)
                     ),
            jax.nn.relu,
            
            # hk.Linear(layer_dims[i]) for i in range(1, len(layer_dims)),
            hk.Linear(num_classes,
                      w_init=hk.initializers.VarianceScaling(scale=2.0),
                      b_init=hk.initializers.VarianceScaling(scale=2.0)
                     ),
            jax.nn.relu
        ]
    )
    # print(network)
    return network(data)

def logit_to_preds(logits, shape=None):
    """Transform logits to images"""
    labels = jnp.argmax(jax.nn.softmax(logits), axis=-1)
    if shape:
        out = jnp.reshape(labels, shape)
    return out.astype(np.float32)
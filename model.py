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

    def __init__(self, config: dict, padding: str):
        """For now we assume the `data` object is a 3-dimensional
            array
    
            :param config: dict, a configuration dictionary
            :param padding: str, define the boundary condition - i.e. either
                "valid" or "same", as in Haiku module

            *Note a previous iteration of this code defined this parameter as `bc`
            for "boundary condition"*
        """
        super().__init__()
        self.num_classes = config['num_classes']
        self.layer_dims = config['layer_dims']  # this should be a list of ints
        self.padding = padding

        # if bc == 'periodic':
        #     # Need to add some wraparound layer
        #     self.data = periodic_padding(data, padding=1)
        #     self.conv_pad = 'valid'
        # else:
        #     self.conv_pad = 'same'

        self.perception_block = hk.Sequential([
            hk.Conv2D(self.layer_dims[0], kernel_shape=(3,3), padding=self.padding,
                      w_init=hk.initializers.VarianceScaling(scale=2.0), 
                      b_init=hk.initializers.VarianceScaling(scale=2.0)
                     ),
            jax.nn.relu,
            hk.Reshape(output_shape=(-1, self.layer_dims[0])),
        ])

        # self.zero_initializer = hk.initializers.Constant(0)
        
        self.final_block = hk.Sequential([
            hk.Linear(self.num_classes,
                      # w_init=self.zero_initializer,
                      w_init=hk.initializers.VarianceScaling(scale=2.0),
                      b_init=hk.initializers.VarianceScaling(scale=2.0)
                     ),
            jax.nn.relu     
        ])

    def __call__(self, x):

        # Handle boundaries
        if self.padding == 'valid':
            x = periodic_padding(x, padding=1)
        else:
            x = x

        # The first layer of the network, just acquire the neighborhood
        x = self.perception_block(x)

        # Run the data through the internal layers
        for i in range(len(self.layer_dims)):
            _internal_layer = hk.Linear(self.layer_dims[i], 
                                        # w_init = self.zero_initializer,
                                           w_init=hk.initializers.VarianceScaling(scale=2.0),
                                           b_init=hk.initializers.VarianceScaling(scale=2.0)
                                        )
            x = jax.nn.relu(_internal_layer(x))

        # Final layer
        x = self.final_block(x)
        
        return x

def network_fn(x: jax.Array, config: dict, n_iter=1, padding="valid",):
    """
    :param x: the X_train data, list of states (shape [B, H, W, 1])
    :param n_iter: int, defines how many times to run the data through the Ca Module 
    This is akin to the Growing NCA model...

    The other params are defined in the docstring for the CaModule class.
    """
    # Instantiate the Ca Module class and run the data through it once
    ca_module = CaModule(config, padding)
    x = ca_module(x) 
    
    # If there is more to run, do it.
    for _ in range(1, n_iter):

        # Because the model currently outputs a one-hot-encoding, we need to convert that
        # encoding back into images in order to pass it into the model again
        # This could be a cause for failure during training, I'm not sure...

        # We want x to have shape [B, H, W, 1]
        x = logit_to_preds(x, shape=(-1, config['wspan'], config['hspan'], 1)) 
        x = ca_module(x)
    
    # Return the final result
    return x

def logit_to_preds(logits, shape=None):
    """Transform logits to images"""
    labels = jnp.argmax(jax.nn.softmax(logits), axis=-1)
    if shape:
        out = jnp.reshape(labels, shape)
    return out.astype(np.float32)


###################### Neural CA Model ######################

## This is bad but write quickly a neural CA model ##

class NeuralCA(hk.Module):
    def __init__(self, channel_n=16, perception_dim=48, hidden_dim=128, padding='valid'):
        super().__init__()
        self.channel_n = channel_n
        self.perception_dim = perception_dim
        self.hidden_dim = hidden_dim

        self.dmodel = hk.Sequential([
            hk.Conv2D(self.perception_dim, kernel_shape=(3,3), padding=padding, with_bias=False,
                      # w_init=hk.initializers.VarianceScaling(scale=2.0), 
                      # b_init=hk.initializers.VarianceScaling(scale=2.0)
                     ),
            hk.Conv2D(self.hidden_dim, kernel_shape=(1,1)),
            jax.nn.relu,
            hk.Conv2D(self.channel_n, kernel_shape=(1,1),
                      w_init=hk.initializers.Constant(0)
                     )
        ])


    # self.update = nn.Sequential(nn.Conv2d(state_dim, 3*state_dim, 3, padding=1, bias=False),  # perceive
    #                             nn.Conv2d(3*state_dim, hidden_dim, 1),  # process perceptual inputs
    #                             nn.ReLU(),                              # nonlinearity
    #                             nn.Conv2d(hidden_dim, state_dim, 1))    # output a residual update            

    def __call__(self, x):
        y = self.dmodel(x)
        return y

def neural_fn(x: jax.Array, config: dict, n_iter=1, padding="valid",):

    neural_ca = NeuralCA(padding=padding)

    # If there is more to run, do it.
    for _ in range(n_iter):
        x = neural_ca(x)
    
    # Return the final result
    return x


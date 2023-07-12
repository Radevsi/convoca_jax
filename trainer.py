## train.py
# Contains the code for running a training sequence
# of some given network

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
import optax

import wandb
import matplotlib.pyplot as plt
from typing import NamedTuple
from tqdm import tqdm
import functools

from model import network_fn, logit_to_preds


class Batch(NamedTuple):
    input_states: np.ndarray  # [B, H, W, 1]
    output_states: np.ndarray  # [B, H, W, 1]

class TrainState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

class Trainer:
    def __init__(
        self, 
        network_fn,
        config,  # a config dictionary to use in the wandb run instance
        X_train,
        Y_train,
    ):
        self.X_train = X_train
        self.Y_train = Y_train

        # Create the pure network transformation
        self.transform = hk.without_apply_rng(hk.transform(network_fn))

        # Create the optimizer
        learning_rate = config['learning_rate']
        self.optimizer = optax.adam(learning_rate)

        # Create a wandb training instance
        wandb.login()
        self.run = wandb.init(
            project='convoca_jax',
            notes='Created group',
            group='preliminary_plots',
            config=config
        )

    # First make an init function to initialize params and opt state
    def init(self) -> TrainState:
        """Initialize params and optimizer state
        """
        print("Initializing model")

        # Set an rng key
        rng = jax.random.PRNGKey(np.random.randint(100))

        # First make a batch of data using X_train and Y_train globals
        # Add a singleton dimension to each dataset
        batch = Batch(np.array(self.X_train[..., np.newaxis]), 
                    np.array(self.Y_train[..., np.newaxis]))

        # Initialize params and optimizer state
        layer_dims = self.run.config['layer_dims']
        params = self.transform.init(rng, batch.input_states, layer_dims)
        opt_state = self.optimizer.init(params)

        return TrainState(params, opt_state), batch

    # @jax.jit
    def loss_fn(self, params: hk.Params, batch: Batch) -> jax.Array:
        """Loss function
        """
        num_classes = self.run.config['num_classes']
        layer_dims = self.run.config['layer_dims']

        # Constructs a one-hot encoding of the true output states
        true_output_states = jnp.squeeze(jax.nn.one_hot(batch.output_states, 
                                                        num_classes))
        labels = np.reshape(true_output_states, newshape=(-1, num_classes))

        # Get the output of the model and convert the shapes
        network_output = self.transform.apply(params, batch.input_states, layer_dims)    
        logits = np.reshape(network_output, newshape=(-1, num_classes))
        
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
        loss = jnp.mean(loss)

        return loss, (loss, network_output)

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: TrainState, batch: Batch) -> TrainState:
        """Implement the learning rule
        """
        # Take gradients first
        grads, (loss, network_output) = jax.grad(self.loss_fn, has_aux=True)(state.params, batch)

        # Pass grads through optimizer
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainState(params, opt_state), (loss, network_output)
        
    def train(self,
              epochs,
              print_every=50, 
              log_every=50
    ):

        # Get initial states
        state, batch = self.init()
        # return state, batch
        columns = ["Input state", "Network Prediction", "Expected Output", "Epoch"]
        table = wandb.Table(columns=columns)
                
        for epoch in tqdm(range(epochs)):
            state, (loss, network_output) = self.update(state, batch)

            self.run.log({
                "epoch": epoch,
                "loss": loss,
                # "model_params": wandb.Histogram(state.params)
            })
            
            if epoch % print_every == 0:
                print(f"loss at epoch {epoch}: {loss:.8f}")

            if epoch % log_every == 0:
                wspan, hspan = self.run.config['wspan'], self.run.config['hspan']
                preds = logit_to_preds(network_output, shape=(-1, wspan, hspan)) 

                # Log images to the table
                ind = np.random.randint(len(preds))
                table.add_data(
                    wandb.Image(plt.imshow(batch.input_states[ind], cmap='gray')),
                    wandb.Image(plt.imshow(preds[ind], cmap='gray')),
                    wandb.Image(plt.imshow(batch.output_states[ind], cmap='gray')),
                    epoch
                )
                plt.close()
    
        self.run.log({"network_predictions": table})

        # self.run.finish()
        # Set globals in case object wants to use them
        self.state = state
        self.loss = loss
        self.batch = batch

        # And return - Is this good design? 
        return state, loss, batch

    ## Include some helpful evaluation functions
    # TODO: could refactor the `apply_network` code to make use of 
    # the `loss_fn`
    def apply_network(self, state: TrainState = None, 
                      input_states: np.ndarray = None, 
                      return_logits=False
    ) -> jax.Array:
        """Use the global `transform` and passed-in state to 
            get a network output from `input_states`.
    
            :param return_logits: bool, whether the function should just 
            return the network logits or convert to predictions
        """
        layer_dims = self.run.config['layer_dims']
        wspan, hspan = self.run.config['wspan'], self.run.config['hspan']

        # Use object's states if passed-in ones are None
        if state is None:
            state = self.state
        if input_states is None:
            input_states = self.batch.input_states
        
        logits = self.transform.apply(state.params, input_states, layer_dims)
        preds = logit_to_preds(logits, shape=(-1, wspan, hspan))   
        if return_logits:
            return logits, preds
        else:
            return preds
    
    def evaluate(self, state: TrainState = None, batch: Batch = None) -> jax.Array:
        """Get model outputs and evaluate model performance on a batch
            of data
    
            Note: By default, the output of the network has shape
            [B, H*W, num_classes], we want to reshape the labels 
            to have this shape.
        """
        num_classes = self.run.config['num_classes']
        wspan, hspan = self.run.config['wspan'], self.run.config['hspan']

        if state is None:
            state = self.state
        if batch is None:
            batch = self.batch

        # Get one-hot-encodings of the expected output
        # and convert to labels
        true_output_states = jnp.squeeze(jax.nn.one_hot(batch.output_states, 
                                                        num_classes)) 
        labels = np.reshape(true_output_states, newshape=(-1, 
                                                          wspan*hspan,
                                                          num_classes))
        # Get network logits
        logits, preds = self.apply_network(state, batch.input_states, return_logits=True)
        print(logits.shape)
        
        # Compute the loss 
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels) 
    
        return jnp.mean(loss, axis=1), preds
        




import jax
import numpy as np
import jax.numpy as jnp

## Use the one from convoca helpers
def periodic_padding(image, padding=1):
    """Do padding (wrap around) for an image stack.
        Similarly to Gilpin's work, pad along the last two
        axes for 3-dimensional, batched images
    """
    
    if len(image.shape) == 2:
        upper_pad = image[-padding:, :]
        lower_pad = image[:padding, :]
    
        partial_image = np.concatenate([upper_pad, image, lower_pad], axis=0)
    
        left_pad = partial_image[:, -padding:]
        right_pad = partial_image[:, :padding]
    
        padded_image = np.concatenate([left_pad, partial_image, right_pad], axis=1)

    if len(image.shape) == 3:
        upper_pad = image[:, -padding:, :]
        lower_pad = image[:, :padding, :]
    
        partial_image = np.concatenate([upper_pad, image, lower_pad], axis=1)
    
        left_pad = partial_image[:, :, -padding:]
        right_pad = partial_image[:, :, :padding]
    
        padded_image = np.concatenate([left_pad, partial_image, right_pad], axis=2)

    else:
        assert True, "Input data shape not understood."
        
    return padded_image

def make_game_of_life():
    """Creates Game of Life functionality only through convolutional operations
        Returns a function (`make_ca`) to be called on a batch/stack of images
    """
    
    pad_size = 1
    center_pixel_filter = np.zeros((3, 3))
    center_pixel_filter[1, 1] = 1
    outer_pixel_filter = np.ones((3, 3))
    outer_pixel_filter[1, 1] = 0

    all_filters = np.dstack(
        (center_pixel_filter, outer_pixel_filter, outer_pixel_filter,
         outer_pixel_filter, outer_pixel_filter)
    )
    all_biases = np.array([0, -1, -2, -3, -4])
    total_filters = len(all_biases)
    kernel = all_filters[:, :, np.newaxis, :]
    biases = all_biases

    wh1 = np.array([
        [0, 0, 4/3, -8/3, -1/3],
        [3/2, 5/4, -5, -1/4, -1/4]
    ]).T
    bh1 = np.array([-1/3,-7/4]).T

    def make_ca(image_stack: jax.Array) -> jax.Array:

        input_padded = periodic_padding(image_stack, pad_size)[..., np.newaxis].astype(float)

        # Apply the convolutions
        conv_image = jax.lax.conv_general_dilated(input_padded,
                                  kernel,
                                  window_strides=(1, 1),
                                  padding='VALID',
                                  dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                 )

        # print(f"Shape of conv_image: {conv_image.shape}")
        
        activation_image = jax.nn.relu(conv_image + biases)

        activated_flat = np.reshape(activation_image, [-1, total_filters])
        h1 = jax.nn.relu(np.matmul(activated_flat, wh1) + bh1)

        scores = jnp.sum(h1, axis=-1)
        next_states = np.reshape(scores, [*activation_image.shape[:3], 1])

        return np.squeeze(next_states)
        
    return make_ca

def dataloader(n_states, wspan, hspan):
    """Generate data

        :param n_states: int, number of cell states possible
        :param wspan: int, width of grid
        :param hspan: int, height of grid

        :return: np.ndarray, np.ndarray
    """ 
    def generate_gol_data(train_size):
        """
        :param train_size: int, number of (x,y) data pairs
        """
        
        # Make X data
        X_train = np.random.choice(list(range(n_states)), 
            size=(train_size, wspan, hspan)
        ).astype(np.float32)
    
        # Generate Y data
        gol = make_game_of_life()
        Y_train = gol(X_train)
    
        return X_train, Y_train

    return generate_gol_data

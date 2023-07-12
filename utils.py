import jax

# This version of the function is to be used inside the network
def periodic_padding(imbatch, padding=1):
    '''
    Create a periodic padding (wrap) around an image batch, to emulate 
    periodic boundary conditions. Padding occurs along the middle two axes
    '''
    pad_u = imbatch[:, -padding:, :]
    pad_b = imbatch[:, :padding, :]

    partial_image = jax.lax.concatenate([pad_u, imbatch, pad_b], 1)

    pad_l = partial_image[..., -padding:, :]
    pad_r = partial_image[..., :padding, :]

    padded_imbatch = jax.lax.concatenate([pad_l, partial_image, pad_r], 2)
          
    return padded_imbatch
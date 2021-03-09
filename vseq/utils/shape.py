from typing import List, Tuple


def concatenate_shapes(shapes: List[Tuple[int]], axis: int):
    """Concatenate shapes along axis"""
    out = list(shapes[0])
    out[axis] = sum(list(s)[axis] for s in shapes)
    return tuple(out)


def elevate_sample_dim(tensor, n_samples):
    """Elevate a tensor's samples dimension from shape (B * S, *D) to (S, B, *D)"""
    batch_size = tensor.shape[0]
    assert (
        batch_size % n_samples == 0
    ), f'Number of samples does not divide the "batch size" ({batch_size}/{n_samples} = {batch_size/n_samples})'

    new_batch_size = batch_size // n_samples
    new_shape = (n_samples, new_batch_size, *(tensor.size()[1:]))

    return tensor.view(new_shape)

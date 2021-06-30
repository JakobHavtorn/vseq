import torch
import numpy as np

def greedy_ctc(logits, seq_lens, blank=0):
    """
    Decodes the output from a CTC network.

    Args:
        logits (torch.Tensor or np.ndarray): CTC output of shape (time x batch x out_dim).
    
    Returns:
        list of strings: The decoded output sequences.
    """

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    
    preds = logits.argmax(axis=2).T
    repeat_filter = np.ones(preds.shape, dtype=np.bool)
    repeat_filter[:, 1:] = (preds[:, 1:] != preds[:, :-1])

    decoded = []
    for i, l in enumerate(seq_lens.tolist()):
        collapsed = preds[i, :l][repeat_filter[i, :l]]
        hyp = collapsed[collapsed != blank].tolist()
        decoded.append(hyp)
    
    return decoded

def greedy_standard(logits, seq_lens, blank=0):
    """
    Decodes the output from a AED model or other cross-entropy trained network.

    Args:
        logits (torch.Tensor or np.ndarray): Output of shape (time x batch x out_dim).
    
    Returns:
        list of ints: The decoded output sequences.
    """

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    decoded = []
    for i, l in enumerate(seq_lens.tolist()):
        hyp = logits[:l, i].argmax(axis=-1).tolist()
        decoded.append(hyp)
    
    return decoded
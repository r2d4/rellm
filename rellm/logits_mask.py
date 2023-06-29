
import numpy as np
from transformers import LogitsProcessor


class LogitsMask(LogitsProcessor):
    """
    LogitsMask is a LogitsProcessor that masks logits for tokens that are 
    not in the allowed token ids set.
    """
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        device = scores.device
        scores = scores.cpu()
        mask = np.ones_like(scores) * -1e10
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        scores = scores + mask 
        return scores.to(device)

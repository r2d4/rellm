
import numpy as np
import regex
from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizer

from rellm.re_token_validator import ReTokenValidator


class CustomLogitsMask(LogitsProcessor):
    """
    CustomLogitsMask is a LogitsProcessor that masks logits for tokens that are 
    not in the allowed token ids set.
    """
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = np.ones_like(scores) * -1e10
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        scores = scores + mask 
        return scores

def complete_re(prompt:str, pattern: regex.Pattern, tokenizer: PreTrainedTokenizer, 
                model: PreTrainedModel, max_new_tokens: int = 3, 
                stop_after_match: bool = True,
                debug: bool = False,
                **model_kwargs):
    """
    Complete a prompt with a regex pattern.
    """
    print("complete_re: prompt={}, pattern={}".format(prompt, pattern))
    gen_tokens = 0
    partial_completion = ""
    prompt_plus_completion = prompt + partial_completion

    token_validator = ReTokenValidator(tokenizer)

    if isinstance(pattern, regex.Pattern):
        pattern = [pattern]

    while gen_tokens < max_new_tokens:
        prompt_token_ids = tokenizer.encode(prompt_plus_completion, return_tensors="pt")
        prompt_length = prompt_token_ids.shape[1]

        allowed_token_ids = token_validator.get_valid_next_tokens(partial_completion, pattern)
        custom_mask_processor = CustomLogitsMask(allowed_token_ids)

        output_ids = model.generate(prompt_token_ids,
                                    max_new_tokens=1,
                                    pad_token_id=tokenizer.eos_token_id,
                                    logits_processor=[custom_mask_processor],
                                    **model_kwargs
        )
        new_token_ids = output_ids[0, prompt_length:]
        output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        partial_completion += output_text
        prompt_plus_completion = prompt_plus_completion + output_text
        if debug:
            print("step={} completion={}".format(gen_tokens, partial_completion))

        if stop_after_match:
            for p in pattern:
                if p.fullmatch(partial_completion):
                    return partial_completion
                
        gen_tokens += 1

    return partial_completion
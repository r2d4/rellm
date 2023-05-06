
import regex
from lark import UnexpectedCharacters, UnexpectedInput
from transformers import PreTrainedModel, PreTrainedTokenizer

from rellm.rellm import complete_re


def extract_terminal_regex(parser, stop_token):
    regex_map = {}
    for term in parser.terminals:
        if term.pattern:
            regex_map[term.name] = regex.compile(term.pattern.to_regexp())
    
    regex_map['$END'] = regex.compile(stop_token)
    return regex_map

class ParserState():
    def __init__(self, parser):
        self.parser = parser
        self.last_expected = []
        self.partial_token = ""
    
    def next_lex(self, input_str):
        try:
            print("input_str: ", input_str)
            self.parser.parse(input_str)
        except UnexpectedCharacters:
            # return the last set of expected tokens if we're mid-token
            print("partial_token: ", self.partial_token, "last_expected: ", self.last_expected)
            self.partial_token = input_str
            return self.last_expected
        except UnexpectedInput as e:
            expected_tokens = e.expected
            self.last_expected = expected_tokens
            print("expected_tokens: ", expected_tokens)
            return expected_tokens
 
        return []

def complete_cf(prompt:str, parser, partial_completion, tokenizer: PreTrainedTokenizer, 
                model: PreTrainedModel, max_new_tokens: int = 3, 
                debug: bool = False,
                **model_kwargs):
    """
    Complete a prompt with a regex pattern.
    """
    gen_tokens = 0
    prompt_plus_completion = prompt + partial_completion

    terminal_regexes = extract_terminal_regex(parser, tokenizer.decode(tokenizer.eos_token_id))
    parser_state = ParserState(parser )
    
    while gen_tokens < max_new_tokens:
        prompt_token_ids = tokenizer.encode(prompt_plus_completion, return_tensors="pt")
        prompt_token_ids.shape[1]

        valid_next_lex = parser_state.next_lex(partial_completion)
        if len(valid_next_lex) == 0 or (len(valid_next_lex) == 1 and '$END' in valid_next_lex):
            break
        r = [terminal_regexes[t] for t in valid_next_lex]

        next_token_completion = complete_re(prompt_plus_completion, r, tokenizer, model, stop_after_match=True, debug=debug, **model_kwargs)

        partial_completion += next_token_completion
        prompt_plus_completion = prompt_plus_completion + next_token_completion
        gen_tokens += 1

    return partial_completion


from lark import Lark
from transformers import AutoModelForCausalLM, AutoTokenizer

from rellm.cf_llm import complete_cf

model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")

# model = AutoModelForCausalLM.from_pretrained("distilgpt2", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2", trust_remote_code=True)

json_grammar = r"""
    ?start: value

    ?value: object
          | array
          | string
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : string ":" value

    string : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS

    %ignore WS
"""


### Create the JSON parser with Lark, using the Earley algorithm
# json_parser = Lark(json_grammar, parser='earley', lexer='basic')
# def parse(x):
#     return TreeToJson().transform(json_parser.parse(x))

### Create the JSON parser with Lark, using the LALR algorithm
json_parser = Lark(json_grammar, parser='lalr',
                # Using the basic lexer isn't required, and isn't usually recommended.
                   # But, it's good enough for JSON, and it's slightly faster.
                   lexer='basic',
                   # Disabling propagate_positions and placeholders slightly improves speed
                   propagate_positions=False,
                   maybe_placeholders=False,
                   regex=True)

prompt = "Write the first three letters of the alphabet in valid JSON format\n"
print(complete_cf(prompt, json_parser, "", 
                  tokenizer, 
                  model,
                  max_new_tokens=15, 
                  debug=True))

print("regular\n", ' '.join(tokenizer.batch_decode(model.generate(tokenizer.encode(prompt, return_tensors="pt"),
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id,
))))
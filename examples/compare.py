import sys
from pathlib import Path

import regex
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rellm import complete_re  # noqa: E402

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

examples = [
    {
        "prompt": "Return the first three letters of the alphabet in a json array:",
        "pattern": regex.compile(r'\[\"[a-z]\", \"[a-z]\", \"[a-z]\"\]'),
        "max_new_tokens": 10,
    },
    {
        "prompt": "Fill in the sentence with an interesting story about the dentist:",
        "pattern": regex.compile(r'Today I\'m going to the [a-z]+ to [a-z]+ because ([a-z]+ )*\.'),
        "max_new_tokens": 20,
    },
    {
        "prompt": "Is this a good demo?",
        "pattern": regex.compile(r'(Yes|No)\.'),
        "max_new_tokens": 2,
    },
    {
        "prompt": "Convert the date May 4, 2023 to the format mm/dd/yyyy:",
        "pattern": regex.compile(r'[0-9]{2}/[0-9]{2}/[0-9]{4}'),
        "max_new_tokens": 20,
    },
    {
        "prompt": "Jeff Dean is a ",
        "pattern": regex.compile(r'(Programmer|Computer Scientist|AGI)'),
        "max_new_tokens": 10,
    },
    {
        "prompt": 'I can eat ',
        "pattern": regex.compile(r'[0-9]{1,10} [a-z]* of [a-z]*'),
        "max_new_tokens": 10,
        "do_sample": True,
    },
    {
        "prompt": 'ReLLM, the best way to get structured data out of LLMs, is an acronym for ',
        "pattern": regex.compile(r'Re[a-z]+ L[a-z]+ L[a-z]+ M[a-z]+'),
        "max_new_tokens": 10,
        "do_sample": True,
    }
]

for example in examples:
    print("\n===Prompt===\n", example["prompt"])
    output = complete_re(tokenizer=tokenizer, model=model,**example)
    print("\n===ReLLM===\n", output)
    vanilla_output_ids = model.generate(tokenizer.encode(example["prompt"], return_tensors="pt"),
                                        max_new_tokens=30, 
                                        pad_token_id=tokenizer.eos_token_id, 
                                        do_sample=True)
    print("\n===Without ReLLM===\n", tokenizer.decode(vanilla_output_ids[0])[len(example["prompt"]):])

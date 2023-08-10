# ReLLM
Regular Expressions for Language Model Completions.

> *Some people, when confronted with a problem, think
“I know, I'll use regular expressions.”   Now they have two problems.*

Exact structure out of any language model completion with regular expressions.

Return specific syntactic structure (e.g. JSON or XML), or specific semantic structure (e.g. a date or a number), or even complete templates (e.g. a sentence with a blank to fill in).

How does it work? ReLLM filters non-matching tokens pre-generation. For each token, ReLLM tests every possible completion against a partial regex. For the potential completions that do not match the pattern, ReLLM masks the logits so that the language model does not generate them.

*If you are looking for a hosted version of ReLLM, check out the Thiggle Regex Completion API at [github.com/thiggle/api](https://github.com/thiggle/api)*

### Installation
```
pip install rellm
```

The preliminary results are interesting -- even for small models, constraining the token space with ReLLM can improve the quality of the completions. Not to mention the ability to more easily parse the output programmatically. Take a look at some of the [examples](examples). For an example of parsing a context-free grammar (like JSON) with ReLLM, see [r2d4/parserllm](https://github.com/r2d4/parserllm).

```python
import regex
from transformers import AutoModelForCausalLM, AutoTokenizer

from rellm import complete_re

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "ReLLM, the best way to get structured data out of LLMs, is an acronym for "
pattern = regex.compile(r'Re[a-z]+ L[a-z]+ L[a-z]+ M[a-z]+')
output = complete_re(tokenizer=tokenizer, 
                     model=model, 
                     prompt=prompt,
                     pattern=pattern,
                     do_sample=True,
                     max_new_tokens=80)
print(output)
```

```
> Realized Logistic Logistics Model
```


## Examples using GPT2 (124 million parameters)

#

Using GPT2 (124m)

**Prompt**: ReLLM, the best way to get structured data out of LLMs, is an acronym for

**Pattern**: Re[a-z]+ L[a-z]+ L[a-z]+ M[a-z]+

**ReLLM**: Realized Logistic Logistics Model

**Without ReLLM**: Largest Largest Address Space (MELSP), which has its roots in the  Internet network, at least when compared
#

**Prompt**: Return the first three letters of the alphabet in a json array:

**Pattern** \[\"[a-z]\", \"[a-z]\", \"[a-z]\"\]

**ReLLM**: ["a", "b", "c"]

**Without ReLLM**: { "index": 0, "id":"1", "description":"", "text": "[{ "id": 0, "name":
#
**Prompt**: Fill in the sentence with an interesting story about the dentist:

**Pattern**: Today I\'m going to the [a-z]+ to [a-z]+ because ([a-z]+ )*\.

**ReLLM**: Today I'm going to the dentist to see because it is a very important day for me

**Without ReLLM**: 'My family bought me an appointment with a dentist when I was 15. The dentist gave me one a year and then I was told on
#

**Prompt**: Is this a good demo?

**Pattern**: (Yes|No)

**ReLLM**: No.

**Without ReLLM**: I don't know, but this is amazing! Even more amazing is how the design can take place on a small stage that uses LEDs.
As

#

**Prompt**: Convert the date May 4, 2023 to the format mm/dd/yyyy:

**Pattern**: [0-9]{2}/[0-9]{2}/[0-9]{4}

**ReLLM**: 00/00/0045

**Without ReLLM**:  mm:ss

A-Z, Z-A, W-H (0-9:9:19)

Z-R

#

**Prompt**: Jeff Dean is a

**Pattern** (Programmer|Computer Scientist|AGI)

**ReLLM**: Computer Scientist

**Without ReLLM**: former national basketball champion and a former professional basketball player. He currently serves as general counsel for the NCAA Office of the Vice President for Academic Affairs.

#

**Prompt**: I can eat 

**Pattern**: [0-9]{1,10} [a-z]* of [a-z]*

**ReLLM**: 800 calories of coffee

**Without ReLLM**: iced coffee here on the west side and do this, so can you?"

"Why, I don't understand. What did you mean by
*This project has been created as part of the 42 curriculum by <a href ="https://github.comsylvzzz">dbotelho</a>.*

# Call-me-Maybe
Introduction to Function calling and parsing with LLM's

# Description

This project introduces `Function Calling` in Large Language Models by building a system that translates natural human language prompts into structured function calls with typed arguments. The main goal and skill larned with this project was the process of `Constrained Decoding` to guarantee valid JSON output, achieving perfect reliability with a small 0.5 Bilion parameter model (for reference a popular LLM as Chat GPT has ~3 Trilion), optimizing generation of valid function calling output in form of JSON on weak models.

# Instructions

## Installation

#### Installing in a 42 School Computer
If you are running this in a 42 school computer, please run this in the `sgoinfre` directory, since in 42 we dont have a lot of space in home directory

```bash
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/sgoinfre/$(USER)/.local/bin sh

# run this when you're in the project directory

export HF_HOME="$PWD/.cache/huggingface"
export UV_CACHE_DIR="$PWD/.cache/uv"
export PIP_CACHE_DIR="$PWD/.cache/pip"

make
```

#### Installing in your personal computer
```bash
curl -LsSf https://astral.sh/uv/install.sh

make
```

## What is an LLM doing, really?
An LLM is basically just a next-word guesser. A user sends it a text and it guesses what comes next.
But computers don't understand text, so everything gets converted to numbers first.

### A simple example of a llm loop
<pre>
The full journey, simply
"What is 2 + 3?"
       |
   TOKENIZATION
       |
["What", "is", "2", "+", "3", "?"]
       |
   INPUT IDs (vocab lookup)
       |
[892, 318, 17, 10, 18, 30]
       |
   LLM (neural network)
       |
   LOGITS (scores for every possible next token)
[token_0: -2.1, token_1: 0.3, token_42: 8.7, token_99: -0.1, ...]
       |
   PICK HIGHEST SCORE → token_42 = "5"
       |
   APPEND "5" TO INPUT, REPEAT...</pre>

## What are logits?
Logits are like a list of scores, one per token in the vocabulary. If the vocab has 100,000 tokens, you get 100,000 scores. The highest score = the token the model thinks comes next.
```python

vocab size = 100,000 tokens
logits     = [0.1, -2.3, 8.7, 0.02, ...]
                          |
                      # this one wins → thats the next token
```

The model repeats this loop, one token at a time, until it generates an EOS token (end of sentence) the llm sending the signal that he's done.

On this project, our workflow differs just a bit from this, because just picking the highest score for the next token is a technique called greedy sampling, and with that we would be relying on the model to return always valid JSON and thats neither a good practice neither whats wanted from this project. What we do instead is what's expected from us, the <b>Constrained Decoding</b>.

## Constrained Decoding

Constrained Decoding is a technique that restricts Large Language Model (LLM) outputs to comply to strict syntactic, structural, or semantic rules by masking invalid tokens during the generation process. Instead of relying on post-processing or prompting, it ensures that every generated token maintains the sequence within a valid language defined by constraints.

Normally the model can pick any token freely. But to generate valid JSON for function calling, we want the output to always be valid JSON.
Before picking the next token, you look at the logits and set invalid tokens to -infinity so they can never be picked:

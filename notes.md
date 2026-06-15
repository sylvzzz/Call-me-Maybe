What is an LLM doing, really?
An LLM is basically a next-word guesser. That's it. You give it some text, it guesses what comes next.
But computers don't understand text, so everything gets converted to numbers first.

The full journey, simply
"What is 2 + 3?"
       ↓
   TOKENIZATION
       ↓
["What", "is", "2", "+", "3", "?"]
       ↓
   INPUT IDs (vocab lookup)
       ↓
[892, 318, 17, 10, 18, 30]
       ↓
   LLM (neural network)
       ↓
   LOGITS (scores for every possible next token)
[token_0: -2.1, token_1: 0.3, token_42: 8.7, token_99: -0.1, ...]
       ↓
   PICK HIGHEST SCORE → token_42 = "5"
       ↓
   APPEND "5" TO INPUT, REPEAT...

The model repeats this loop, one token at a time, until it generates an EOS token (end of sentence — its way of saying "I'm done").

What are logits?
Just a list of scores, one per token in the vocabulary. If the vocab has 100,000 tokens, you get 100,000 scores. The highest score = the token the model thinks comes next.
vocab size = 100,000 tokens
logits     = [0.1, -2.3, 8.7, 0.02, ...] ← 100,000 numbers
                              ↑
                         this one wins → that's the next token

What is constrained decoding?
Normally the model can pick any token freely. But for function calling, you want the output to always be valid JSON — so you cheat.
Before picking the next token, you look at the logits and set invalid tokens to -infinity so they can never be picked:
You're generating JSON and just wrote: {"function_name": "
                                                          ↑
                               only string tokens are valid here!

logits before constraint:
  token "{" : 5.2   ← INVALID here, set to -inf
  token "add": 3.1  ← valid, keep it
  token "123": 2.8  ← INVALID here, set to -inf
  token "sub": 1.9  ← valid, keep it

logits after constraint:
  token "{" : -inf
  token "add": 3.1  ← wins
  token "123": -inf
  token "sub": 1.9
This guarantees the output is always parseable JSON, no hallucinated structure, no broken brackets.

How your SDK fits in
pythonmodel = Small_LLM_Model()

# Step 1: convert your prompt to token IDs
input_ids = model.encode("What is 2 + 3?").tolist()[0]
# → [892, 318, 17, 10, 18, 30]

# Step 2: get scores for every possible next token
logits = model.get_logits_from_input_ids(input_ids)
# → [0.1, -2.3, 8.7, ...] (one score per vocab token)

# Step 3: pick the best token
next_id = logits.index(max(logits))

# Step 4: add it to input and repeat
input_ids.append(next_id)
For constrained decoding, you'd modify the logits between steps 2 and 3 — zeroing out anything that would break your JSON schema.

Summary in one sentence

The LLM scores every possible next token → you block the ones that would break your JSON → you pick the best remaining one → repeat until done.


Want me to now look at your functions_definition.json to understand what schema you need to enforce?

User question: "what is 2 + 3?"
        ↓
Build a prompt that includes the function definitions
        ↓
LLM generates JSON token by token
        ↓
At each step, constrained decoding blocks invalid tokens
  - wrong function name → blocked
  - string where number expected → blocked
  - broken JSON structure → blocked
        ↓
Output is guaranteed valid JSON → parse it → call the function

# Export bc of space on 42 pc

# Point HuggingFace cache to sgoinfre instead of home
export HF_HOME=/sgoinfre/dbotelho/hf_cache

# Then run again
python3 main.py


What is a prompt made of?
A good prompt for function calling has 4 parts:
1. ROLE        → tell the model what it is
2. TASK        → tell it exactly what to output
3. CONTEXT     → give it the available functions
4. REQUEST     → the actual user question
In our code that maps to:
pythonreturn f"""You are a function calling assistant.       ← ROLE

Given a user request, output ONLY a valid JSON object  ← TASK
with exactly these keys: ...

Available functions:                                   ← CONTEXT
- fn_add_numbers(a: number, b: number)
...

User request: What is the sum of 2 and 3?             ← REQUEST
Output:"""                                            ← triggers generation


generation loop — this is the core of the project.

What it needs to do
1. Take the prompt
2. Encode it to token IDs
3. Get logits from the model
4. (constrained decoding) block invalid tokens
5. Pick the best remaining token
6. Add it to the input
7. Repeat until we have complete valid JSON
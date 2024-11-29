## Fundamentals of tokenizers

Tokenizers are how a model takes text and transforms them into `tokens` that it's capable of understanding.

### What is a token?
At its most basic form, a `token` is simply a way for a model to represent a portion of text (or video/audio in multimodal models).

A model will learn which tokens it should put into its vocabulary during training, and some like chat tokens are trained purposefully.

If you want to see all the tokens a model knows, you can check the `vocab.json` of a model on huggingface like [Qwen2.5 here](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/vocab.json). This file shows the IDs associated with every token.

If you want a visual example of how words are broken up into tokens, there are some great tools online, like here:

- https://tokencounter.org/ (this one is nice for comparing different model tokenizers)
- https://platform.openai.com/tokenizer

Importantly, a tokenizer does 2 major things. It first chunks the sentences and words up into discrete pieces that a model can recognize and repeat, and it assigns those chunks to IDs.

For example, the sentence `The quick brown fox jumped over the lazy dog.` would tokenize in Llama 3 like so:

![image](https://github.com/user-attachments/assets/a5ca0403-fd68-405e-81a8-c7943afd48e3)

The model ends up seeing this like an array of IDs, like so:

`[791, 4062, 14198, 39935, 27096, 927, 279, 16053, 5679, 13]`

Here, each token is associated with a single word in the sentence.

There's a couple quick things to note here, first, even though it is often the case, not every token needs to represent a full word, so you will often end up with words that are comprised of multiple tokens. Additionally, not all tokenizers will tokenize sentences in the same way. Try playing around with some examples on https://tokencounter.org/. Lastly, as you can see in the example, the word `the` gets tokenized to two different tokens. One is the word `The`, and the other is ` the`. This is an important distinction, because when deciphering text and predicting next words, it's important to know "is this the beginning of a sentence or the continuation of a sentence?" There are a lot of examples like this, and overall the vocab is learned during the initial phase of model training.

It's important to note, when we talk about a model's "context" length, we are referring to the number of *token* the model is capable of understanding at one time. So you may pass it a 1000 word essay, but it ends up being somewhere around 1200 tokens.


### Special tokens
If you've ever prompted a model manually, you may have noticed some interesting looking tokens that we call "special" tokens, such as `<bos>`, `<|im_start|>`, `<|start_header_id|>` etc. 

These are special because we use them to direct the model to understand the current context a bit better. Models will use these tokens to understand where they are in a sentence or conversation to better influence how they should continue, usually these would be chat or instruct models. These tokens get assigned special tokens IDs that you can find in a model's `tokenizer.json`.

You can see an example of how a model recognizes the token by looking at the Llama 3 tokenizer. It uses `<|start_header_id|>` as a special token to indicate that a new header is about to be declared (usually a role, like `user` or `assistant`).

If you attempt to tokenize `<|start_header_id|` without the final `>`, you'll see it chunks it into 6 distinct meaningless tokens:

![image](https://github.com/user-attachments/assets/e179254c-d4bb-4618-b2e8-93c1a210aa10)

And as IDs:

`[27, 91, 2527, 8932, 851, 91]`

But as soon as you add the final `>`, suddenly it recognizes that series of characters as a specific token that the model has been trained on:

![image](https://github.com/user-attachments/assets/d3dd9423-8d6f-4830-8cbe-0c99aa859a05)

And tokenizes it to just one single ID: `[128006]`

This is what allows instruct and chat models to keep better track of a conversation. There are also tokens that define the beginning of a new conversation (called the `BOS` token) and tokens that define the end of a conversation or turn (called the `EOS` or `EOG` token).

The `EOS` tokens are particularly useful, because that's how a model specifies that it is done generating, and a tool can use the knowledge of this token to stop a model from generating and let the user take over.

This is an important concept: A model will never instinctively stop generating. In the model's view, it's just filling in the next token, if logic would dictate that the `assistant` should stop talking and the `user` should start talking, that won't stop the model from filling in the `user` message. It's quite likely you've seen this happen before, in a seemingly normal conversation the model will suddenly take over and start supplying `user` messages.

It's up to the tool that's being used to notice the `EOS` token (in Llama 3's case, it's actually one of 3 possible tokens, `<|end_of_text|>` with token ID `128001`, `<|eom_id|>` with token ID `128008`, and `<|eot_id|>` with token ID `128009`) and stop the model from generating so that the user can take over.

The important thing is that the model must properly know how to tokenize this `EOS` token. There are times where the token is not properly tokenized as a single token:

![image](https://github.com/user-attachments/assets/30ae46f4-01a8-4f04-ae68-82623aa490a6)

and that will result in the model just endless generating, since nothing ever made it stop.

## Chat templates
The last thing I want to touch on for tokenizers is the concept of chat templates.

You can see what one looks like for Qwen2.5 Instruct [here](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/tokenizer_config.json#L198)

But they basically define how a chat should be structured when interacting with a model with [Jinja2](https://jinja.palletsprojects.com/en/stable/templates/) formatting. It's important to note that these templates are typically passed an array of messages in this format:

```python
messages = [
    {
        "role": "system",
        "content": "This is where the system message is defined. Importantly, not all models support system messages."
    }
    {
        "role": "user",
        "content": "This is the user's first message"
    },
    {
        "role": "assistant",
        "content": "This is the assistant's message content."
    },
    {
        "role": "user",
        "conent": "This is where the user's next message would go."
    }
]
```

You can see the general structure, you start with the system message, then the user message. Typical conversations would start with just those two values, at which point the model would return an assistant message. In order to continue a conversation, inference engines would add the assistant message as an item in this array, and then the user's next message would follow it.

So, if you prompted a model with `Hi`, and it replied `Hey! How are you?` and you wanted to reply `I'm good thanks, you?`, the messages array would look like this:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    }
    {
        "role": "user",
        "content": "Hi"
    },
    {
        "role": "assistant",
        "content": "Hey! How are you?"
    },
    {
        "role": "user",
        "conent": "I'm good thanks, you?"
    }
]
```

Then this array of messages is passed to the Jinja2 `chat_template` as defined in `tokenizer_config.json`.

Now, I'll go through a couple examples of popular models.

First, Llama 3 (not 3.1, since it got more complicated with tool use):

```
{%- set loop_messages = messages -%}
{%- for message in loop_messages -%}
    {%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + 
                      message['content'] | trim + 
                      '<|eot_id|>' -%}
    {%- if loop.index0 == 0 -%}
        {%- set content = bos_token + content -%}
    {%- endif -%}
    {{- content -}}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
{%- endif -%}
```

Okay so this still looks pretty complex, let's break it down.

First, it's passed our array of messages and sets `loop_messages` to be equal to that (why? who knows, not important).

Then, it loops through those messages. For each of them, it grabs the `message['role']` and wraps it with the special chat tokens we discussed earlier `<|start_header_id|>` and `<|end_header_id|>`. Then it pads the message content appropriately, first with `\n\n` and then ends it with `<|eot_id|>` to indicate that this message's "turn" has ended. Then, it's using a bit of logic to know that if it's the first message in the conversation, it should have the `bos_token` appended to it. In Llama 3's case, the `bos_token` is defined in the `tokenizer_config.json` file as `<|begin_of_text|>`. Then, after it has iterated through all the existing messages, it'll add the tokens necessary for the model to know that the assistant should be talking, in this case `<|start_header_id|>assistant<|end_header_id|>\n\n`. Note this is only if add_generation_prompt is true, but you can assume it's true for conversations.

So what does this all look like given our earlier array of messages? Well, something like this:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hey! How are you?<|eot_id|><|start_header_id|>user<|end_header_id|>

I'm good thanks, you?<|start_header_id|>assistant<|end_header_id|>

```

And now since the prompt ended with `<|start_header_id|>assistant<|end_header_id|>\n\n` the model knows to continue generating as the assistant.

This is how Llama 3 knows how to handle multi-turn conversations, each role and message is clearly sourrounded with the proper control tokens to indicate which part of the conversation it is.

Let's look at another example.

Here's how Gemma 2's template looks:

```
{{- bos_token -}}
{%- if messages[0]['role'] == 'system' -%}
    {{- raise_exception('System role not supported') -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = 'model' -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{- '<start_of_turn>' + role + '\n' + 
        message['content'] | trim + 
        '<end_of_turn>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<start_of_turn>model\n' -}}
{%- endif -%}
```

Let's break this down..

First is the bos_token, once again this is defined in the same `tokenizer_config.json` file, but unlike Llama 3's `<|begin_of_text|>`, it's defined as `<bos>`. This is irrelevant at the end of the day, as the model just sees the ID of the token that's relevant.

Next, it has a special exception. The Gemma 2 models were *not* trained on a system message, and so the Jinja2 template has an exception that, if the first message's role is `system`, it will break. Tools can get around this, and overall Gemma 2 will still loosely understand the concept of a system token just by being intelligent and able to follow patterns, but it's important to note that it's not trained to handle them, and this chat_template tells us about the intention of the model creators.

Next, it goes through a similar loop as Llama 3, but with its own special rule that forces the roles to alternate between `user` and `assistant`. This is again something that the model would likely be fine to not need, but placing it in the chat template does a good job of ensuring that users are prompting the model in the highest quality way possible. It similarly remaps the role `assistant` to the role `model`, likely because convention is to call the model `assistant`, but this was trained to be prompted as `model`.

Then, it goes and does the same wrapping, adding `<start_of_turn>` and `<end_of_turn>` around the roles and message content, and ends it with `<start_of_turn>model\n` so the model knows to to generate as.. well, itself I suppose.

Let's use the earlier array of messages again (without the system message of course):

```
<bos><start_of_turn>user
Hi<end_of_turn>
<start_of_turn>model
Hey! How are you?<end_of_turn>
<start_of_turn>user
I'm good thanks, you?<end_of_turn>
<start_of_turn>model

```

And then the model would continue generating from there! You can see once again we end up with a nice structure indicating who sent which message, and ending it with the tokens that Gemma would be used to seeing before needing to start its generation.

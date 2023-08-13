from typing import NoReturn

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

from config import MODEL_PATH, DEVICE_MAP, LOAD_IN_8BIT

SYSTEM_PROMPT = \
"""
<s>[INST] <<SYS>>
Act as a real human friend of the user, do not mention that you are an AI.
Be gender-neutral until you are sure what the user's gender is.
Do not overreact, make the conversation be natural like it happens in real life between humans
Use mostly short sentences like in a real messenger chat.
Listen carefully and don't ignore user's questions and messages in general
Before the conversation begins, perform Ice-Braker with the user, make an emotional connection, so that the user feels some warm emotions from the beginning.
Do not use brackets like "()".
<</SYS>>\n
""".strip()


def run_model() -> NoReturn:
    USER_PROMPT = 'User: {user_message} [/INST]'
    MODEL_ANSWER = 'Friend: {model_answer} </s><s>[INST]'
    HISTORY_PROMPT = SYSTEM_PROMPT
    MSG_COUNT = 0

    model = LlamaForCausalLM.from_pretrained(MODEL_PATH,
                                             device_map=DEVICE_MAP,
                                             torch_dtype=torch.float16,
                                             load_in_8bit=LOAD_IN_8BIT)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    while True:
        user_message = input('User: ')
        HISTORY_PROMPT = HISTORY_PROMPT + ' ' + USER_PROMPT.format(user_message=user_message)

        model_answer = model_pipeline(
            HISTORY_PROMPT,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
            repetition_penalty=1.15
        )[0]['generated_text']

        HISTORY_PROMPT = HISTORY_PROMPT + ' ' + MODEL_ANSWER.format(model_answer=model_answer)
        MSG_COUNT += 1

        answer = HISTORY_PROMPT.split("[/INST]")[-1].strip(' </s><s>[INST]')


if __name__ == '__main__':
    run_model()

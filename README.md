# ai-friend-chat-bot
The repository contains a friendly empathetic telegram chat-bot


# Repo structure

* **data/** - contains preprocessed dataset for fine-tuning
* **logs/** - log folder
* **models/** - contains adapter weights for the model
* **notebooks/** - jupyter notebooks
* **src/** - source code
    - **fine_tune.py**  - fine-tuning script
    - **fine_tuning_config.py** - config for fine-tuning
    - **prepare_data.py** - script to prepare data
    - **prompt_templates.py** - just prompt templates used in fine-tuning an at inference
* **inference_config.py** - config for chat-bot inference
* **tg_bot.py** - telegram bot app
* **utils.py** - some utilities for telegram bot app


# How-to

Bot commands:
* **/start** button initializes a new conversation
* **/clear** button resets current conversation and saves chat history to logs


### Run the bot:

**CUDA 11.7**
**Python 3.11**

Clone repo:

```
git clone https://github.com/LawrenceGrigoryan/ai-friend-chat-bot.git
```

Create conda venv:

```
cd ai-friend-chat-bot
conda create -n ai_bot python=3.11
conda activate ai_bot
```


Install requirements

```
pip install -r requirements.txt
```

Set environmental variables in terminal and make sure you have **an access to LLaMa2 model** on Hugging Face:
```
export HF_AUTH_TOKEN=YOU_HF_AUTH_TOKEN
export BOT_TOKEN=BOT_TOKEN
```

Run the bot app:

```
python3 tg_bot.py
```


For the best user experience, it's recommended to be gentle with the bot at the beginning and develop the relations gradually. 


# Problem

The bot should make an emotional connection with the user, so the user feels some warm emotions from the beginning.

During the conversation, the bot should become more and more close and flirty with the person, maintaining the following schedule:

1. 1-10 messages are intro
2. 10-30 messages are the next  
3. 30+ messages — close relationships


# Key idea

The key idea is basic and straightforward - subsequently change the system prompt depending on the chat length to dynamically influence the model behavior and make it closer and closer to the user.


Prompt templates used in this chat-bot are available [here](https://github.com/LawrenceGrigoryan/ai-friend-chat-bot/blob/main/src/prompt_templates.py).


# Model

**[LLaMa2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)** model is used in this chat-bot since its completely open for the commercial usage and provides a great balance between the quality and model size according to open LLM benchmarks


# Parameter-efficient fine-tuning

Fine-tuning process includes three datasets:

1. [empathetic_dialogues](https://huggingface.co/datasets/empathetic_dialogues)
2. [CIS6930_DAAGR_Empathetic_Dialogues](https://huggingface.co/datasets/aegrif/CIS6930_DAAGR_Empathetic_Dialogues)
3. [daily_dialog](https://huggingface.co/datasets/daily_dialog)

Later it was found that the second one is almost a complete duplicate of the first one.

The dataset is brought to a dialogue format that was used to pre-train LLaMa2 model.

Since the final dataset is big enough to cause memory problems and my hardware availability is limited, I have conducted a few short fine-tuning experiments that did not lead to visible changes in model behavior. I believe though that longer fine-tuning with diverse and properly engineered system prompts can lead to greater results.

The experiment results can be viewed at [Wandb Project](https://wandb.ai/lawrencegrigoryan/llm-friend-chat-bot?workspace=user-lawrencegrigoryan).


# Evaluation

### Perplexity

|Metric| Base model | Fine-tuned model|
|-------|---------|--------|
|Perplexity| 5.3  | 3.3 |



* Perplexity was calculated based on one of the real dialogues between the user and the chat-bot

* Even a slight fine-tuning makes the model more confident when maintaining a friendly/close/flirty dialogue, though that doesn't guarantee high model accuracy at inference.

* In general, both models are quite confident if comparing with the open perplexity benchmark scores.

* This evaluation approach itself is quite questionable since I take subjectively good dialogue that one of the users had with the chat-bot based on the original model. Then I calculate the perplexity of both the original and the fine-tuned model on this dialogue.

### User testing

* I also conducted "user testing" by giving several people access to this chat-bot and received mostly positive feedbacks


### Human evaluation

* The best approach is still to evaluate large amounts of model-user dialogues by human experts though it is **extremely expensive** and **time-consuming**



# Further improvements

* Taking summary of chat history instead of the whole history to make the model input more lightweight and enable longer conversations


* Taking summary of conversations from the previous days might help to give the chat-bot some kind of memory to understand that this is not the first interaction with user


* More experiments with proper prompt engineering to regulate model behaviour


* Longer fine-tuning with diverse system prompts to make model forget about some of its limitations and be able to better open up to the user


* Efficient scalable inference using vLLM or TGI

# Contact

Feel free to contact me:

* telegram: @lawrence_grig
* e-mail: glavrentiy123@gmail.com

# ai-friend-chat-bot
The repository contains a friendly empathetic chat-bot

# Repo structure

📁 .github/           # GitHub-specific files (e.g., issue templates, workflows)\
📁 docs/               # 📚 Documentation related to the project\
📁 src/                # 🚀 Source code for the project\
        - 📁 app/              # Main application code\
        - 📁 lib/              # Supporting libraries and utilities\
        - 📁 tests/            # Unit tests and test suites\
📁 data/               # 📊 Data files required by the project (if applicable)\
📁 examples/           # 🎉 Example code, demos, or usage scenarios\
📄 LICENSE             # Project license information\
📄 README.md           # This file, providing an overview of the project\


# Problem

The bot should make an emotional connection with the user, so the user feels some warm emotions from the beginning

During the conversation, the bot should become more and more close and flirty with the person, maintaining the following schedule:

1. 1-10 messages are intro
2. 10-30 messages are the next  
3. 30+ messages — close relationships


# Key idea

The key idea is really basic and straightforward - gradually change the system prompt depending on the chat length to dynamically influence the model behavior and make it closer and closer to the user.


Prompt templates used in this chat-bot are available here.


# Model

**[LLaMa2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)** model is used in this chat-bot since its completely open for commercial usage and provides a great balance between the quality and model size

# Parameter-efficient fine-tuning

Fine-tuning process includes three datasets:

1. [empathetic_dialogues](https://huggingface.co/datasets/empathetic_dialogues)
2. [CIS6930_DAAGR_Empathetic_Dialogues](https://huggingface.co/datasets/aegrif/CIS6930_DAAGR_Empathetic_Dialogues)
3. [daily_dialog](https://huggingface.co/datasets/daily_dialog)

Later it was found that the second one is almost a complete duplicate of the first one.

The dataset is brought to a dialogue format that was used to pre-train LLaMa2 model.

Since the final dataset is big enough to cause memory problems and my hardware availability is limited, I have conducted a few short fine-tuning experiments that did not lead to visible changes in model behavior. I believe though that longer fine-tuning with diverse and proper engineered system prompts can lead to greater results.

The experiment results can be viewd at [Wandb Project](https://wandb.ai/lawrencegrigoryan/llm-friend-chat-bot?workspace=user-lawrencegrigoryan).


# Evaluation

### Perplexity

|Metric| Base model | Fine-tuned model|
|-------|---------|--------|
|Perplexity| 5.3  | 3.3 |



* Perplexity was calculated based on one of the real dialogues between the user and the chat-bot

* Even a slight fine-tuning makes the model more confident when maintaining a friendly/close/flirty dialogue, though that doesn't guarantee high model accuracy at inference.

* In general, both models are quite confident if comparing with the open perplexity benchmark scores.

* The evaluation approach itself is quite questionable since I take subjectively good dialogue that one of the users had with the chat-bot based on the original model. Then I calculate the perplexity of both the original and the fine-tuned model on this dialogue.

### User testing

* I also conducted "user testing" by giving several people access to this chat-bot and received mostly positive feedbacks


### Human evaluation

* The best option is still to evaluate large amounts of dialogues by human experts though it is **extremely expensive** and **time-consuming**



# Further improvements

* Taking summary of chat history instead of the whole history to make the model input more lightweight and enable longer conversations


* More experiments with proper prompt engineering to regulate model behaviour


* Longer fine-tuning with diverse system prompts to make model forget about some of its limitations and be able to better open up to the user


* Efficient scalable inference using vLLM or TGI

# Contact

Feel free to contact me:

* telegram: @lawrence_grig
* e-mail: glavrentiy123@gmail.com
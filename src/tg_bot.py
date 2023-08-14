"""
Telegram chat-bot
"""
import logging
import os
import json
from typing import NoReturn
from collections import defaultdict
from pathlib import Path

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    filters,
    MessageHandler,
    ApplicationBuilder,
    CommandHandler,
    CallbackContext
)

from utils import get_model
from inference_config import (
    MODEL_PATH,
    ADAPTER_WEIGHTS_PATH,
    MODEL_LOAD_PARAMS,
    MODEL_INFERENCE_PARAMS
)
from prompt_templates import (
    INIT_SYSTEM_PROMPT,
    CLOSE_SYSTEM_PROMPT,
    FLIRTY_SYSTEM_PROMPT,
    USER_PROMPT,
    MODEL_OUTPUT
)

# Get bot token
bot_token = os.environ["BOT_TOKEN"]

# Create logging structure
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logs_dir = Path("../logs/")

# Load the model pipeline
model_pipeline = get_model(
    MODEL_PATH,
    MODEL_LOAD_PARAMS,
    ADAPTER_WEIGHTS_PATH
)
# Initialize a dictionary to store user conversation history
user_history = defaultdict(dict)


async def start(update: Update, context: CallbackContext):
    """
    Implements /start command (button)
    """
    # Get user ID and set user history to initial state if no user_id found
    user_id = update.effective_user.id
    if not user_history.get(user_id):
        user_history[user_id]["prompt"] = INIT_SYSTEM_PROMPT
        user_history[user_id]["msg_count"] = 0

    # Button
    reply_markup = ReplyKeyboardMarkup([[KeyboardButton("/start")]],
                                       resize_keyboard=True)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm your AI-Friend. Let's have a chat!",
        reply_markup=reply_markup
        )


async def respond(update: Update, context: CallbackContext):
    """
    Implements a response of the LLM to user message
    """
    user_id = update.effective_user.id

    # Buttons
    reply_markup = ReplyKeyboardMarkup([[KeyboardButton("/start"),
                                         KeyboardButton("/clear")]],
                                       resize_keyboard=True)

    # Conversation might need a restart if the bot was restarted
    try:
        # Add user prompt to the system prompt
        user_prompt = user_history[user_id]["prompt"] + \
            " " + \
            USER_PROMPT.format(user_message=update.message.text)
    except KeyError:
        # If no user found, send message
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please, restart the conversation using /start command",
            reply_markup=reply_markup
            )

    # Parse model output
    model_output = model_pipeline(
        user_prompt,
        **MODEL_INFERENCE_PARAMS
    )[0]["generated_text"]
    response = model_output.split("[/INST]")[-1].strip()

    # Add model output to current
    user_prompt = user_prompt + \
        " " + \
        MODEL_OUTPUT.format(model_output=response)

    # Update user initial prompt to cover all (or almost all) the previous chat history
    user_history[user_id]["prompt"] = user_prompt
    user_history[user_id]["msg_count"] += 2

    # Replace the system prompt with the next one
    if 10 < user_history[user_id]["msg_count"] <= 30:
        user_history[user_id]["prompt"] = user_history[user_id]["prompt"].replace(INIT_SYSTEM_PROMPT,
                                                                                  CLOSE_SYSTEM_PROMPT)
    elif user_history[user_id]["msg_count"] > 30:
        user_history[user_id]["prompt"] = user_history[user_id]["prompt"].replace(INIT_SYSTEM_PROMPT,
                                                                                  FLIRTY_SYSTEM_PROMPT)

    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=response,
                                   reply_markup=reply_markup)


async def clear(update: Update, context: CallbackContext):
    """
    Clears the conversation history between the bot and the user
    Implements /clear command (button)
    """
    user_id = update.effective_user.id

    # Save user history
    with open(logs_dir.joinpath(f"user_history_{user_id}.json"), "w") as fp:
        json.dump(user_history, fp)

    # Reset user history
    user_history[user_id]["prompt"] = INIT_SYSTEM_PROMPT
    user_history[user_id]["msg_count"] = 0

    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Conversation history is clear now. \
                                         Feel free to start a new one!")


def run_bot() -> NoReturn:
    application = ApplicationBuilder().token(bot_token).build()

    start_handler = CommandHandler("start", start)
    clear_handler = CommandHandler("clear", clear)
    respond_handler = MessageHandler(filters.TEXT & (~filters.COMMAND),
                                     respond)

    application.add_handler(start_handler)
    application.add_handler(clear_handler)
    application.add_handler(respond_handler)

    application.run_polling()


if __name__ == "__main__":
    run_bot()

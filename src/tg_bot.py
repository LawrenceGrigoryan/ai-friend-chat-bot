"""
Telegram bot
"""
import logging
import os
from typing import NoReturn

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    filters,
    MessageHandler,
    ApplicationBuilder,
    CommandHandler,
    CallbackContext
)

from utils import get_model
from config import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    MODEL_OUTPUT,
    MODEL_PARAMS
)

# Get bot token
bot_token = os.environ['BOT_TOKEN']
# Create logging structure
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load the model
model_pipeline = get_model()
# Initialize a dictionary to store user conversation history
user_history = {}


async def start(update: Update, context: CallbackContext):
    """
    Implements /start command
    """
    user_id = update.effective_user.id
    if not user_history.get(user_id):
        user_history[user_id] = SYSTEM_PROMPT
    reply_markup = ReplyKeyboardMarkup([[KeyboardButton('/start')]],
                                       resize_keyboard=True)
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="I'm your AI-Friend. Let's have a chat!",
                                   reply_markup=reply_markup)


async def respond(update: Update, context: CallbackContext):
    """
    Implements a response of the LLM to user message
    """
    user_id = update.effective_user.id
    user_prompt = user_history[user_id] + ' ' + USER_PROMPT.format(user_message=update.message.text)
    model_output = model_pipeline(
        user_prompt,
        **MODEL_PARAMS
    )[0]['generated_text']
    user_prompt = user_prompt + ' ' + MODEL_OUTPUT.format(model_output=model_output)
    user_history[user_id] = user_prompt
    answer = user_prompt.split("[/INST]")[-1].strip(' </s><s>[INST]')
    reply_markup = ReplyKeyboardMarkup([[KeyboardButton('/start'),
                                         KeyboardButton('/clear')]],
                                       resize_keyboard=True)
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=answer,
                                   reply_markup=reply_markup)


async def clear(update: Update, context: CallbackContext):
    """
    Clears the conversation history between the bot and the user
    Implements /clear command (button)
    """
    user_id = update.effective_user.id
    user_history[user_id] = SYSTEM_PROMPT
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Conversation history is clear now. Feel free to start a new one!")


def run_bot() -> NoReturn:
    application = ApplicationBuilder().token(bot_token).build()

    start_handler = CommandHandler('start', start)
    clear_handler = CommandHandler('clear', clear)
    respond_handler = MessageHandler(filters.TEXT & (~filters.COMMAND),
                                     respond)

    application.add_handler(start_handler)
    application.add_handler(clear_handler)
    application.add_handler(respond_handler)

    application.run_polling()


if __name__ == '__main__':
    run_bot()

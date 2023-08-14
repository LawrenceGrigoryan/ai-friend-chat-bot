FROM python:3.11.4-slim
WORKDIR /app
ENV BOT_TOKEN=BOT_TOKEN
COPY . /app/
RUN pip install -r requirements.txt
CMD ["python3", "tg_bot.py"]
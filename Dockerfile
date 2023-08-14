FROM python:3.11.4-slim
WORKDIR /app
ENV BOT_TOKEN=6678539224:AAEOBX2hOBkIaSU9Y7tF6fi5Uj831KeLPbM
COPY logs/* /app/logs/
COPY src/tg_bot.py /app/
COPY src/prompt_templates.py /app/
COPY src/utils.py /app/
COPY .env /app/
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["python3", "tg_bot.py"]
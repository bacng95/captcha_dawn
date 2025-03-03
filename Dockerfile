FROM python:3.8

WORKDIR /

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# RUN pip install --upgrade pip

WORKDIR /captchaResolver

RUN pip install -r requirements.txt

CMD ["python", "server.py"]
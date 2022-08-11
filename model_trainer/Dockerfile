From ubuntu:20.04

WORKDIR /home
COPY code/* ./

RUN  apt update && apt install python3-pip -y --no-install-recommends && pip install -r requirements.txt

ENTRYPOINT ["python3", "train.py"]

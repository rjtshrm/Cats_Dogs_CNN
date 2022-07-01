From ubuntu:20.04

RUN  apt-get update  && apt install python3-pip -y

WORKDIR /home
COPY code/* ./

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "gen.py"]

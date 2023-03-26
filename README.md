# Intructions for freshly installed Ubuntu 22.04 LTS

## Start with updating packages
```console
$ sudo apt update && sudo apt upgrade -y
$ sudo apt-get install ffmpeg
```

<br>

## Python setup
**Instalation**
```console
$ sudo apt install python3.10
# verify python version
$ python3.10 --version
```

**Venv setup**
```console
$ cd src/
$ python3.10 -m pip install virtualenv
$ python3.10 -m venv <your_venv_name>
```

**Activate your venv**
```console
$ source <your_venv_name>/bin/activate
```

**Install requirements**
```console
$ pip install -r requirements.txt
```

<br>

## Kafka setup
**Installation**
```console
$ cd /opt/
$ sudo apt install default-jdk
$ sudo wget https://dlcdn.apache.org/kafka/3.4.0/kafka_2.13-3.4.0.tgz
$ sudo tar -xvzf kafka_2.12-3.4.0.tgz

$ cd kafka_2.12-3.4.0/config
$ sudoedit server.properties
# find advertised.listeners
# uncomment line
# change to advertised.listeners=PLAINTEXT://localhost:9092
# ctrl+x -> y -> enter
```

**Start Zookeper**
```console
$ cd ..
$ sudo ./kafka-commands/start-zookeeper.sh
```

**Start Kafka**
```console
$ sudo ./kafka-commands/start-kafka.sh
```

**Stop Kafka Server and Zookeeper**
```console 
$ sudo ./kafka-commands/stop-kafka-and-zookeeper.sh
```

**Create topics for requests and results**
```console
$ sudo ./kafka-commands/create-topics.sh
```

<br>

## Run the application

**Start the consumer**
```console
(venv)$ cd src/
(venv)$ python3 consumer.py
```

**Start the producer**
```console
(venv)$ uvicorn producer:app --reload 
```

**Test application**
```console
$ python3 test_server.py
```


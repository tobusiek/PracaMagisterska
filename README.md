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
$ python3.10 --version  # verify python version
```

**Venv setup**
```console
$ cd src/
$ python3.10 -m pip install virtualenv
$ python3.10 -m venv <your_venv_name>
```

*If it doesn't work try:*
```console
$ sudo apt install python3-pip
$ sudo apt install python3-venv
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
$ sudo wget https://dlcdn.apache.org/kafka/3.4.0/kafka_2.12-3.4.0.tgz
$ sudo tar -xvzf kafka_2.12-3.4.0.tgz
```

**Configuration**
```console
$ cd kafka_2.12-3.4.0/config
$ sudo nano server.properties
```

1. Find *advertised.listeners*
2. Uncomment line
3. Change to *advertised.listeners=PLAINTEXT://localhost:9092*
4. ctrl+x -> y -> enter

```console
$ sudo nano producer.properties
```

5. Find *max.request.size*
6. Uncomment line
7. Change to *max.request.size=2097152*
8. ctrl+x -> y -> enter

</br>

**Make Kafka startup quicker**
```console
$ cd <project-root>/kafka-commands/
$ chmod a+x chmod-kafka.sh
$ sudo ./chmod-kafka.sh
```

**Start Zookeper**
```console
$ sudo ./start-zookeeper.sh
```

**Start Kafka**
```console
$ sudo ./start-kafka.sh
```

**Stop Kafka Server and Zookeeper if needed**
```console 
$ sudo ./stop-kafka-and-zookeeper.sh
```

**Create topics for requests and results**
```console
$ sudo ./create-topics.sh
```

<br>

## Run the application

**Start the consumer**
```console
(venv) $ cd <project-root>/predictions_server/
(venv) $ python main.py
```

**Start the producer**
```console
(venv) $ cd  <project-root>/api_server/
(venv) $ python main.py
```

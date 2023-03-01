## Python setup
**Instalation**

    $ sudo apt update && sudo apt upgrade -y
    $ sudo apt install python3.10
    # verify python version
    $ python3.10 --version


**Venv setup**

    $ python3.10 -m pip install virtualenv
    $ python3.10 -m venv <your_venv_name>


**Activate your venv**

    $ source <your_venv_name>/bin/activate


**Install requirements**

    $ pip install -r requirements.txt


<br>

## Kafka setup
**Installation**

    $ cd /opt/
    $ sudo wget https://dlcdn.apache.org/kafka/3.4.0/kafka_2.13-3.4.0.tgz
    $ sudo tar -xvzf kafka_2.12-3.4.0.tgz


    $ cd kafka_2.12-3.4.0/config
    $ sudoedit server.properties
    # find advertised.listeners
    # uncomment line
    # change to advertised.listeners=PLAINTEXT://localhost:9092
    # ctrl+x -> y -> enter


**Start Zookeper**
    
    $ sudo ./kafka-commands/start-zookeeper.sh


**Start Kafka**

    $ sudo ./kafka-commands/start-kafka.sh


**Stop Kafka Server and Zookeeper**
    
    $ sudo ./kafka-commands/stop-kafka-and-zookeeper.sh


**Create topics for requests and results**

    $ sudo ./kafka-commands/create-topics.sh


<br>

## Run the application

**Start the consumer**

    (venv) $ python3 consumer.py


**Start the producer**

    (venv) $ python3 producer.py


**Test application**

    $ python3 test_server.py



#!/bin/bash

cd /opt/kafka_2.12-3.4.0

./bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 \
    --create --topic "requests_topic" \
    --partitions 1 \
    --replication-factor 1 \
    --config max.message.bytes=2097152


./bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 \
    --create --topic "results_topic" \
    --partitions 1 \
    --replication-factor 1
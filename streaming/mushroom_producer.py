"""Envoie des champignons du CSV dans le topic Kafka `mushrooms` (un toutes les 0,5 s)."""
import csv
import json
import random
import sys
import time
import uuid

from kafka import KafkaProducer

CSV = "/work/data/mushrooms.csv"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
producer = KafkaProducer(bootstrap_servers=["kafka1:19092", "kafka2:19093", "kafka3:19094"])

for row in random.sample(rows, N):
    row["id"] = str(uuid.uuid4())  # clé primaire Cassandra ; "class" = vraie classe, gardée pour comparer
    producer.send("mushrooms", json.dumps(row).encode("utf-8"))
    time.sleep(0.5)
producer.flush()
print(f"{N} champignons envoyés")

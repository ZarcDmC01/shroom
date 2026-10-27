"""Spark Structured Streaming : lit `mushrooms` (Kafka), applique le modèle sauvegardé, écrit dans Cassandra."""
import json

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

meta = json.load(open("/work/models/metrics.json"))
fields = ["id", "class"] + meta["features"]
schema = StructType([StructField(f, StringType()) for f in fields])

spark = (SparkSession.builder.appName("MushroomStreaming")
         .config("spark.cassandra.connection.host", "cassandra")
         .config("spark.cassandra.auth.username", "cassandra")
         .config("spark.cassandra.auth.password", "cassandra")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
model = PipelineModel.load("/work/models/best_model")

raw = (spark.readStream.format("kafka")
       .option("kafka.bootstrap.servers", "kafka1:19092,kafka2:19093,kafka3:19094")
       .option("subscribe", "mushrooms").option("startingOffsets", "earliest").load())
parsed = raw.select(F.from_json(F.col("value").cast("string"), schema).alias("d")).select("d.*")

scored = (model.transform(parsed)
          .select("id", F.col("class").alias("real_class"),
                  F.when(F.col("prediction") == 1.0, "p").otherwise("e").alias("predicted_class"),
                  vector_to_array("probability")[1].alias("proba_toxique"), "odor", "cap_color", "habitat"))

query = (scored.writeStream.format("org.apache.spark.sql.cassandra").outputMode("append")
         .option("checkpointLocation", "/tmp/checkpoint_mushrooms")
         .options(table="mushroom_predictions", keyspace="spark_streaming").start())
query.awaitTermination()

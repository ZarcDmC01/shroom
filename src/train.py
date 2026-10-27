"""Entraîne, compare et sauvegarde le meilleur classifieur champignons (PySpark ML)."""
import json
import os
import sys
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mushrooms.csv"
MODEL_DIR = ROOT / "models" / "best_model"
REPORT = ROOT / "models" / "metrics.json"

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
spark = (SparkSession.builder.master("local[*]").appName("mushrooms")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# 1. Exploration
df = spark.read.csv(str(DATA), header=True, inferSchema=True)
print(f"Lignes: {df.count()}  Colonnes: {len(df.columns)}")
df.groupBy("class").count().show()
df.createOrReplaceTempView("mushrooms")  # C2 : requête SQL
spark.sql("SELECT odor, class, COUNT(*) n FROM mushrooms GROUP BY odor, class ORDER BY odor").show(30)
print("Valeurs '?' par colonne:",
      {c: n for c in df.columns if (n := df.filter(F.col(c) == "?").count())})

# 2. Pipeline de préparation (la colonne cible: p = toxique, e = comestible)
features = [c for c in df.columns if c != "class"]
# veil_type est constante -> inutile
features = [c for c in features if df.select(c).distinct().count() > 1]
label_idx = StringIndexer(inputCol="class", outputCol="label", stringOrderType="alphabetAsc")  # e=0, p=1
idx = [StringIndexer(inputCol=c, outputCol=f"{c}_i") for c in features]
enc = OneHotEncoder(inputCols=[f"{c}_i" for c in features], outputCols=[f"{c}_v" for c in features])
asm = VectorAssembler(inputCols=[f"{c}_v" for c in features], outputCol="features")
prep = [label_idx, *idx, enc, asm]

train, test = df.randomSplit([0.8, 0.2], seed=42)

# 3. Modèles + validation croisée
lr, dt, rf = (LogisticRegression(maxIter=50), DecisionTreeClassifier(seed=42),
              RandomForestClassifier(seed=42))
candidates = {
    "LogisticRegression": (lr, ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.01, 0.1]).build()),
    "DecisionTree": (dt, ParamGridBuilder().addGrid(dt.maxDepth, [3, 5, 8]).build()),
    "RandomForest": (rf, ParamGridBuilder().addGrid(rf.numTrees, [20, 50]).addGrid(rf.maxDepth, [5, 8]).build()),
}
f1 = MulticlassClassificationEvaluator(metricName="f1")
acc = MulticlassClassificationEvaluator(metricName="accuracy")
rec_tox = MulticlassClassificationEvaluator(metricName="recallByLabel", metricLabel=1.0)  # rappel « toxique »

results, models = {}, {}
for name, (est, grid) in candidates.items():
    cv = CrossValidator(estimator=Pipeline(stages=[*prep, est]), estimatorParamMaps=grid,
                        evaluator=f1, numFolds=3, seed=42, parallelism=2)
    model = cv.fit(train)
    pred = model.transform(test)
    cm = {f"{r['class']}->{int(r['prediction'])}": r["count"]
          for r in pred.groupBy("class", "prediction").count().collect()}
    results[name] = {"accuracy": acc.evaluate(pred), "f1": f1.evaluate(pred),
                     "recall_toxique": rec_tox.evaluate(pred), "cv_f1": max(model.avgMetrics),
                     "confusion(class->pred, e=0 p=1)": cm}
    models[name] = model
    print(name, results[name])

# 4. Meilleur = meilleur rappel « toxique », puis F1 (une erreur toxique->comestible est la plus grave)
best = max(results, key=lambda k: (results[k]["recall_toxique"], results[k]["f1"]))
print("Meilleur modèle:", best)
MODEL_DIR.parent.mkdir(exist_ok=True)
models[best].bestModel.write().overwrite().save(str(MODEL_DIR))
REPORT.write_text(json.dumps({"best": best, "features": features, "results": results}, indent=2))
spark.stop()

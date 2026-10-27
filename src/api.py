"""API REST : prédit si un champignon est comestible ou toxique (modèle PySpark sauvegardé)."""
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)


def load_schema(path: Path) -> dict[str, dict[str, str]]:
    """Lit data/about_file.md -> {variable_underscore: {mot: code}} (ex. odor: {'none': 'n'})."""
    schema = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, values = line.partition(":")
        if not sep or "=" not in values or name.startswith(("About", "Attribute")):
            continue
        pairs = (p.split("=", 1) for p in values.split(",") if "=" in p)
        schema[name.strip().replace("-", "_")] = {w.strip().lower(): c.strip() for w, c in pairs}
    return schema


SCHEMA = load_schema(ROOT / "data" / "about_file.md")
meta = json.loads((ROOT / "models" / "metrics.json").read_text())
FEATURES = meta["features"]

spark = (SparkSession.builder.master("local[1]").appName("mushrooms-api")
         .config("spark.ui.enabled", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
model = PipelineModel.load(str(ROOT / "models" / "best_model"))

app = FastAPI(title="Mushroom classifier", version="1.1")


class Mushroom(BaseModel):
    """Valeurs en mots (odor='none') ou en codes UCI (odor='n'). Tirets ou underscores acceptés."""
    features: dict[str, str]


def to_code(feature: str, value: str) -> str:
    """Traduit un mot ou un code en code UCI, ou lève une erreur 422 si la valeur est inconnue."""
    options = SCHEMA[feature]
    v = value.strip().lower()
    if v in options:
        return options[v]
    if v in {c.lower() for c in options.values()}:
        return next(c for c in options.values() if c.lower() == v)
    raise HTTPException(422, f"{feature}: '{value}' inconnu. Valeurs possibles: {sorted(options)}")


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["best"]}


@app.get("/features")
def features():
    """Variables attendues et valeurs possibles (mot -> code)."""
    return {f: SCHEMA[f] for f in FEATURES}


@app.post("/predict")
def predict(m: Mushroom):
    given = {k.replace("-", "_"): v for k, v in m.features.items()}
    missing = [f for f in FEATURES if f not in given]
    if missing:
        raise HTTPException(422, f"Champs manquants: {missing}")
    row = {f: to_code(f, given[f]) for f in FEATURES}
    row["class"] = "e"  # valeur factice : la colonne est requise par le pipeline mais pas utilisée pour prédire
    r = model.transform(spark.createDataFrame([row])).select("prediction", "probability").first()
    label = "toxique" if r["prediction"] == 1.0 else "comestible"
    return {"prediction": label, "proba_toxique": float(r["probability"][1])}

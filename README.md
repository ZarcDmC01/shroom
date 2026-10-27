# Classification des champignons avec PySpark

Jeu de données : UCI Mushroom (8 124 lignes, 22 variables catégorielles, cible `class` : e = comestible, p = toxique).

## Structure
- `data/mushrooms.csv` : données avec en-têtes
- `src/train.py` : exploration (DataFrame + SQL), pipeline ML, comparaison de 3 modèles par validation croisée, sauvegarde
- `src/api.py` : API REST FastAPI qui charge le modèle sauvegardé
- `models/` : meilleur modèle (`best_model`) et `metrics.json`

## Lancer (Windows)
Prérequis : Java 17/21, Python 3.11, et `winutils.exe` + `hadoop.dll` dans `hadoop/bin` (variable `HADOOP_HOME`).

```powershell
pip install -r requirements.txt
$env:HADOOP_HOME="$PWD\hadoop"; $env:PATH="$env:HADOOP_HOME\bin;$env:PATH"
python src/train.py
uvicorn src.api:app --port 8000
```

Endpoints : `GET /health`, `GET /features` (variables et valeurs possibles), `POST /predict` avec `{"features": {"odor": "pungent", "cap-color": "brown", ...}}` (les 21 variables utilisées). Les valeurs s'écrivent en mots ou en codes UCI (`"odor": "p"`), tirets ou underscores acceptés. Les correspondances viennent de `data/about_file.md`. Une valeur inconnue ou un champ manquant renvoie une erreur 422.

## Résultats (test = 20 %, seed 42)
| Modèle | Accuracy | F1 | Rappel « toxique » |
|---|---|---|---|
| Régression logistique | 1.00 | 1.00 | 1.00 |
| Arbre de décision | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 |

Les trois modèles sont à égalité parfaite sur le jeu de test (788 comestibles et 763 toxiques tous bien classés). Ce jeu est quasi séparable (ex. `odor` seul discrimine presque toutes les classes), donc ces scores ne sont pas surprenants, mais ils ne permettent pas de les départager. Critère de choix : rappel « toxique » (l'erreur grave est de classer un champignon toxique comme comestible), puis F1. En cas d'égalité, `max` retient le premier modèle : la régression logistique, plus simple.

## Limites
- Un score de 100 % sur un seul découpage ne garantit pas la même performance sur de vrais champignons : ne pas utiliser pour décider de manger un champignon.
- `stalk_root` contient 2 480 valeurs `?` traitées comme une catégorie.
- Workshop 2 (Docker/Kafka/Airflow/Cassandra) non réalisé.

## Veille (C6)
Sources : documentation Spark ML Tuning, tutoriel Spark By Examples, article Towards AI sur MLlib. Pistes : Spark 4.0 est sorti, validation par jeu externe, explicabilité (importance des variables).

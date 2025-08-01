# Databricks notebook source
# MAGIC %md
# MAGIC # Product Classification & GenAI Evaluation
# MAGIC
# MAGIC This notebook shows how to classify product descriptions into four categories—**clothing**, **shoes**, **accessories**, and **furniture**—with Databricks **AI functions**, and then evaluate and track results in mlflow using both traditional ML metrics and the new `` scorers.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## End‑to‑end workflow
# MAGIC
# MAGIC | Step | What happens                                                                                                     | Key tooling                                          |
# MAGIC | ---- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
# MAGIC | 1    | **Setup** – install required libraries and restart the Python process.                                           | `%pip install mlflow databricks-agents scikit-learn` |
# MAGIC | 2    | **Classification** – call `ai_classify()` in Spark SQL to label each description.                                | Databricks SQL AI functions                          |
# MAGIC | 3    | **SQL evaluation** – compute accuracy / precision / recall / F1 directly in SQL.                                 | Spark SQL                                            |
# MAGIC | 4    | **MLflow evaluation** – log a scikit‑learn report plus predictions.                                              | `mlflow.start_run()`                                 |
# MAGIC | 5    | **GenAI evaluation** – build an *eval dataset* and run scorers like `Correctness`, `RelevanceToQuery`, `Safety`. | `mlflow.genai.evaluate()`                            |
# MAGIC

# COMMAND ----------

# MAGIC %pip install mlflow databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Classification and evaluation - SQL approach

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW product_predictions AS
# MAGIC SELECT
# MAGIC   description,
# MAGIC   true_label,
# MAGIC   ai_classify(description, ARRAY('clothing', 'shoes', 'accessories', 'furniture')) AS predicted_label
# MAGIC FROM (
# MAGIC   VALUES
# MAGIC     ('Red cotton t-shirt with short sleeves', 'clothing'),
# MAGIC     ('Leather ankle boots for women', 'shoes'),
# MAGIC     ('Gold-plated hoop earrings', 'accessories'),
# MAGIC     ('Wooden dining table with 6 chairs', 'furniture'),
# MAGIC     ('Men’s running sneakers with breathable mesh', 'shoes'),
# MAGIC     ('Silk neck scarf with floral print', 'accessories'),
# MAGIC     ('Adjustable office chair with lumbar support', 'furniture'),
# MAGIC     ('Canvas backpack with multiple compartments', 'accessories'),
# MAGIC     ('Designer sunglasses with UV protection', 'accessories'),
# MAGIC     ('Velvet sofa with reclining seats', 'furniture')
# MAGIC ) AS t(description, true_label);
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*) AS total,
# MAGIC   SUM(CASE WHEN true_label = predicted_label THEN 1 ELSE 0 END) AS correct,
# MAGIC   ROUND(SUM(CASE WHEN true_label = predicted_label THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS accuracy
# MAGIC FROM product_predictions;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Precision, Recall, F1 for each class
# MAGIC SELECT
# MAGIC   true_label AS label,
# MAGIC   SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END) AS tp,
# MAGIC   SUM(CASE WHEN predicted_label != true_label THEN 1 ELSE 0 END) AS fn,
# MAGIC   SUM(CASE WHEN predicted_label = true_label THEN 0 ELSE 1 END) AS fp,
# MAGIC   ROUND(SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0), 3) AS precision,
# MAGIC   ROUND(SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END) * 1.0 / 
# MAGIC         NULLIF(SUM(CASE WHEN true_label = true_label THEN 1 ELSE 0 END), 0), 3) AS recall
# MAGIC FROM product_predictions
# MAGIC GROUP BY true_label;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Classification and evaluation - Classic ML mlflow approach

# COMMAND ----------

from pyspark.sql.functions import expr

data = [
    ("Red cotton t-shirt with short sleeves", "clothing"),
    ("Leather ankle boots for women", "shoes"),
    ("Gold-plated hoop earrings", "accessories"),
    ("Wooden dining table with 6 chairs", "furniture"),
    ("Men’s running sneakers with breathable mesh", "shoes"),
    ("Silk neck scarf with floral print", "accessories"),
    ("Adjustable office chair with lumbar support", "furniture"),
    ("Canvas backpack with multiple compartments", "accessories"),
    ("Designer sunglasses with UV protection", "accessories"),
    ("Velvet sofa with reclining seats", "furniture")
]

columns = ["description", "true_label"]
df = spark.createDataFrame(data, columns)


# COMMAND ----------

df_predicted = df.withColumn(
    "predicted_label",
    expr("ai_classify(description, array('clothing', 'shoes', 'accessories', 'furniture'))")
)

display(df_predicted)

# COMMAND ----------

df_pd = df_predicted.toPandas()

# COMMAND ----------

import mlflow
from sklearn.metrics import classification_report

mlflow.set_registry_uri("databricks")
mlflow.set_experiment("/Users/kyra.wulffert@databricks.com/product_classifier_eval")

with mlflow.start_run(run_name="ai_classify_eval"):
    y_true = df_pd["true_label"]
    y_pred = df_pd["predicted_label"]

    # Log metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                mlflow.log_metric(f"{label}_{metric}", value)
        else:  # accuracy
            mlflow.log_metric("accuracy", scores)

    # Log predictions table
    predictions_path = "/tmp/predictions.csv"
    df_pd.to_csv(predictions_path, index=False)
    mlflow.log_artifact(predictions_path)

    print("MLflow run logged with classification metrics.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use mlflow.genai.evaluate()

# COMMAND ----------

#  Curate the evaluation dataset to the expected format

# COMMAND ----------

import mlflow

request = """Classify content into one of the following: 'clothing', 'shoes', 'accessories', 'furniture'"""
eval_dataset = [
    {
        "inputs": {"query": f"{request} Content: {row['description']}"},
        "expectations": {"expected_facts": [row["true_label"]]},
    }
    for _, row in df_pd.iterrows()
]

print(eval_dataset[:2])


# COMMAND ----------

type(df_pd)

# COMMAND ----------

import pandas as pd

eval_dataset = pd.DataFrame({
    "inputs": df_pd.apply(
            lambda row: {
                "query": f"{request}  Content: {row['description']}"
            },
            axis=1,
        ),
    "outputs": df_pd["predicted_label"],               
    "expectations": df_pd["true_label"].apply(
            lambda y: {"expected_facts": [y]}
        ),
})
eval_dataset.head().T

# COMMAND ----------

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[Correctness(), RelevanceToQuery(), Safety()],
)

# COMMAND ----------


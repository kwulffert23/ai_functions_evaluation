# Databricks notebook source
# MAGIC %pip install mlflow databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

df = spark.sql("""
WITH summarized_data AS (
  SELECT
    content,
    ai_summarize(content, 50) AS summary,
    ground_truth
  FROM (
    VALUES
      ('Databricks is a cloud-based platform that provides collaborative environments for data engineering, machine learning, and analytics, built around Apache Spark.', 
       'Databricks is a collaborative cloud platform built on Apache Spark for data science and engineering.'),
       
      ('MLflow is an open-source platform for managing the ML lifecycle. It enables experiment tracking, reproducibility, and model deployment.', 
       'MLflow is used for managing experiments, models, and deployment in ML workflows.'),
       
      ('The company reported a 20% increase in revenue for Q2, citing strong performance in the European market and successful product launches.', 
       'Q2 revenue increased by 20% because of growth in Asia and reduced costs.'),
       
      ('Apache Spark is a unified analytics engine known for its speed and ease of use in big data processing, supporting multiple languages like Python and Scala.', 
       'Apache Spark is a fast analytics engine supporting Python and Scala for big data processing.'),
       
      ('Delta Lake brings ACID transactions to big data workloads, ensuring reliability and consistency. It is built on top of open-source Apache Spark.', 
       'Delta Lake introduces ACID guarantees to big data pipelines for better reliability.')
  ) AS t(content, ground_truth)
)
SELECT * FROM summarized_data
""").toPandas()


# COMMAND ----------

# Change needed so that the columns match the schema expected by the databricks-agent evaluator
df["request"] = "Summarize the following content"
# df["retrieved_context"] = df["content"].apply(lambda x: [{"content": x}])

df["retrieved_context"] = [
    # Context matches summary and content
    [{"doc_uri": "doc1.txt", "content": "Databricks is a cloud-based platform that integrates Apache Spark for collaborative analytics."}],
    
    # Context unrelated — mentions Apache Kafka instead of MLflow
    [{"doc_uri": "doc2.txt", "content": "Apache Kafka is a distributed event streaming platform used for building real-time data pipelines."}],
    
    # Context talks about 2020 performance, not Q2 or regions
    [{"doc_uri": "doc3.txt", "content": "The company saw a 12% revenue increase in 2020 due to cost-saving measures."}],
    
    # Context is generic and omits language or speed details
    [{"doc_uri": "doc4.txt", "content": "Spark is a popular open-source data tool used in many organizations."}],
    
    # Context matches summary and content
    [{"doc_uri": "doc5.txt", "content": "Delta Lake brings ACID transactions to big data workloads, ensuring reliability and consistency. It is built on top of open-source Apache Spark."}]
]

df.rename(columns={"summary": "response"}, inplace=True)

# COMMAND ----------

import pandas as pd
pd.set_option('display.max_colwidth', None)
df.T

# COMMAND ----------

import mlflow

mlflow.evaluate(
    data=df,
    model_type="databricks-agent",  
    evaluators=["databricks-agent"]  
)


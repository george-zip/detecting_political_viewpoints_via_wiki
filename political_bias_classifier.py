# Databricks notebook source
import pyspark
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
pd.set_option("max_colwidth", 800)

# COMMAND ----------

import pickle
import os

dir = "/dbfs/FileStore/tables"
all_sentences = []
for filename in os.listdir(dir):
    if filename[-3:] == "pkl":
      with open(os.path.join(dir, filename), "rb") as f:
          tmp = pickle.load(f)
          all_sentences.extend(tmp)

rational_rdd = spark.sparkContext.parallelize(all_sentences, 100)

# COMMAND ----------

rational_rdd = rational_rdd.map(lambda x: ("rational", x))
rational_rdd.count()

# COMMAND ----------

from pyspark.sql.types import StructField, StructType, StringType

rational_schema = StructType([
    StructField("source", StringType()),
    StructField("sample", StringType()),
])

rational_df = spark.createDataFrame(rational_rdd, rational_schema)
rational_df = rational_df.dropDuplicates()
rational_df.count()
# rational_df.sample(True, 0.00001).toPandas()

# COMMAND ----------



# COMMAND ----------

metapedia_schema = StructType([
    StructField("id", StringType()),
    StructField("source", StringType()),
    StructField("sample", StringType()),
])

metapedia_df = spark.read.schema(metapedia_schema)\
  .option("recursiveFileLookup", "true")\
  .option("header", "true")\
  .csv("s3://wallerstein-wikipedia-bias/metapedia/")
metapedia_df = metapedia_df.dropDuplicates()
metapedia_df.count()

# COMMAND ----------

# create union that is not skewed
ratio = metapedia_df.count() / rational_df.count()
all_df = rational_df.sample(True, ratio).union(metapedia_df.select("source", "sample"))

# COMMAND ----------

# clean-ups
from pyspark.sql.functions import regexp_replace

all_df = all_df.withColumn('sample_clean', regexp_replace("sample", "(https?):\/\/(www\.)?[a-z0-9\.:].*?(?=\s)", ""))

# COMMAND ----------

# tokenize
from pyspark.ml.feature import RegexTokenizer, Tokenizer

regex_tokenizer = RegexTokenizer(inputCol="sample_clean", outputCol="words", gaps=False, pattern="[a-zA-Z]+")
tokenized = regex_tokenizer.transform(all_df)
tokenized.sample(True, 0.00001).toPandas()

# COMMAND ----------

# term frequency vectors
from pyspark.ml.feature import CountVectorizer

cv = CountVectorizer(inputCol="words", outputCol="features")
cv_model = cv.fit(tokenized)
wiki_data = cv_model.transform(tokenized)

# COMMAND ----------

from random import choices
choices(cv_model.vocabulary, k=10)

# COMMAND ----------

# transform label column (source)
from pyspark.ml.feature import StringIndexer

si = StringIndexer(inputCol="source", outputCol="label")
si_model = si.fit(wiki_data)
wiki_data = si_model.transform(wiki_data)
wiki_data.filter("source = 'metapedia'").show(5)
wiki_data.filter("source = 'rational'").show(5)

# COMMAND ----------

train, test = wiki_data.select("features", "label").randomSplit([0.8,0.2])
print(train.count())
print(test.count())

# COMMAND ----------

train.select("label").summary().show()

# COMMAND ----------

# Logistic regression
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(family="binomial")
lr_model = lr.fit(train)
lr_predictions = lr_model.transform(test)
lr_predictions.limit(5).show()
lr_model.write().overwrite().save("/dbfs/FileStore/models/binary_logistic_regression")

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

mcc_evaluator = MulticlassClassificationEvaluator()
print(f"Logistic regression F1 (multi-class) {mcc_evaluator.evaluate(lr_predictions):.4f}")
binary_evaluator = BinaryClassificationEvaluator()
print(f"Logistic regression F1 (binary) {binary_evaluator.evaluate(lr_predictions):.4f}")

# COMMAND ----------

# print("Coefficients: " + str(lr_model.coefficients))
# print("Intercept: " + str(lr_model.intercept))

# COMMAND ----------



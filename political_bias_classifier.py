# Databricks notebook source
# MAGIC %pip install syllapy

# COMMAND ----------

import pyspark
import numpy as np
import pandas as pd
import pickle
import os
from pyspark.sql.types import StructField, StructType, StringType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

pd.set_option("max_colwidth", 800)

# COMMAND ----------


# COMMAND ----------

# load rational wiki sentences
dir = "/dbfs/FileStore/tables"
all_sentences = []
for filename in os.listdir(dir):
    if filename[-3:] == "pkl":
      with open(os.path.join(dir, filename), "rb") as f:
          tmp = pickle.load(f)
          all_sentences.extend(tmp)

rational_rdd = spark.sparkContext.parallelize(all_sentences, 100)
rational_rdd = rational_rdd.map(lambda x: ("rational", x))

rational_schema = StructType([
    StructField("source", StringType()),
    StructField("sample", StringType()),
])

rational_df = spark.createDataFrame(rational_rdd, rational_schema)
rational_df = rational_df.dropDuplicates()
rational_df.count()

# COMMAND ----------

# load metapedia wiki
csv_schema = StructType([
    StructField("id", StringType()),
    StructField("source", StringType()),
    StructField("sample", StringType()),
])

metapedia_df = spark.read.schema(csv_schema)\
  .option("recursiveFileLookup", "true")\
  .option("header", "true")\
  .csv("s3://wallerstein-wikipedia-bias/metapedia/")
metapedia_df = metapedia_df.drop("id").dropDuplicates()
metapedia_df.count()

# COMMAND ----------

# load powerbase wiki
powerbase_df = spark.read.schema(csv_schema)\
  .option("recursiveFileLookup", "true")\
  .option("header", "true")\
  .csv("s3://wallerstein-wikipedia-bias/powerbase/")
powerbase_df = powerbase_df.drop("id").dropDuplicates()
powerbase_df.count()

# COMMAND ----------

# load conservapedia wiki
conservapedia_df = spark.read.schema(csv_schema)\
  .option("recursiveFileLookup", "true")\
  .option("header", "true")\
  .csv("s3://wallerstein-wikipedia-bias/conservapedia/")
conservapedia_df = conservapedia_df.drop("id").dropDuplicates()
conservapedia_df.count()

# COMMAND ----------

# create union of equal numbers of each corpus
ratio = metapedia_df.count() / rational_df.count()
all_df = rational_df\
  .select(F.lit("left").alias("source"), "sample")\
  .sample(True, ratio)\
  .union(metapedia_df.select(F.lit("right").alias("source"), "sample"))
ratio = metapedia_df.count() / powerbase_df.count() 
all_df = all_df.union(powerbase_df\
                      .select(F.lit("left").alias("source"), "sample")\
                      .sample(True, ratio))
ratio = metapedia_df.count() / conservapedia_df.count() 
all_df = all_df.union(conservapedia_df\
                      .select(F.lit("right").alias("source"), "sample")\
                      .sample(True, ratio))
all_df.count()

# COMMAND ----------

# clean-ups
from pyspark.sql.functions import regexp_replace

cleaned = all_df.select("source", regexp_replace(F.col("sample"), "(https?):\/\/(www\.)?[a-z0-9\.:].*?(?=\s)", "").alias("sample"))
cleaned = all_df.select(
  "source", 
  regexp_replace(F.col("sample"), "([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})", "").alias("sample")
)

# COMMAND ----------

# tokenize
from pyspark.ml.feature import RegexTokenizer, Tokenizer

regex_tokenizer = RegexTokenizer(inputCol="sample", outputCol="words", gaps=False, pattern="[a-zA-Z]+")
tokenized = regex_tokenizer.transform(cleaned)
tokenized.show(5)

# COMMAND ----------

# calculate sentence complexity metrics
from pyspark.sql.types import FloatType
import syllapy

@F.udf(returnType=FloatType())
def avg_syllables(words):
  total_words = len(words)
  if total_words:
    syllables = sum([syllapy.count(w) for w in words])
    return syllables / total_words
  else:
    return 0.0

tokenized = tokenized.filter(F.size(F.col("words")) > 1)
tokenized_with_complexity = tokenized.withColumn("avg_syllables", avg_syllables(F.col("words")))
tokenized_with_complexity = tokenized_with_complexity.withColumn("words_per_sentence", F.size(F.col("words")))
tokenized_with_complexity.sample(True, 0.00002).select("words", "avg_syllables", "words_per_sentence").toPandas()

# COMMAND ----------

# term frequency vectors
from pyspark.ml.feature import CountVectorizer

cv = CountVectorizer(inputCol="words", outputCol="vectorized_words")
cv_model = cv.fit(tokenized_with_complexity)
wiki_data = cv_model.transform(tokenized_with_complexity)

# COMMAND ----------

# ngrams 
from pyspark.ml.feature import NGram
from pyspark.ml.feature import CountVectorizer

bigram = NGram(inputCol="words", outputCol="ngrams", n=2)
bigram = bigram.transform(tokenized_with_complexity).select("source", "ngrams", "avg_syllables", "words_per_sentence")

cv = CountVectorizer(inputCol="ngrams", outputCol="vectorized")
cv_model = cv.fit(bigram)
wiki_data = cv_model.transform(bigram)

# COMMAND ----------

# merge ngrams with sentence complexity metrics
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(
  inputCols=["words_per_sentence", "avg_syllables", "vectorized"],
  outputCol="features"
)
vectorized = vec_assembler.transform(wiki_data).select("source", "features")
# display(wiki_data.sample(True, 0.00002))

# COMMAND ----------

# transform label column (source)
from pyspark.ml.feature import StringIndexer

si = StringIndexer(inputCol="source", outputCol="label")
si_model = si.fit(vectorized)
cleaned_and_transformed = si_model.transform(vectorized)
cleaned_and_transformed.select("source", "label").dropDuplicates().show()

# COMMAND ----------

train, test = cleaned_and_transformed.select("features", "label").randomSplit([0.8,0.2])
print(train.count())
print(test.count())

# COMMAND ----------

# Logistic regression
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression()
lr_model = lr.fit(train)
lr_predictions = lr_model.transform(test)
lr_predictions.limit(5).show()
lr_model.write().overwrite().save("/dbfs/FileStore/models/logistic_regression")

# COMMAND ----------

sources = cleaned_and_transformed.select("source").dropDuplicates().toLocalIterator()
for source, measure in zip(sources, lr_model.summary.fMeasureByLabel()):
  print(f"Source: {source} Measure {measure}")

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

bc_evaluator = BinaryClassificationEvaluator()
print(f"Logistic regression F1 {bc_evaluator.evaluate(lr_predictions):.3f}")
# 0.778 bag-of-words without sentence complexity metrics
# 0.779 bag-of-words with sentence complexity metrics
# 0.842 bi-grams with sentence complexity
# 0.833 tri-grams with sentence complexity

# COMMAND ----------

arr = lr_model.coefficientMatrix.toArray()
words_and_coefficients = list(zip(cv_model.vocabulary, arr[0]))
sorted(words_and_coefficients, key=lambda p: p[1], reverse=False)[:10]

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()
nb_model = nb.fit(train)
nb_predictions = nb_model.transform(test)
nb_model.write().overwrite().save("/dbfs/FileStore/models/naive_bayes")
print(f"Naive bayes F1 {bc_evaluator.evaluate(nb_predictions):.3f}")

# COMMAND ----------

"""
Next steps:
3. Investigate better sentence segmentation
4. Correlate coefficientMatrix back to words to see which words are making the most impact
5. Try other classification models (Gradient boosted trees, naive bayes, ...)
6. Make pipeline of transformations
"""

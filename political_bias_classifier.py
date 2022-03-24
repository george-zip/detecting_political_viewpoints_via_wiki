# Databricks notebook source
# MAGIC %pip install syllapy

# COMMAND ----------

import pyspark
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
pd.set_option("max_colwidth", 800)

# COMMAND ----------

# load rational wiki sentences
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

# label rational wiki sentences
rational_rdd = rational_rdd.map(lambda x: ("rational", x))
rational_rdd.count()

# COMMAND ----------

# create dataframe
from pyspark.sql.types import StructField, StructType, StringType
from pyspark.sql.functions import col

rational_schema = StructType([
    StructField("source", StringType()),
    StructField("sample", StringType()),
])

rational_df = spark.createDataFrame(rational_rdd, rational_schema)
rational_df = rational_df.dropDuplicates()
rational_df.count()
# rational_df.sample(True, 0.00001).toPandas()

# COMMAND ----------

spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key", )
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key", )

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
  .select(F.lit("left-wing").alias("source"), "sample")\
  .sample(True, ratio)\
  .union(metapedia_df.select(F.lit("right-wing").alias("source"), "sample"))
ratio = metapedia_df.count() / powerbase_df.count() 
all_df = all_df.union(powerbase_df\
                      .select(F.lit("left-wing").alias("source"), "sample")\
                      .sample(True, ratio))
ratio = metapedia_df.count() / conservapedia_df.count() 
all_df = all_df.union(conservapedia_df\
                      .select(F.lit("right-wing").alias("source"), "sample")\
                      .sample(True, ratio))
all_df.count()

# COMMAND ----------



# COMMAND ----------

# clean-ups
from pyspark.sql.functions import regexp_replace

all_df = all_df.select("source", regexp_replace(F.col("sample"), "(https?):\/\/(www\.)?[a-z0-9\.:].*?(?=\s)", "").alias("sample"))
all_df = all_df.select("source", regexp_replace(F.col("sample"), "([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})", "").alias("sample"))
all_df.show(5)

# COMMAND ----------

# tokenize
from pyspark.ml.feature import RegexTokenizer, Tokenizer

regex_tokenizer = RegexTokenizer(inputCol="sample", outputCol="words", gaps=False, pattern="[a-zA-Z]+")
tokenized = regex_tokenizer.transform(all_df)

# COMMAND ----------

# calculate flesch-kincaid score
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

tokenized = tokenized.filter(F.size(col("words")) > 1)
tokenized_with_complexity = tokenized.withColumn("avg_syllables", avg_syllables(F.col("words")))
tokenized_with_complexity = tokenized_with_complexity.withColumn("words_per_sentence", F.size(F.col("words")))
tokenized_with_complexity.sample(True, 0.00002).toPandas()

# COMMAND ----------

# term frequency vectors
from pyspark.ml.feature import CountVectorizer

cv = CountVectorizer(inputCol="words", outputCol="vectorized_words")
# tokenized_with_complexity = tokenized
# cv = CountVectorizer(inputCol="words", outputCol="features")
cv_model = cv.fit(tokenized_with_complexity)
wiki_data = cv_model.transform(tokenized_with_complexity)

# COMMAND ----------

# merge term vectors with flesch kincaid score
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(
  inputCols=["words_per_sentence", "avg_syllables", "vectorized_words"],
  outputCol="features"
)
wiki_data = vec_assembler.transform(wiki_data)
# display(wiki_data.sample(True, 0.00002))

# COMMAND ----------

from random import choices
print(len(cv_model.vocabulary))
choices(cv_model.vocabulary, k=5)

# COMMAND ----------

# transform label column (source)
from pyspark.ml.feature import StringIndexer

si = StringIndexer(inputCol="source", outputCol="label")
si_model = si.fit(wiki_data)
wiki_data = si_model.transform(wiki_data)
wiki_data.select("source", "label").dropDuplicates().show()

# COMMAND ----------

train, test = wiki_data.select("features", "label").randomSplit([0.8,0.2])
print(train.count())
print(test.count())

# COMMAND ----------

# Logistic regression
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression()
lr_model = lr.fit(train)
lr_predictions = lr_model.transform(test)
lr_predictions.limit(5).show()
lr_model.write().overwrite().save("/dbfs/FileStore/models/multinomial_logistic_regression")

# COMMAND ----------

print(lr_model.summary.fMeasureByLabel())
# print(lr_model.summary.labels)
wiki_data.select("source", "label").dropDuplicates().show()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

mcc_evaluator = MulticlassClassificationEvaluator()
print(f"Logistic regression F1 (multi-class) {mcc_evaluator.evaluate(lr_predictions):.3f}")
# 0.7786 without sentence complexity metrics
# 0.7798 with sentence complexity metrics

# COMMAND ----------

arr = lr_model.coefficientMatrix.toArray()
words_and_coefficients = list(zip(cv_model.vocabulary, arr[0]))
sorted(words_and_coefficients, key=lambda p: p[1], reverse=False)[:10]

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()
# print(dt.explainParams())
nb_model = nb.fit(train)
nb_predictions = nb_model.transform(test)
# dt_predictions.limit(5).show()
nb_model.write().overwrite().save("/dbfs/FileStore/models/naive_bayes")
mcc_evaluator = MulticlassClassificationEvaluator()
print(f"Naive bayes F1 (multi-class) {mcc_evaluator.evaluate(nb_predictions):.3f}")

# COMMAND ----------

"""
Next steps:
3. Investigate better sentence segmentation
4. Correlate coefficientMatrix back to words to see which words are making the most impact
5. Try other classification models (Gradient boosted trees, naive bayes, ...)
6. Make pipeline of transformations
"""

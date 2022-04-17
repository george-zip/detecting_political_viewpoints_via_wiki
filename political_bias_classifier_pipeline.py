# Databricks notebook source
# MAGIC %pip install syllapy

# COMMAND ----------

spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key", dbutils.secrets.get("aws_s3", "access_key"))
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key", dbutils.secrets.get("aws_s3", "secret_key"))

# COMMAND ----------

import pyspark
import numpy as np
import pandas as pd
import pickle
import os
from pyspark.sql.types import StructField, StructType, StringType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

csv_schema = StructType([
    StructField("id", StringType()),
    StructField("source", StringType()),
    StructField("sample", StringType()),
])

pd.set_option("max_colwidth", 800)

# COMMAND ----------

def load_rational_corpus():
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
  return rational_df.dropDuplicates()

# COMMAND ----------

def load_metapedia_corpus(schema=csv_schema):
  metapedia_df = spark.read.schema(schema)\
    .option("recursiveFileLookup", "true")\
    .option("header", "true")\
    .csv("s3://wallerstein-wikipedia-bias/metapedia/")
  return metapedia_df.drop("id").dropDuplicates()

# COMMAND ----------

def load_powerbase_corpus(schema=csv_schema):
  powerbase_df = spark.read.schema(csv_schema)\
    .option("recursiveFileLookup", "true")\
    .option("header", "true")\
    .csv("s3://wallerstein-wikipedia-bias/powerbase/")
  return powerbase_df.drop("id").dropDuplicates()

# COMMAND ----------

def load_conservapedia_corpus(schema=csv_schema):
  conservapedia_df = spark.read.schema(csv_schema)\
    .option("recursiveFileLookup", "true")\
    .option("header", "true")\
    .csv("s3://wallerstein-wikipedia-bias/conservapedia/")
  return conservapedia_df.drop("id").dropDuplicates()

# COMMAND ----------

rational_df = load_rational_corpus()
metapedia_df = load_metapedia_corpus()
powerbase_df = load_powerbase_corpus()
conservapedia_df = load_conservapedia_corpus()

# merge with equal numbers of each dataset
# partition the dataset into conservative and liberal
ratio = metapedia_df.count() / rational_df.count()
all_df = rational_df\
  .select(F.lit("liberal").alias("source"), "sample")\
  .sample(True, ratio)\
  .union(metapedia_df.select(F.lit("conservative").alias("source"), "sample"))
ratio = metapedia_df.count() / powerbase_df.count() 
all_df = all_df.union(powerbase_df\
                      .select(F.lit("liberal").alias("source"), "sample")\
                      .sample(True, ratio))
ratio = metapedia_df.count() / conservapedia_df.count() 
all_df = all_df.union(conservapedia_df\
                      .select(F.lit("conservative").alias("source"), "sample")\
                      .sample(True, ratio))
print(f"Total records {all_df.count()}")
all_df.printSchema()

# COMMAND ----------

# clean-ups
from pyspark.sql.functions import regexp_replace

# remove URLs and email addresses that might give away the source
cleaned = all_df.select("source", regexp_replace(F.col("sample"), "(https?):\/\/(www\.)?[a-z0-9\.:].*?(?=\s)", "").alias("sample"))
cleaned = cleaned.select(
  "source", 
  regexp_replace(F.col("sample"), "([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})", "").alias("sample")
)

# COMMAND ----------

import syllapy
from pyspark.sql.types import FloatType
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import Identifiable

@F.udf(returnType=FloatType())
def avg_syllables(words):
  total_words = len(words)
  if total_words:
    syllables = sum([syllapy.count(w) for w in words])
    return syllables / total_words
  else:
    return 0.0

class SentenceComplexity(Transformer, Identifiable):
  
  @keyword_only
  def __init__(self, inputCol="words"):
    self.inputCol = inputCol
    self.syllablesCol = "avg_syllables"
    self.wordsCol = "words_per_sentence"
    
  def getOutputCols(self):
    return [self.syllablesCol, self.wordsCol]
    
  def _transform(self, dataset):
    tmp = dataset.filter(F.size(F.col(self.inputCol)) > 1)
    return tmp.withColumn(self.syllablesCol, avg_syllables(F.col(self.inputCol)))\
      .withColumn(self.wordsCol, F.size(F.col(self.inputCol)))

# COMMAND ----------

train, test = cleaned.randomSplit([0.8,0.2])
print(train.count())
print(test.count())

# COMMAND ----------

# construct and fit pipeline
from pyspark.ml.feature import RegexTokenizer, CountVectorizer, NGram, StringIndexer, VectorAssembler, Word2Vec
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline

# break text into words
regex_tokenizer = RegexTokenizer(inputCol="sample", outputCol="words", gaps=False, pattern="[a-zA-Z]+")
# construct bigrams
# bigrams = NGram(inputCol=regex_tokenizer.getOutputCol(), outputCol="ngrams", n=2)
# vectorize bigrams by count
# bigram_vectorizer = CountVectorizer(inputCol=bigrams.getOutputCol(), outputCol="ngrams_vectorized")
# add sentence complexity measures
# sentence_complexity = SentenceComplexity(inputCol=regex_tokenizer.getOutputCol())
# add unigrams
unigrams = CountVectorizer(inputCol=regex_tokenizer.getOutputCol(), outputCol="uni_vecs")
word2vec = Word2Vec(inputCol=regex_tokenizer.getOutputCol(), outputCol="word2vec")
# combine bigram and sentence complexity measures into single feature vector
vec_assembler = VectorAssembler(
  inputCols= [unigrams.getOutputCol(), word2vec.getOutputCol()],
  outputCol="features"
)
# generate numeric label column 
si = StringIndexer(inputCol="source", outputCol="label")
# best model using Ridge Regression
# lr = LogisticRegression(regParam=0.1, elasticNetParam=0.0, threshold=0.5)
rf = RandomForestClassifier()
pipeline = Pipeline(stages=[regex_tokenizer, word2vec, unigrams, vec_assembler, si, rf])

# model = pipeline.fit(train)
# model.save("/dbfs/FileStore/models/logistic_regression_pipeline")

# COMMAND ----------

lr_predictions = model.transform(test).select("label", "prediction", "rawPrediction", "probability")
# lr_predictions.sample(True, 0.0002).toPandas()""

# COMMAND ----------

results = lr_predictions.toPandas()
                  
tp = results[(results.label == 1.0) & (results.prediction == 1.0)].prediction.count()
tn = results[(results.label == 0.0) & (results.prediction == 0.0)].prediction.count()
fp = results[(results.label == 0.0) & (results.prediction == 1.0)].prediction.count()
fn = results[(results.label == 1.0) & (results.prediction == 0.0)].prediction.count()

total = results.prediction.count()
print(f"Total test results: {total:,}")
print(f"Test set skew {results.skew()[0]:.3f}")
print(f"True positives: {tp:,}")
print(f"True negatives: {tn:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
accuracy = (tp + tn) / total
print(f"Accuracy: {accuracy:.3f}")
recall = tp / (tp + fn)
print(f"Recall: {recall:.3f}")
precision = tp / (tp + fp)
print(f"Precision: {precision:.3f}")
f1 = 2 * ((precision * recall) / (precision + recall))
print(f"F1 score: {f1:.3f}")

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

bc_evaluator = BinaryClassificationEvaluator()
evaluation = bc_evaluator.evaluate(lr_predictions)
print(f"Area under ROC {evaluation:.3f}")

# COMMAND ----------

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
plt.style.use('seaborn')

fpr, tpr, _ = roc_curve(results.label, results.probability.apply(lambda v: v.toArray()[1]))
plt.plot(fpr, tpr, linestyle='--',color='orange', label='Logistic Regression')
random_probs = [0 for i in range(len(results))]
p_fpr, p_tpr, _ = roc_curve(results.label, random_probs)
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show()

# COMMAND ----------



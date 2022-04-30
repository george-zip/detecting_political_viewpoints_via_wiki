# Databricks notebook source
# MAGIC %pip install syllapy
# MAGIC %pip install gensim
# MAGIC %pip install spacy

# COMMAND ----------

import spacy
!python -m spacy download en_core_web_sm

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

nlp = spacy.load("en_core_web_sm")

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
print(f"rational records {rational_df.count()}")
metapedia_df = load_metapedia_corpus()
print(f"metapedia records {metapedia_df.count()}")
powerbase_df = load_powerbase_corpus()
print(f"powerbase records {powerbase_df.count()}")
conservapedia_df = load_conservapedia_corpus()
print(f"conservapedia records {conservapedia_df.count()}")

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

import gensim.parsing.preprocessing as gsp
from gensim import utils
import re

def preprocess(df):
  filters = [
    gsp.strip_tags, 
    gsp.strip_punctuation,
    gsp.strip_multiple_whitespaces,
    gsp.strip_numeric,
    gsp.strip_short, 
    gsp.remove_stopwords
  ]

  def clean_text(df):
    s = df.sample
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
      s = f(s)
    return (df.source, s)
  
  return df.rdd.map(lambda x : clean_text(x))

input_rdd = preprocess(cleaned)

# COMMAND ----------

input_df = input_rdd.toDF(['source','sample'])

# COMMAND ----------

import syllapy
from pyspark.sql.types import FloatType, IntegerType, StringType, MapType
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import Identifiable, DefaultParamsWritable
from collections import Counter

@F.udf(returnType=FloatType())
def avg_syllables(words):
  total_words = len(words)
  if total_words:
    syllables = sum([syllapy.count(w) for w in words])
    return syllables / total_words
  else:
    return 0.0

@F.udf(returnType=FloatType())
def avg_chars(words):
  total_words = len(words)
  if total_words:
    return sum([len(w) for w in words]) / total_words
  else:
    return 0.0  
  
def lexical_category(text):
  doc = nlp(text)
  return dict(Counter([token.pos_ for token in doc]))

adjectives_udf = udf(lambda c: lexical_category(c), MapType(StringType(), IntegerType()))
  
class SentenceComplexity(Transformer, Identifiable):
  
  @keyword_only
  def __init__(self, inputCol="words"):
    self.inputCol = inputCol
    self.syllablesCol = "avg_syllables"
    self.wordsCol = "words_per_sentence"
    self.charsCol = "avg_chars"
    
  def getOutputCols(self):
    return [
              self.syllablesCol, self.wordsCol, self.charsCol,
              "adj_ratio", "n_ratio", "v_ratio", "adv_ratio", 
              "c_ratio"
           ]
    
  def _transform(self, dataset):
    step1 = dataset\
      .withColumn("pos_count", adjectives_udf(F.col("sample")))\
      .withColumn(self.syllablesCol, avg_syllables(F.col(self.inputCol)))\
      .withColumn(self.wordsCol, F.size(F.col(self.inputCol)))\
      .withColumn(self.charsCol, avg_chars(F.col(self.inputCol)))
    
    step2 = step1\
      .withColumn("adj_ratio", F.element_at(step1.pos_count, F.lit("ADJ")) / F.col(self.wordsCol))\
      .withColumn("n_ratio", F.element_at(step1.pos_count, F.lit("NOUN")) / F.col(self.wordsCol))\
      .withColumn("v_ratio", F.element_at(step1.pos_count, F.lit("VERB")) / F.col(self.wordsCol))\
      .withColumn("co_count", F.element_at(step1.pos_count, F.lit("CONJ")))\
      .withColumn("cc_count", F.element_at(step1.pos_count, F.lit("CCONJ")))\
      .withColumn("sc_count", F.element_at(step1.pos_count, F.lit("SCONJ")))\
      .withColumn("adv_ratio", F.element_at(step1.pos_count, F.lit("ADV")) / F.col(self.wordsCol))
    
    step3 = step2\
      .na.fill(0)\
      .withColumn("c_ratio", (F.col("co_count") + F.col("cc_count") + F.col("sc_count")) / F.col(self.wordsCol))
    
    return step3.fillna(0)

# COMMAND ----------

training_set, test_set = input_df.randomSplit([0.8,0.2])
print(training_set.count())
print(test_set.count())

# COMMAND ----------

# construct and fit pipeline
from pyspark.ml.feature import RegexTokenizer, Tokenizer, CountVectorizer, NGram, StringIndexer, VectorAssembler, Word2Vec
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline

# break text into words
tokenizer = Tokenizer(inputCol="sample", outputCol="words")
# construct ngrams
ngrams = NGram(inputCol=tokenizer.getOutputCol(), outputCol="ngrams", n=1)
# vectorize bigrams by count
ngrams_vectorizer = CountVectorizer(inputCol=ngrams.getOutputCol(), outputCol="ngrams_vec")
# add sentence complexity measures
sentence_complexity = SentenceComplexity(inputCol=tokenizer.getOutputCol())
# combine ngram and sentence complexity measures into single feature vector
vec_assembler = VectorAssembler(
  inputCols= sentence_complexity.getOutputCols() + [ngrams_vectorizer.getOutputCol()],
  outputCol="features"
)
# generate numeric label column 
si = StringIndexer(inputCol="source", outputCol="label")
# best model using Ridge Regression
lr = LogisticRegression(regParam=0.1, elasticNetParam=0.0)
pipeline = Pipeline(stages=[tokenizer, sentence_complexity, ngrams, ngrams_vectorizer, vec_assembler, si, lr])

model = pipeline.fit(training_set)

# COMMAND ----------

lr_predictions = model.transform(test_set)

# COMMAND ----------

def print_accuracy_measures(predictions):
  results = predictions.toPandas()

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
  return results

results = print_accuracy_measures(lr_predictions.select("label", "prediction", "probability"))

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


sentence_complexity_cols = [
  "avg_syllables", "words_per_sentence", "avg_chars",
  "adj_ratio", "n_ratio", "v_ratio", "adv_ratio", 
  "c_ratio"
]
features = sentence_complexity_cols + model.stages[3].vocabulary
features_and_coefficients = list(zip(features, model.stages[-1].coefficients))
feature_map = {p[0]:p[1] for p in sorted(features_and_coefficients, key=lambda p: p[1], reverse=True)}


# COMMAND ----------

for f in feature_map:
  if f in sentence_complexity_cols:
    print(f"{f}: {feature_map[f]}")

# COMMAND ----------

sorted(features_and_coefficients, key=lambda p: p[1], reverse=True)[:20]

# COMMAND ----------

speeches_schema = StructType([
  StructField("source", StringType()),
  StructField("sample", StringType()),
])

speeches_df = spark.read.schema(speeches_schema)\
  .option("header", "true")\
  .csv("s3://wallerstein-wikipedia-bias/political_speeches/test_set.csv").dropDuplicates()

print(speeches_df.count())
speeches_cleaned = preprocess(speeches_df)
lr_speech_predictions = model.transform(speeches_df)

# COMMAND ----------

results = print_accuracy_measures(lr_speech_predictions.select("label", "prediction", "probability"))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

bc_evaluator = BinaryClassificationEvaluator()
evaluation = bc_evaluator.evaluate(lr_speech_predictions)
print(f"Area under ROC {evaluation:.3f}")

# COMMAND ----------

lr_speech_predictions\
  .groupBy("source").avg(\
  "adj_ratio", "n_ratio", "v_ratio", 
  "avg_syllables", "avg_chars", "words_per_sentence",
  "c_ratio", "adv_ratio").show()

# COMMAND ----------



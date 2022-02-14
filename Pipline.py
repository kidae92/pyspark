from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from sklearn.datasets import load_iris
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext
import pandas as pd
import findspark
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


findspark.init()
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


iris = load_iris()
iris_data = iris.data
iris_label = iris.target
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_pdf = pd.DataFrame(iris_data, columns=iris_columns)
iris_pdf['target'] = iris_label

iris_sdf = spark.createDataFrame(iris_pdf)
train_sdf, test_sdf = iris_sdf.randomSplit([0.8, 0.2], seed=42)
train_sdf.cache()

# 첫번째 stage는 Feature Vectorization을 위해 VectorAssembler 객체 생성.
stage_1 = VectorAssembler(inputCols=iris_columns, outputCol='features')
# 두번째 stage는 학습을 위한 결정 트리 Estimator 생성.
stage_2 = DecisionTreeClassifier(
    featuresCol='features', labelCol='target', maxDepth=5)

# Feature Vectorization 변환-> 학습 pipeline을 생성.
pipeline = Pipeline(stages=[stage_1, stage_2])

# Estimator가 포함된 Pipeline객체의 fit(train_sdf)를 호출하면 학습 데이터에 transformation을 적용하여 Estimator의 학습까지 수행된 PipelineModel 객체를 반환.
# train_sdf_vectorized = stage_1.transform(train_sdf) , estimator_model = stage_2.fit(train_sdf_vectorized)
pipeline_model = pipeline.fit(train_sdf)

# print(type(pipeline), type(pipeline_model))

# test_sdf_vectorized = stage_1.transform(test_sdf), estimator_model.transform(test_sdf_vectorized)
predictions = pipeline_model.transform(test_sdf)
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol='target', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator_accuracy.evaluate(predictions)
print('정확도:', accuracy)

# stages 속성은 pipeline_model이 가지는 stage별 객체를 리스트로 가지고 있음.
# print(pipeline_model.stages)

# PipelineModel의 stages 속성에서 개별 stage에 있는 객체를 가져 올 수 있음.
vector_assembler = pipeline_model.stages[0]
dt_model = pipeline_model.stages[-1]
# print(dt_model)
# print(vector_assembler)
vec_assembler = VectorAssembler(inputCols=iris_columns, outputCol='features')

test_feature_vector_df = vec_assembler.transform(test_sdf)
predictions = dt_model.transform(test_feature_vector_df)

accuracy = evaluator_accuracy.evaluate(predictions)
print('정확도:', accuracy)

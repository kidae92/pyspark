from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
import scipy as sp
import findspark
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext
import pyspark  # Call this only after findspark
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

findspark.init()
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# iris 데이터 세트 로딩
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

# iris 데이터 세트를 numpy에서 pandas DataFrame으로 변환
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

iris_pdf = pd.DataFrame(iris_data, columns=iris_columns)
iris_pdf['label'] = iris_label

# iris Pandas DataFrame을 Spark DataFrame으로 변환
iris_sdf = spark.createDataFrame(iris_pdf)


train_sdf, test_sdf = iris_sdf.randomSplit([0.7, 0.3], seed=0)

# VectorAssembler 객체와 Estimator 객체 생성.
vector_assembler = VectorAssembler(
    inputCols=iris_columns, outputCol='features')

# 학습 데이터 feature vectorization 적용.
train_sdf_vectorized = vector_assembler.transform(train_sdf)

dt = DecisionTreeClassifier(featuresCol='features',
                            labelCol='label', maxDepth=10)


# CrossValidator에서 하이퍼파라미터 튜닝을 위한 그리드 서치(Grid Search)용 ParamGrid 생성.
# Spark ML DecisionTreeClassifier의 maxDepth는 max_depth, minInstancesPerNode는 min_samples_split(노드 분할 시 최소 sample 건수)
param_grid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10])\
                               .addGrid(dt.minInstancesPerNode, [3, 6])\
                               .build()

# CrossValidator에서 적용할 Evaluator 객체 생성.
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='accuracy')

# Estimator 객체, 하이퍼파라미터 Grid를 가지는, Evaluator 객체, Fold수를 인자로 입력하여 CrossValidator 객체 생성.
cv = CrossValidator(estimator=dt, estimatorParamMaps=param_grid,
                    evaluator=evaluator_accuracy, numFolds=3)

# Cross validation 과 하이퍼파라미터 튜닝 수행.
cv_model = cv.fit(train_sdf_vectorized)

print(type(cv_model))

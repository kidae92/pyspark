from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
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
# print(spark)

iris = load_iris()
iris_data = iris.data
iris_label = iris.target
# print(type(iris_data), type(iris_label), iris_data.shape, iris_label.shape)
# print(iris.feature_names)


# iris 데이터 세트를 numpy에서 pandas DataFrame으로 변환
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_pdf = pd.DataFrame(iris_data, columns=iris_columns)
iris_pdf['target'] = iris_label


'''
#Spark DataFrame 생성 후 ML 알고리즘 학습 및 예측 수행.
1. iris_sdf DataFrame을 randomSplit()을 이용하여 train용과 test용 DataFrame으로 분할
2. VectorAssembler를 이용하여 모든 feature 컬럼들을 하나의 feature vector로 변환
3. Estimator 객체를 생성하고, fit() 메소드를 호출하여 ML Model 반환
4. ML Model을 이용하여 테스트 DataFrame에 예측 수행.
'''

iris_sdf = spark.createDataFrame(iris_pdf)
# print(type(iris_sdf))


'''
iris_sdf DataFrame을 randomSplit()을 이용하여 train용과 test용 DataFrame으로 분할 
'''

train_sdf, test_sdf = iris_sdf.randomSplit([0.8, 0.2], seed=42)
train_sdf.cache()
# print(iris_sdf.count(), train_sdf.count(), test_sdf.count())
# print(iris_sdf, train_sdf, test_sdf)


# 이 부분이 굉장히 특이함
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# VectorAssembler() 생성 인자로 outputCols가 아닌 outputCol이 입력됨에 유의
vec_assembler = VectorAssembler(inputCols=iris_columns, outputCol='features')
'''
VectorAssembler 객체의 transform() 메소드를 호출하여 모든 feature 컬럼을 하나의 feature vector로 변환. 
'''
train_feature_vector_df = vec_assembler.transform(train_sdf)

# print(type(train_feature_vector_df))


# Decision Tree 로 학습 수행.
# feature vectorization된 column을 무조건 넣어줘야 함
dt = DecisionTreeClassifier(featuresCol='features',
                            labelCol='target', maxDepth=5)

# ML 알고리즘 객체의 fit()메소드를 이용하여 train feature vector 데이터 세트를 학습하고 이를 ML Model로 반환함.
# scikit learn은 ML 알고리즘 객체로 fit()만 호출하면 해당 객체가 학습이 되었으나 Spark ML을 반드시 ML Model로 반환 받아야 함.
dt_model = dt.fit(train_feature_vector_df)  # dt.fit(X_train, y_label)
# print('DecisionTree Estimator type:', type(dt),
#       'DecisionTree Estimator Model type:', type(dt_model))
# print(dt)
# print(dt_model)

# 테스트 데이터를 feature vector로 변환.
test_feature_vector_df = vec_assembler.transform(test_sdf)

# print(type(test_feature_vector_df))


# 테스트 데이터 + 변환된 feature vector로 구성된 DataFrame을 학습된 모델의 transform() 메소드를 이용하여 예측 수행.
# 학습된 모델은 ML 알고리즘 객체의 생성자로 featuresCols인자로 'features' 컬럼이 주어졌으므로 feature vector컬럼명을 인지하고 있음.
# transform() 수행 결과로 입력 DataFrame에 rawPrediction, probability, prediction 3개의 컬럼이 추가.
# rawPrediction은 ML 알고리즘 별로 다를 수 있음. Logistic Regression의 경우 예측 label 별로, 예측 수행 전 sigmoid함수 적용 전 값.
# probability는 예측 label별 예측 확률값, prediction은 최종 예측 값.
predictions = dt_model.transform(test_feature_vector_df)
# print(type(predictions))


# ML 알고리즘 객체 생성.
lr = LogisticRegression(featuresCol='features', labelCol='target', maxIter=10)

# ML 알고리즘 객체의 fit()메소드를 이용하여 train feature vector 데이터 세트를 학습하고 이를 ML Model로 반환함.
# scikit learn은 ML 알고리즘 객체로 fit()만 호출하면 해당 객체가 학습이 되었으나 Spark ML을 반드시 ML Model로 반환 받아야 함.
lr_model = lr.fit(train_feature_vector_df)

predictions = lr_model.transform(test_feature_vector_df)
# print(type(predictions))


# ML 알고리즘 객체 생성.
lr = LogisticRegression(featuresCol='features', labelCol='target', maxIter=10)

# ML 알고리즘 객체의 fit()메소드를 이용하여 train feature vector 데이터 세트를 학습하고 이를 ML Model로 반환함.
# scikit learn은 ML 알고리즘 객체로 fit()만 호출하면 해당 객체가 학습이 되었으나 Spark ML을 반드시 ML Model로 반환 받아야 함.
lr_model = lr.fit(train_feature_vector_df)

predictions = lr_model.transform(test_feature_vector_df)


evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol='target', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator_accuracy.evaluate(predictions)
print('정확도:', accuracy)

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
print(spark)

iris = load_iris()
iris_data = iris.data
iris_label = iris.target
# print(type(iris_data), type(iris_label), iris_data.shape, iris_label.shape)
# print(iris.feature_names)


# iris 데이터 세트를 numpy에서 pandas DataFrame으로 변환
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_pdf = pd.DataFrame(iris_data, columns=iris_columns)


'''
#Spark DataFrame 생성 후 ML 알고리즘 학습 및 예측 수행.
1. iris_sdf DataFrame을 randomSplit()을 이용하여 train용과 test용 DataFrame으로 분할
2. VectorAssembler를 이용하여 모든 feature 컬럼들을 하나의 feature vector로 변환
3. Estimator 객체를 생성하고, fit() 메소드를 호출하여 ML Model 반환
4. ML Model을 이용하여 테스트 DataFrame에 예측 수행.
'''

iris_sdf = spark.createDataFrame(iris_pdf)
print(type(iris_sdf))

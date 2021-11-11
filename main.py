import os
import sys
import tempfile
import shutil

import numpy as np
from numpy.core._methods import _mean
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, Imputer, StandardScaler, StringIndexer, OneHotEncoder, Tokenizer, \
    HashingTF
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.stat import Correlation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import map_filter, array_max, col, countDistinct
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("main").getOrCreate()
path = "/Users/gaogao/PycharmProjects/pythonProject/housing.csv"

# path = ""你可以把你的地址到时候写在这块

housing = spark.read.option("delimiter", ",").option("header", True).csv(path)
housing.show()
housing.show()

housing.printSchema()

print(housing.head(5))

print("Loaded training data as a dataframe with " + str(housing.count()) + " records")

housing.filter('population>10000').show()
# 2-3-1
housing.describe(['housing_median_age']).show()
housing.describe(['total_rooms']).show()
housing.describe(['median_house_value']).show()
housing.describe(['population']).show()

# df.groupby().max('housing_median_age').collect()
# df.describe("housing_median_age").filter("summary = 'max'").select(("housing_median_price").first().asDict()['housing_median_price'])
# max_housing_median_age=df.select("housing_median_age").rdd.max()[0]
# print("the maximum of housing median age is : "+max_housing_median_age)
# min_total_rooms=df.select("total_rooms").rdd.min()[0]
# print("the number of minimum of total rooms is : "+min_total_rooms)

# 2-3-2
housing.agg({'housing_median_age': 'max'}).show()
housing.agg({'total_rooms': 'min'}).show()
housing.agg({'median_house_value': 'mean'}).show()
# 2-4-1
housing.groupby("ocean_proximity").count().orderBy(col("count").desc()).show()
# 2-4-2
newdf = housing.groupBy('ocean_proximity').agg({"median_house_value": "avg"})
newdf.withColumnRenamed('avg(median_house_value)', 'avg_value').show()
# 2-4-3

housing.createOrReplaceTempView("df")
sqldf = spark.sql("SELECT ocean_proximity,avg(median_house_value) as avg_value FROM df GROUP by ocean_proximity")
sqldf.collect()
sqldf.show()

# 2-5
housing_0 = housing \
    .withColumn("median_house_value", housing["median_house_value"].cast('float')) \
    .withColumn("total_rooms", housing["total_rooms"].cast('float')) \
    .withColumn("housing_median_age", housing["housing_median_age"].cast('float')) \
    .withColumn("population", housing["population"].cast('float')) \
    .withColumn("total_bedrooms", housing["total_bedrooms"].cast('float')) \
    .withColumn("longitude", housing["longitude"].cast('float')) \
    .withColumn("latitude", housing["latitude"].cast('float')) \
    .withColumn("households", housing["households"].cast('float')) \
    .withColumn("median_income", housing["median_income"].cast('float'))

housing_0.printSchema()
assembler = VectorAssembler(outputCol="features")
assembler.setInputCols(["median_house_value", "total_rooms", "housing_median_age", "population"])
features_housing_0 = assembler.transform(housing_0)
features_housing_0.show()
features_housing_0.select('features').show()

corr1 = Correlation.corr(features_housing_0, 'features', 'pearson').head()
print("pearson correlation matrix : " + str(corr1[0]))

# 2-6
housingCol1 = housing_0.withColumn('rooms_per_household', housing_0.total_rooms / housing_0.households)
housingCol2 = housingCol1.withColumn('bedrooms_per_room', housingCol1.total_bedrooms / housingCol1.total_rooms)
housingExtra = housingCol2.withColumn('population_per_household', housing_0.population / housing_0.households)
housingExtra.show(5)

# 3-1
renamedHousing = housingExtra.withColumnRenamed('median_house_value', 'label')
colLabel = "label"
colCat = "ocean_proximity"
colNum = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
          'households', 'median_income', 'median_house_value', 'bedrooms_per_room', 'population_per_household']
# ?????????????????????
# for col in renamedHousing.head(0):
#   if col != "label" or "ocean_proximity":
#      colNum.append(col)
# print(colNum)
################################################不太会

for c in renamedHousing.columns:
    print(c, " has null values : ", renamedHousing.filter(renamedHousing[c].isNull()).count())

imputer = Imputer()
imputer.setInputCols(["total_bedrooms", "bedrooms_per_room"])
imputer.setOutputCols(["out_total_bedrooms", "out_bedrooms_per_room"])
imputedHousing = imputer.setStrategy('median').setMissingValue(414).fit(renamedHousing).transform(renamedHousing)
imputedHousing = imputedHousing.drop('total_bedrooms').drop('bedrooms_per_room')

for c in imputedHousing.columns:
    print(c, " has null values : ", imputedHousing.filter(imputedHousing[c].isNull()).count())

colNum_to_scale = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population',
          'households', 'median_income', 'rooms_per_household','population_per_household','out_total_bedrooms','out_bedrooms_per_room']
va = VectorAssembler().setInputCols(colNum_to_scale).setOutputCol('features')
featuredHousing = va.transform(imputedHousing)
featuredHousing.show()

scaler = StandardScaler(withMean=True, withStd=True)
scaler.setInputCol("features").setOutputCol("scaled_features")
scaledHousing = scaler.fit(featuredHousing).transform(featuredHousing)
scaledHousing.select('scaled_features').show()

# 3-2 bu tai ming bai zhe li?????????????????????????????
distinct = renamedHousing.select('ocean_proximity').distinct().collect()
print(distinct)
renamedHousing.agg(countDistinct("ocean_proximity")).show()

indexer = StringIndexer().setInputCol('ocean_proximity').setOutputCol('idx_ocean_proximity')
idxHousing = indexer.fit(renamedHousing).transform(renamedHousing)
idxHousing.show()

encoder = OneHotEncoder().setInputCol('idx_ocean_proximity').setOutputCol('one_hot_ocean_proximity')
ohHousing = encoder.fit(idxHousing).transform(idxHousing)
ohHousing.show()

#4
numPipeline = [imputer,va,scaler]
catPipeline = [indexer,encoder]
pipeline = Pipeline(stages=numPipeline)
newHousing = pipeline.fit(renamedHousing).transform(renamedHousing)
newHousing = newHousing.drop('features')
newHousing.show()
pipeline = pipeline.setStages(catPipeline)
newHousing = pipeline.fit(newHousing).transform(newHousing)
newHousing.show()


va2 = VectorAssembler().setInputCols(['scaled_features','one_hot_ocean_proximity']).setOutputCol('features')
dataset = va2.transform(newHousing).select("features","label")
#dataset.withColumnRenamed('final_features','features')
dataset.show(n=100,truncate=False)

#path_0 = "/Users/gaogao/PycharmProjects/pythonProject/feature.csv"
(trainingData, testData) = dataset.randomSplit([0.8, 0.2])

lr = LinearRegression()
lrModel = lr.fit(trainingData)
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

prediction = lrModel.transform(testData)
prediction.select("prediction","label","features").show(5)

evaluator = RegressionEvaluator(labelCol='label',predictionCol='prediction',metricName='rmse')
rmse = evaluator.evaluate(prediction)
print(rmse,"rmse for normal regression")

dt = DecisionTreeRegressor()
dtModel = dt.fit(trainingData)
predictions = dtModel.transform(testData)
predictions.select("prediction","label","features").show(5)

rmse = evaluator.evaluate(predictions)
print("decision tree rmse : ",rmse)

rf = RandomForestClassifier()
rmModel = rf.fit(trainingData)
predictions = rmModel.transform(testData)
predictions.select("prediction","label","features").show(5)

rmse = evaluator.evaluate(predictions)
print("random forest rmse : ",rmse)

gb = GBTRegressor()
gbModel = gb.fit(trainingData)
predictions = gbModel.transform(testData)
predictions.select("prediction", "label", "features").show(5)
evaluator = RegressionEvaluator()
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = ", rmse)

paramGrid = (ParamGridBuilder()
             #.addGrid(rf.maxDepth, [2, 5, 10, 20, 30])
               .addGrid(rf.numTrees, [1, 5, 10])
             #.addGrid(rf.maxBins, [10, 20, 40, 80, 100])
               .addGrid(rf.maxDepth, [5, 10, 15])
             #.addGrid(rf.numTrees, [5, 20, 50, 100, 500])
            #   .addGrid(rf.numTrees, [1, 2, 3])
             .build())

# Create 5-fold CrossValidator
rfcv = CrossValidator(estimator = rf,
                      estimatorParamMaps = paramGrid,
                      evaluator = evaluator,
                      numFolds = 3)

# Run cross validations.
rfcvModel = rfcv.fit(trainingData)


# Use test set here so we can measure the accuracy of our model on new data
predictions = rfcvModel.transform(testData)
prediction.show(5)
# cvModel uses the best model found from the Cross Validation
# Evaluate best model
print('RMSE:', evaluator.evaluate(predictions))

path= '/Users/gaogao/Downloads/lab1/data/predictions.csv'
predictions = predictions.select('label','predictions')
predictions.write.csv(path)

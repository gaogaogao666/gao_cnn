import os
import sys
import tempfile
import shutil

from numpy.core._methods import _mean
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import map_filter, array_max, col
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
#2-3-1
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

#2-3-2
housing.agg({'housing_median_age': 'max'}).show()
housing.agg({'total_rooms': 'min'}).show()
housing.agg({'median_house_value': 'mean'}).show()
#2-4-1
housing.groupby("ocean_proximity").count().orderBy(col("count").desc()).show()
#2-4-2
newdf=housing.groupBy('ocean_proximity').agg({"median_house_value": "avg"})
newdf.withColumnRenamed('avg(median_house_value)','avg_value').show()
#2-4-3

housing.createOrReplaceTempView("df")
sqldf = spark.sql("SELECT ocean_proximity,avg(median_house_value) as avg_value FROM df GROUP by ocean_proximity")
sqldf.collect()
sqldf.show()

#2-5
housing_0=housing\
    .withColumn("median_house_value",housing["median_house_value"].cast('float'))\
    .withColumn("total_rooms",housing["total_rooms"].cast('float'))\
    .withColumn("housing_median_age",housing["housing_median_age"].cast('float'))\
    .withColumn("population",housing["population"].cast('float'))
housing_0.printSchema()
va = VectorAssembler(outputCol="features")
va.setInputCols(["median_house_value","total_rooms","housing_median_age", "population"])
features_housing_0=va.transform(housing_0)
features_housing_0.show()
features_housing_0.select('features').show()

corr1=Correlation.corr(features_housing_0,'features','pearson').head()
print("pearson correlation matrix : "+str(corr1[0]))

#2-6
housingCol1=housing.withColumn('rooms_per_household',housing.total_rooms/housing.households)
housingCol2=housingCol1.withColumn('bedrooms_per_room',housingCol1.total_bedrooms/housingCol1.total_rooms)
housingExtra=housingCol2.withColumn('population_per_household',housing.population/housing.households)
housingExtra.show(5)


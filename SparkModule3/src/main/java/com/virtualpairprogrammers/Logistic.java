package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;


public class Logistic {

	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder()
				.appName("CaseStudy")
				.config("spark.sql.warehouse.dir", "file:///C:/tmp")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("src/main/resources/part-r-*.csv")
				.filter("is_cancelled = FALSE")
				.drop("observation_date");
		
		csvData = csvData.na().fill(0, new String[] {"firstSub", 
													"all_time_views",
													"last_month_views",
													"next_month_views"})
				.withColumnRenamed("next_month_views", "label");
		
		//Want as label: 1 = didn0t watch any videos, 0 = watched videos
		
		csvData = csvData.withColumn("label", when (col("label").$greater(0), 0).otherwise(1));
		
		Dataset<Row>[] trainAndTest = csvData.randomSplit(new double[] {0.9,0.1});
		Dataset<Row> trainAndTestData = trainAndTest[0];
		Dataset<Row> holdoutData = trainAndTest[1];
		
//		csvData.show();
		
		//paymentMethod
		StringIndexer paymentIndexer = new StringIndexer()
				.setInputCol("payment_method_type")
				.setOutputCol("paymentIndex");
		
		//country
		StringIndexer countryIndexer = new StringIndexer()
				.setInputCol("country")
				.setOutputCol("countryIndex");
		
		//Rebill period
		StringIndexer rebillIndexer = new StringIndexer()
				.setInputCol("rebill_period_in_months")
				.setOutputCol("rebillIndex");
		
		//firstsub
		StringIndexer firstsubIndexer = new StringIndexer()
				.setInputCol("firstSub")
				.setOutputCol("firstsubIndex");
		
		OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
				.setInputCols(new String[] {"paymentIndex", "countryIndex", "rebillIndex", "firstsubIndex"})
				.setOutputCols(new String[] {"paymentVector", "countryVector", "rebillVector", "firstsubVector"});
		
		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(new String[] {"age", "all_time_views", "last_month_views",
						"paymentVector", "countryVector", "rebillVector", "firstsubVector"})
				.setOutputCol("features");
		
		LogisticRegression logisticRegression = new LogisticRegression();
		
		ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
		
		ParamMap[] paramMap = paramGridBuilder.addGrid(logisticRegression.regParam(), new double[] {0.01, 0.1, 0.5})
				.addGrid(logisticRegression.elasticNetParam(), new double[] {0, 0.5, 1})
				.build();
		
		TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
				.setEstimator(logisticRegression)
				.setEvaluator(new RegressionEvaluator().setMetricName("r2"))
				.setEstimatorParamMaps(paramMap)
				.setTrainRatio(0.9);
		
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] {paymentIndexer, 
												countryIndexer,
												rebillIndexer,
												firstsubIndexer,
												encoder,
												vectorAssembler,
												trainValidationSplit});
		
		PipelineModel pipelineModel = pipeline.fit(trainAndTestData);
		
		Dataset<Row> holdoutResults = pipelineModel.transform(holdoutData).drop("probability","rawPrediction", "prediction");
		
		TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[6];
		LogisticRegressionModel logregModel = (LogisticRegressionModel) model.bestModel();
		
		System.out.println("traindata accuracy = " +logregModel.summary().accuracy());
		System.out.println("testdata accuracy = " +logregModel.evaluate(holdoutResults).accuracy());
	
		//what are we really interested in?? --> False negatives
		// also true versus false positives
		
		double truepositive = logregModel.evaluate(holdoutResults).truePositiveRateByLabel()[1];
		double falsepositive = logregModel.evaluate(holdoutResults).truePositiveRateByLabel()[0];
		
		System.out.println("True positives = " + truepositive);
		System.out.println("False positives = " + falsepositive);
		
		//confusionmatrix
		logregModel.transform(holdoutResults).groupBy("label", "prediction").count().show();
				
				
		
				
				
				}

}

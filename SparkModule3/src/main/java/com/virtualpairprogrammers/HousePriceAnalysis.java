package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
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

public class HousePriceAnalysis {

	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder()
				.appName("HousePriceAnalyses")
				.config("spark.sql.warehouse.dir", "file:///C:/tmp")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("src/main/resources/kc_house_data.csv");
		
		csvData = csvData.withColumn("sqft_above_perc", col("sqft_above").divide(col("sqft_living")))
				.withColumnRenamed("price", "label");
		
		Dataset<Row>[] trainAndTest = csvData.randomSplit(new double[] {0.8,0.2});
		Dataset<Row> trainAndTestData = trainAndTest[0];
		Dataset<Row> holdoutData = trainAndTest[1];
		
		//Condition
		StringIndexer conditionIndexer = new StringIndexer()
				.setInputCol("condition")
				.setOutputCol("conditionIndex");
		
//		csvData = conditionIndexer.fit(csvData)
//				.transform(csvData);
		
		//Grade
		StringIndexer gradeIndexer = new StringIndexer()
				.setInputCol("grade")
				.setOutputCol("gradeIndex");
		
//		csvData = gradeIndexer.fit(csvData)
//				.transform(csvData);
		
		//Zipcode
		StringIndexer zipcodeIndexer = new StringIndexer()
				.setInputCol("zipcode")
				.setOutputCol("zipcodeIndex");
		
//		csvData = zipcodeIndexer.fit(csvData)
//				.transform(csvData);
		
		//Waterfront
		StringIndexer wfIndexer = new StringIndexer()
				.setInputCol("waterfront")
				.setOutputCol("waterfrontIndex");
		
//		csvData = wfIndexer.fit(csvData)
//				.transform(csvData);
		
		
		OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
				.setInputCols(new String[] {"conditionIndex", "gradeIndex", "zipcodeIndex", "waterfrontIndex"})
				.setOutputCols(new String[] {"conditionVector", "gradeVector", "zipcodeVector", "waterfrontVector"});
		
//		csvData = encoder.fit(csvData).transform(csvData);
		
		
		
		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(new String[] {"bedrooms", "bathrooms", "sqft_living", "floors", "sqft_above_perc",
						"conditionVector", "gradeVector", "zipcodeVector", "waterfrontVector"})
				.setOutputCol("features");
		
//		Dataset<Row> modelInputData = vectorAssembler.transform(csvData)
//				.select("label", "features");
		
//		modelInputData.show();
		
		
//		LinearRegressionModel model = new LinearRegression()
//				.setMaxIter(10)
//				.setRegParam(0.3)
//				.setElasticNetParam(0.8)
//				.fit(traindata);
		
		LinearRegression linearRegression = new LinearRegression();
		
		ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
		
		ParamMap[] paramMap = paramGridBuilder.addGrid(linearRegression.regParam(), new double[] {0.01, 0.1, 0.5})
				.addGrid(linearRegression.elasticNetParam(), new double[] {0, 0.5, 1})
				.build();
		
		TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
				.setEstimator(linearRegression)
				.setEvaluator(new RegressionEvaluator().setMetricName("r2"))
				.setEstimatorParamMaps(paramMap)
				.setTrainRatio(0.8);
		
//		TrainValidationSplitModel model = trainValidationSplit.fit(trainAndTestData);
//		LinearRegressionModel lrModel =  (LinearRegressionModel) model.bestModel();
			
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] {conditionIndexer, 
												gradeIndexer,
												zipcodeIndexer,
												wfIndexer,
												encoder,
												vectorAssembler,
												trainValidationSplit});
		
		PipelineModel pipelineModel = pipeline.fit(trainAndTestData);
		
		Dataset<Row> holdoutResults = pipelineModel.transform(holdoutData).drop("predicition");
		
		TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[6];
		LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();
		
		
		System.out.println("traindata R2 = " +lrModel.summary().r2() + " and RMSE = " +lrModel.summary().rootMeanSquaredError());
		System.out.println("testdata R2 = " +lrModel.evaluate(holdoutResults).r2() + " and RMSE = " +lrModel.evaluate(holdoutResults).rootMeanSquaredError());
		System.out.println("coefficients = " + lrModel.coefficients() + " Intercept = " +lrModel.intercept());
		System.out.println("regParam = " + lrModel.getRegParam() + " elasticReg = " + lrModel.getElasticNetParam());
	}

}

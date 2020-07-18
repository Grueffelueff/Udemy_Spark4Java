package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GymCompetitors {

	public static void main(String[] args) {
		
		System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder()
				.appName("GymCompetitors")
				.config("spark.sql.warehouse.dir", "file:///C:/tmp")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("src/main/resources/GymCompetition.csv");
		
//		csvData.show();
		
//		Need 2 colums, one called "label" and one called "features" (Vector (spark != java) of features)
		
		StringIndexer genderIndexer = new StringIndexer()
				.setInputCol("Gender")
				.setOutputCol("GenderIndex");
		
		csvData = genderIndexer.fit(csvData)
				.transform(csvData);
		
		OneHotEncoderEstimator genderEncoder = new OneHotEncoderEstimator()
				.setInputCols(new String[] {"GenderIndex"})
				.setOutputCols(new String[] {"GenderVector"});
		
		csvData = genderEncoder.fit(csvData).transform(csvData);
		
		VectorAssembler vectorAssembler = new VectorAssembler();
		vectorAssembler.setInputCols(new String[] {"Age", "Height", "Weight", "GenderVector"});
		vectorAssembler.setOutputCol("features");
		Dataset<Row> DatawithFeatures = vectorAssembler.transform(csvData);
		
//		DatawithFeatures.show();
		
		Dataset<Row> modelInputData = DatawithFeatures.select("NoOfReps", "features")
		.withColumnRenamed("NoOfReps", "label");
		
		LinearRegression linearRegression = new LinearRegression();
		LinearRegressionModel model = linearRegression.fit(modelInputData);
		System.out.println("The model has Intercept " + model.intercept() + " and coefficients " + model.coefficients());
		
		Dataset<Row> modelledData = model.transform(modelInputData);
		modelledData.show();
	}

}

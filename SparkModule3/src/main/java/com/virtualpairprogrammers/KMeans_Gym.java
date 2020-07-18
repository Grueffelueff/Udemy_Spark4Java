package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KMeans_Gym {
	
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
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] {"GenderVector",
											"Age",
											"Height",
											"NoOfReps"})
				.setOutputCol("features");
		
		Dataset<Row> inputData = assembler.transform(csvData).select("features");
		
		KMeans kMeans = new KMeans()
				.setK(3);
		
		KMeansModel model = kMeans.fit(inputData);
		
		Dataset<Row> predictions = model.transform(inputData);
		
		Vector[] centers = model.clusterCenters();
		for(Vector vf : centers) { System.out.println(vf); };
		
		predictions.groupBy("prediction").count().show();
		
		System.out.println("The SSE-Value is "+ model.computeCost(inputData));
		
		ClusteringEvaluator evaluator = new ClusteringEvaluator();
		
		System.out.println("The Silhouette with squared euclidian distance is " + evaluator.evaluate(inputData));
	
	
	}
		

}

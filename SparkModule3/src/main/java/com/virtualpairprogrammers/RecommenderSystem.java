package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

import java.util.List;

public class RecommenderSystem {
	
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
				.csv("src/main/resources/VVPcourseView.csv");
	
		csvData = csvData.withColumn("proportionWatched", col("proportionWatched").multiply(100));
		
//		csvData.groupBy("userId").pivot("courseId").sum("proportionWatched").show();
		
//		Dataset<Row>[] dataSplit = csvData.randomSplit(new double[] {0.9,0.1});
//		Dataset<Row> traindata = dataSplit[0];
//		Dataset<Row> testdata = dataSplit[1];
		
		ALS als = new ALS()
				.setMaxIter(10)
				.setRegParam(0.1)
				.setUserCol("userId")
				.setItemCol("courseId")
				.setRatingCol("proportionWatched");
		
		ALSModel model = als.fit(csvData);
		model.setColdStartStrategy("drop"); //could also use "NaN" instead of drop
		
		Dataset<Row> userRecs = model.recommendForAllUsers(5);
		
		userRecs.show();
		
		List<Row> userRecsList = userRecs.takeAsList(5);
		
		for(Row r: userRecsList) {
			int userId = r.getAs(0);
			String recs = r.getAs(1).toString();
			System.out.println("For user " + userId + " we might want to recommend " + recs);
			System.out.println("This user has already watched: "+ csvData.filter("userId= "+ userId));
			
		}
	}

}

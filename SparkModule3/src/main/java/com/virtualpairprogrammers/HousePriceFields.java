package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceFields {

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
		
//		csvData.describe().show();
		
		csvData = csvData.drop("id", "date", "waterfront", "view", "condition", "grade", "yr_renovated",
								"zipcode", "lat", "long", "sqft_lot", "sqft_lot15", "yr_built", "sqft_living15");
		
		for(String col : csvData.columns()) {
			for(String col2 : csvData.columns()) {
				System.out.println(col + " has corr " + csvData.stat().corr(col, col2) + " to " + col2);
			}
		}
		
	}

}

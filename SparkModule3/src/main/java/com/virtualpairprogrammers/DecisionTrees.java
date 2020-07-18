package com.virtualpairprogrammers;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.*;

public class DecisionTrees {
	
	public static UDF1<String,String> countryGrouping = new UDF1<String,String>() {

		@Override
		public String call(String country) throws Exception {
			List<String> topCountries =  Arrays.asList(new String[] {"GB","US","IN","UNKNOWN"});
			List<String> europeanCountries =  Arrays.asList(new String[] {"BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE","CH","IS","NO","LI","EU"});
			
			if (topCountries.contains(country)) return country; 
			if (europeanCountries .contains(country)) return "EUROPE";
			else return "OTHER";
		}
		
	};
	
	
	
	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder()
				.appName("CaseStudy")
				.config("spark.sql.warehouse.dir", "file:///C:/tmp")
				.master("local[*]").getOrCreate();
		
		spark.udf().register("countryGrouping", countryGrouping, DataTypes.StringType);
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("src/main/resources/vppFreeTrials.csv")
				.withColumn("country", callUDF("countryGrouping",(col("country"))))
				.withColumn("label", when(col("payments_made").geq(1),lit(1)).otherwise(lit(0)));
		
		StringIndexer countryIndexer = new StringIndexer();
		
		csvData = countryIndexer
				.setInputCol("country")
				.setOutputCol("countryIndex")
				.fit(csvData).transform(csvData);
		
		new IndexToString()
				.setInputCol("countryIndex")
				.setOutputCol("value")
				.transform(csvData.select("countryIndex").distinct()).show();
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] {"countryIndex", 
											"rebill_period", 
											"chapter_access_count", 
											"seconds_watched"})
				.setOutputCol("features");
		
		Dataset<Row> inputData = assembler.transform(csvData).select("label", "features");
		
		Dataset<Row>[] trainingAndHoldout = inputData.randomSplit(new double[] {0.8,0.2});
		Dataset<Row> trainingdata = trainingAndHoldout[0];
		Dataset<Row> testingdata = trainingAndHoldout[1];
		
		DecisionTreeClassifier dtClassifier = new DecisionTreeClassifier()
				.setMaxDepth(3);
		
		DecisionTreeClassificationModel model = dtClassifier.fit(trainingdata);
		Dataset<Row> predictions = model.transform(testingdata);
		
		System.out.println(model.toDebugString());
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setMetricName("accuracy");
		
		System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));
		
		RandomForestClassifier rfClassifier = new RandomForestClassifier()
				.setMaxDepth(3);
		
		RandomForestClassificationModel rfmodel = rfClassifier.fit(trainingdata);
		
		Dataset<Row> rfPredictions = rfmodel.transform(testingdata);
		rfPredictions.show();
		
		System.out.println("The accuracy of the forest model is " + evaluator.evaluate(rfPredictions));
		
		
		
	}

}

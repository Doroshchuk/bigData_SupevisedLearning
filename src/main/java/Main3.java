import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Main3 {
    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "/Users/m/Desktop/hadoop/hadoop-2.7.5/bin/hadoop");
        SparkSession sparkSession = SparkSession
                .builder()
                .appName("LinearRegressionSample")
                .master("local[*]")
                .config("spark.sql.warehouse.dir", "/tmp/")
                .getOrCreate();

        final DataFrameReader dataFrameReader = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true");

        final Dataset<Row> csvDataFrame = dataFrameReader.csv("./assets/BlackFriday.csv").na().drop();

        // Split the data into training and test sets (10% held out for testing)
        Dataset<Row>[] splits = csvDataFrame.randomSplit(new double[]{0.9, 0.1});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];


        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("Age")
                .setOutputCol("indexedLabel")
                .fit(csvDataFrame);

        StringIndexerModel labelIndexerGender = new StringIndexer()
                .setInputCol("Gender")
                .setOutputCol("indexedLabelGender")
                .fit(csvDataFrame);

        StringIndexerModel labelIndexerProduct = new StringIndexer()
                .setInputCol("Product_ID")
                .setOutputCol("indexedLabelProduct_ID")
                .fit(csvDataFrame);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"indexedLabelGender", "indexedLabelProduct_ID", "Product_Category_1", "Product_Category_2", "Product_Category_3", "Purchase"})
                .setOutputCol("categories");

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setMaxDepth(15)
                .setImpurity("entropy") // or "gini"
                .setFeaturesCol("categories")
                .setLabelCol("indexedLabel")
                .setMaxBins(1000);

        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());

        //TrainingData
        Dataset transformed = labelIndexer.transform(trainingData);

        transformed = labelIndexerGender.transform(transformed);

        transformed = labelIndexerProduct.transform(transformed);

        //TestData
        Dataset transformedTestData = labelIndexer.transform(testData);

        transformedTestData = labelIndexerGender.transform(transformedTestData);

        transformedTestData = labelIndexerProduct.transform(transformedTestData);

        transformed.printSchema();

        Dataset features = assembler.transform(transformed);

        Dataset featuresTestData = assembler.transform(transformedTestData);

        DecisionTreeClassificationModel modelDebug = dt.train(features);
        System.out.println("Learnt classification tree model:\n");
        System.out.println(modelDebug.toDebugString());

        Dataset<Row> predictions = modelDebug.setFeaturesCol("categories").transform(featuresTestData);

        predictions = labelConverter.transform(predictions);

//        Pipeline pipeline = new Pipeline()
//                .setStages(new PipelineStage[]{labelIndexer, labelIndexerGender, labelIndexerProduct, assembler, dt, labelConverter});
//        // Train model.
//        PipelineModel model = pipeline.fit(trainingData);
//
//        // Make predictions.
//        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("predictedLabel", "Age", "categories").show(1115);

        // Select (prediction, true label) and compute test error.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

    }
}

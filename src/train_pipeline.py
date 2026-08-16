import os
import argparse
import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, when, avg, lit, concat_ws, abs as spark_abs, unix_timestamp
)
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, SQLTransformer
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

def create_spark_session():
    return SparkSession.builder \
        .appName("NYCTaxiFarePrediction") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def build_baseline(train_df, test_df):
    """
    Computes a naive baseline predictor: mean fare per pickup_zone and pickup_hour.
    Returns the test_df joined with this baseline prediction, filling nulls with the global mean.
    """
    print("Computing baseline...")
    # Add hour to train if not exists
    train_with_hour = train_df.withColumn("pickup_hour", hour(col("pickup_ts")))
    
    # Calculate global mean for fallback
    global_mean = train_df.select(avg(col("fare_amount"))).first()[0]
    
    # Calculate group means
    baseline_model = train_with_hour.groupBy("pickup_zone", "pickup_hour") \
        .agg(avg(col("fare_amount")).alias("prediction"))
        
    # Apply to test
    test_with_hour = test_df.withColumn("pickup_hour", hour(col("pickup_ts")))
    test_baseline = test_with_hour.join(baseline_model, ["pickup_zone", "pickup_hour"], "left")
    
    # Fill missing with global mean
    test_baseline = test_baseline.withColumn(
        "prediction", 
        when(col("prediction").isNull(), lit(global_mean)).otherwise(col("prediction"))
    )
    return test_baseline

def build_ml_pipeline():
    """
    Builds the PySpark MLlib Pipeline for feature engineering and regression.
    """
    # 1. Feature Engineering Transformers
    temporal_transformer = SQLTransformer(
        statement="""
            SELECT *,
                   HOUR(pickup_ts) as pickup_hour,
                   DAYOFWEEK(pickup_ts) as pickup_dow,
                   CASE WHEN DAYOFWEEK(pickup_ts) IN (1, 7) THEN 1 ELSE 0 END as is_weekend,
                   CONCAT(IFNULL(pickup_zone, 'Unknown'), '-', IFNULL(dropoff_zone, 'Unknown')) as zone_pair
            FROM __THIS__
        """
    )
    
    from pyspark.ml.feature import FeatureHasher
    
    # Identify categorical columns (base zones)
    base_cat_cols = ["pickup_borough", "pickup_zone", "dropoff_borough", "dropoff_zone"]
    indexed_cols = [c + "_idx" for c in base_cat_cols]
    encoded_cols = [c + "_ohe" for c in base_cat_cols]
    
    # Clean nulls in categorical cols just to be safe
    string_indexer = StringIndexer(
        inputCols=base_cat_cols, 
        outputCols=indexed_cols, 
        handleInvalid="keep"
    )
    
    ohe = OneHotEncoder(
        inputCols=indexed_cols, 
        outputCols=encoded_cols,
        handleInvalid="keep"
    )
    
    # High cardinality feature: zone_pair (~67k combinations)
    # Using FeatureHasher to avoid 67k dimensional OHE which causes OOM
    hasher = FeatureHasher(
        inputCols=["zone_pair"],
        outputCol="zone_pair_hashed",
        numFeatures=1024
    )
    
    # Numeric features
    numeric_cols = [
        "trip_distance_miles", "passenger_count", "pickup_hour", "pickup_dow", 
        "is_weekend", "trip_seq_in_zone_day", "time_since_prev_pickup_minutes"
    ]
    
    # Assemble all features
    assembler_inputs = encoded_cols + ["zone_pair_hashed"] + numeric_cols
    vector_assembler = VectorAssembler(
        inputCols=assembler_inputs, 
        outputCol="features",
        handleInvalid="skip" # skip nulls in numeric cols
    )
    
    # Regressor
    rf = RandomForestRegressor(
        featuresCol="features", 
        labelCol="fare_amount",
        seed=42
    )
    
    # Assemble Pipeline
    pipeline = Pipeline(stages=[
        temporal_transformer,
        string_indexer,
        ohe,
        hasher,
        vector_assembler,
        rf
    ])
    
    return pipeline, rf

def time_based_split(df, train_ratio=0.8, ts_col="pickup_ts"):
    """
    Splits df on ts_col so every test row's timestamp comes after every train row's.
    The threshold is the approximate train_ratio quantile of ts_col; rows before it
    go to train, rows at or after it go to test. Kept as its own function so the
    no-leakage property (train max < test min) is something a test can pin down
    directly, instead of a claim that only lives in a README.
    """
    ts_seconds = unix_timestamp(col(ts_col)).cast("double")
    split_threshold = df.select(ts_seconds.alias("_ts")).approxQuantile("_ts", [train_ratio], 0.01)[0]

    train_df = df.filter(ts_seconds < split_threshold)
    test_df = df.filter(ts_seconds >= split_threshold)
    return train_df, test_df


def evaluate_predictions(predictions, label_col="fare_amount", pred_col="prediction"):
    """
    Evaluates predictions returning RMSE, MAE, R2.
    """
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def run_pipeline(data_path, fraction=1.0):
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"Reading gold data from {data_path}...")
    df = spark.read.parquet(data_path)
    
    if fraction < 1.0:
        print(f"Sampling {fraction*100}% of data for smoke test...")
        df = df.sample(withReplacement=False, fraction=fraction, seed=42)
        
    print(f"Total rows (after sampling): {df.count()}")
    
    # Drop rows with nulls in numeric cols to prevent assembler failure
    critical_cols = [
        "fare_amount", "trip_distance_miles", "passenger_count", 
        "trip_seq_in_zone_day", "time_since_prev_pickup_minutes", "pickup_ts"
    ]
    df = df.dropna(subset=critical_cols)
    print(f"Rows after dropping nulls in critical numeric columns: {df.count()}")
    
    # Time-based split: hold out the last 20% by time
    print("Performing time-based train/test split...")
    train_df, test_df = time_based_split(df, train_ratio=0.8, ts_col="pickup_ts")

    print(f"Train size: {train_df.count()}, Test size: {test_df.count()}")
    
    # 1. Baseline
    test_baseline = build_baseline(train_df, test_df)
    baseline_metrics = evaluate_predictions(test_baseline)
    print("\n--- Baseline Metrics ---")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 2. ML Pipeline & Tuning
    pipeline, rf = build_ml_pipeline()
    
    # Hyperparameter Tuning using TrainValidationSplit
    # We use TVS instead of CV for faster execution on large data.
    # Grid is intentionally small.
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10]) \
        .addGrid(rf.numTrees, [20]) \
        .build()
        
    evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
    
    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=42
    )
    
    print("\nTraining PySpark ML Pipeline (this may take a while)...")
    model = tvs.fit(train_df)
    
    # 3. Evaluation
    best_model = model.bestModel
    print("\nBest Model Parameters:")
    rf_model = best_model.stages[-1]
    print(f"  Max Depth: {rf_model.getOrDefault('maxDepth')}")
    print(f"  Num Trees: {rf_model.getOrDefault('numTrees')}")
    
    print("Generating predictions on test set...")
    predictions = model.transform(test_df)
    
    model_metrics = evaluate_predictions(predictions)
    print("\n--- Model Metrics ---")
    for k, v in model_metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Error breakdown by borough and hour
    print("\n--- Error Breakdown ---")
    preds_with_error = predictions.withColumn("absolute_error", spark_abs(col("fare_amount") - col("prediction")))
    
    print("Top Boroughs with Highest Average Error:")
    preds_with_error.groupBy("pickup_borough") \
        .agg(avg("absolute_error").alias("avg_abs_error")) \
        .orderBy(col("avg_abs_error").desc()) \
        .show(5, truncate=False)
        
    print("Average Error by Hour of Day:")
    preds_with_error.groupBy("pickup_hour") \
        .agg(avg("absolute_error").alias("avg_abs_error")) \
        .orderBy("pickup_hour") \
        .show(24)
        
    # 4. Reproducibility Check
    print("\nRunning Reproducibility Check...")
    print("Refitting pipeline with the best parameters and same seed...")
    # Recreate pipeline with best params to test determinism
    pipeline_repro, rf_repro = build_ml_pipeline()
    rf_repro.set(rf_repro.maxDepth, rf_model.getOrDefault('maxDepth'))
    rf_repro.set(rf_repro.numTrees, rf_model.getOrDefault('numTrees'))
    
    repro_model = pipeline_repro.fit(train_df)
    repro_preds = repro_model.transform(test_df)
    repro_metrics = evaluate_predictions(repro_preds)
    
    print("Reproducibility Metrics:")
    for k, v in repro_metrics.items():
        print(f"{k}: {v:.4f}")
        
    diff_rmse = abs(model_metrics["RMSE"] - repro_metrics["RMSE"])
    print(f"RMSE Difference between runs: {diff_rmse:.6f}")
    if diff_rmse < 1e-4:
        print("Reproducibility check passed.")
    else:
        print("WARNING: Reproducibility check failed! Output is non-deterministic.")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/gold/yellow_tripdata_gold_2023_01")
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    
    run_pipeline(args.data, args.fraction)

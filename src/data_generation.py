import os
import urllib.request
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, round, when, unix_timestamp,
    row_number, lag, date_format
)
from pyspark.sql.window import Window

def download_data():
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    taxi_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
    zone_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
    
    taxi_file = os.path.join(raw_dir, "yellow_tripdata_2023-01.parquet")
    zone_file = os.path.join(raw_dir, "taxi_zone_lookup.csv")
    
    if not os.path.exists(taxi_file):
        print(f"Downloading {taxi_url}...")
        urllib.request.urlretrieve(taxi_url, taxi_file)
        
    if not os.path.exists(zone_file):
        print(f"Downloading {zone_url}...")
        urllib.request.urlretrieve(zone_url, zone_file)
        
    return taxi_file, zone_file

def generate_gold_table():
    taxi_file, zone_file = download_data()
    
    spark = SparkSession.builder \
        .appName("DataGeneration") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    print("Reading raw data...")
    df = spark.read.parquet(taxi_file)
    zones = spark.read.csv(zone_file, header=True, inferSchema=True)
    
    print("Processing columns...")
    # Rename for consistency
    df = df.withColumnRenamed("tpep_pickup_datetime", "pickup_ts") \
           .withColumnRenamed("tpep_dropoff_datetime", "dropoff_ts") \
           .withColumnRenamed("trip_distance", "trip_distance_miles") \
           .withColumnRenamed("PULocationID", "pickup_location_id") \
           .withColumnRenamed("DOLocationID", "dropoff_location_id")
    
    # Calculate duration
    df = df.withColumn("trip_duration_minutes", 
                       (unix_timestamp("dropoff_ts") - unix_timestamp("pickup_ts")) / 60.0)
    
    # Filter basic anomalies to have clean data for ML
    df = df.filter((col("trip_duration_minutes") > 0) & 
                   (col("trip_distance_miles") > 0) & 
                   (col("fare_amount") >= 0) &
                   (col("fare_amount") < 1000))
                   
    # Calculate implied speed
    df = df.withColumn("implied_speed_mph", 
                       col("trip_distance_miles") / (col("trip_duration_minutes") / 60.0))
                       
    # Join zones for pickup
    zones_pu = zones.withColumnRenamed("LocationID", "pickup_location_id") \
                    .withColumnRenamed("Borough", "pickup_borough") \
                    .withColumnRenamed("Zone", "pickup_zone") \
                    .withColumnRenamed("service_zone", "pickup_service_zone")
                    
    # Join zones for dropoff
    zones_do = zones.withColumnRenamed("LocationID", "dropoff_location_id") \
                    .withColumnRenamed("Borough", "dropoff_borough") \
                    .withColumnRenamed("Zone", "dropoff_zone") \
                    .withColumnRenamed("service_zone", "dropoff_service_zone")
                    
    df = df.join(zones_pu, "pickup_location_id", "left") \
           .join(zones_do, "dropoff_location_id", "left")
           
    # Window features
    print("Calculating window features...")
    # Partition by pickup_zone and day, order by pickup_ts
    df = df.withColumn("pickup_day", date_format("pickup_ts", "yyyy-MM-dd"))
    window_spec = Window.partitionBy("pickup_location_id", "pickup_day").orderBy("pickup_ts")
    
    df = df.withColumn("trip_seq_in_zone_day", row_number().over(window_spec))
    df = df.withColumn("prev_pickup_ts", lag("pickup_ts").over(window_spec))
    df = df.withColumn("time_since_prev_pickup_minutes", 
                       (unix_timestamp("pickup_ts") - unix_timestamp("prev_pickup_ts")) / 60.0)
                       
    # Drop intermediate columns
    df = df.drop("pickup_day", "prev_pickup_ts")
    
    # Select final schema as specified
    gold_cols = [
        "pickup_ts", "dropoff_ts", "trip_distance_miles", "trip_duration_minutes",
        "passenger_count", "fare_amount", "total_amount", "implied_speed_mph",
        "payment_type", "pickup_location_id", "pickup_borough", "pickup_zone",
        "pickup_service_zone", "dropoff_location_id", "dropoff_borough", "dropoff_zone",
        "dropoff_service_zone", "trip_seq_in_zone_day", "time_since_prev_pickup_minutes"
    ]
    
    df_gold = df.select(*[c for c in gold_cols if c in df.columns])
    
    gold_dir = "data/gold/yellow_tripdata_gold_2023_01"
    os.makedirs("data/gold", exist_ok=True)
    
    print(f"Writing gold table to {gold_dir}...")
    df_gold.write.mode("overwrite").parquet(gold_dir)
    print("Data generation complete!")
    
    spark.stop()

if __name__ == "__main__":
    generate_gold_table()

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, to_date, to_timestamp
from datetime import datetime

# --- Configuration ---
HDFS_DATA_DIR = "hdfs:///user/dennislee/telco"
ICEBERG_CATALOG = "hive_catalog"
ICEBERG_DATABASE = "dlee_telco" # Using the telco database

TABLES_TO_PROCESS = [
    ("customers", "customers.csv"),
    ("plans", "plans.csv"),
    ("subscriptions", "subscriptions.csv"),
    ("usage_records", "usage_records.csv"),
    ("recharges", "recharges.csv")
]

# --- Spark Session Initialization for Iceberg ---
spark = SparkSession.builder \
    .appName("AppendTelcoDataToIceberg") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG}", "org.apache.iceberg.spark.SparkCatalog") \
    .config(f"spark.sql.catalog.{ICEBERG_CATALOG}.type", "hive") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session created for Iceberg Append-Only job.")

# --- Data Load and Append Logic ---
try:
    for table_name, csv_file in TABLES_TO_PROCESS:
        full_table_name = f"{ICEBERG_CATALOG}.{ICEBERG_DATABASE}.{table_name}"
        csv_file_path = os.path.join(HDFS_DATA_DIR, csv_file)
        
        print(f"\n--- Processing {table_name} from {csv_file_path} ---")
        
        # Step 1: Read the CSV data. We no longer need dateFormat options.
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(csv_file_path)
            
        # ✅ --- START: EXPLICIT CASTING SECTION ---
        # Step 2: Manually cast the date/timestamp columns for each specific table.
        # This is the most reliable way to ensure correct data types.
        if table_name == "customers":
            df = df.withColumn("registration_date", to_date(col("registration_date"), "yyyy-MM-dd"))
            
        elif table_name == "subscriptions":
            df = df.withColumn("start_date", to_date(col("start_date"), "yyyy-MM-dd"))

        elif table_name == "recharges":
            df = df.withColumn("recharge_date", to_date(col("recharge_date"), "yyyy-MM-dd"))
            
        elif table_name == "usage_records":
            df = df.withColumn("usage_date", to_timestamp(col("usage_date"), "yyyy-MM-dd HH:mm:ss"))
        # ✅ --- END: EXPLICIT CASTING SECTION ---

        print(f"Schema after casting for '{table_name}':")
        df.printSchema()

        print(f"Appending {df.count()} records to Iceberg table: {full_table_name}...")
        
        # Step 3: Append data. The schema will now match perfectly.
        df.writeTo(full_table_name).append()
        
        print(f"✅ Append operation successful for {table_name}.")
        
        final_count = spark.table(full_table_name).count()
        print(f"Total records in {table_name} after append: {final_count}")

except Exception as e:
    print(f"❌ An error occurred during the process: {e}")

finally:
    spark.stop()
    print("\nSpark Session stopped.")
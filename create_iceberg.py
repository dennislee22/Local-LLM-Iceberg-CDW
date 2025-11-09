import os
import random
import pandas as pd
from faker import Faker
from datetime import datetime
from impala.dbapi import connect
from impala.error import Error as ImpalaError

fake = Faker()

IMPALA_HOST = 'coordinator-ares-impala-vw.apps.cdppvc.ares.olympus.cloudera.com'
IMPALA_PORT = 443
USERNAME = 'dennislee'
# IMPORTANT: Use an environment variable for the password for better security.
PASSWORD = os.environ.get('IMPALA_PASSWORD', 'blah')
HTTP_PATH = '/cliservice'
ICEBERG_DATABASE = "dlee_telco"

def generate_customers(num_customers=200):
    """Generates synthetic customer data as a Pandas DataFrame."""
    customers = []
    for i in range(1, num_customers + 1):
        customers.append({
            'customer_id': i,
            'name': fake.name(),
            'email': fake.unique.email(),
            'phone_number': fake.unique.phone_number(),
            'address': fake.address(),
            'registration_date': fake.date_between(start_date='-5y', end_date='today')
        })
    print(f"Generated {num_customers} customers.")
    return pd.DataFrame(customers)

def generate_plans():
    """Generates predefined telco plans as a Pandas DataFrame."""
    plans = [
        {'plan_id': 1, 'plan_name': 'Basic Prepaid', 'plan_type': 'Prepaid', 'monthly_fee': 10.00, 'data_allowance_gb': 5, 'voice_minutes': 100, 'sms_allowance': 50},
        {'plan_id': 2, 'plan_name': 'Standard Prepaid', 'plan_type': 'Prepaid', 'monthly_fee': 20.00, 'data_allowance_gb': 15, 'voice_minutes': 300, 'sms_allowance': 100},
        {'plan_id': 3, 'plan_name': 'Data Hog Prepaid', 'plan_type': 'Prepaid', 'monthly_fee': 35.00, 'data_allowance_gb': 50, 'voice_minutes': 50, 'sms_allowance': 50},
        {'plan_id': 4, 'plan_name': 'Basic Postpaid', 'plan_type': 'Postpaid', 'monthly_fee': 30.00, 'data_allowance_gb': 20, 'voice_minutes': 500, 'sms_allowance': 200},
        {'plan_id': 5, 'plan_name': 'Premium Postpaid', 'plan_type': 'Postpaid', 'monthly_fee': 50.00, 'data_allowance_gb': 100, 'voice_minutes': 1000, 'sms_allowance': 500},
        {'plan_id': 6, 'plan_name': 'Ultimate Postpaid', 'plan_type': 'Postpaid', 'monthly_fee': 80.00, 'data_allowance_gb': None, 'voice_minutes': None, 'sms_allowance': None}
    ]
    print(f"Generated {len(plans)} plans.")
    return pd.DataFrame(plans)

def generate_subscriptions(num_customers=200):
    """Generates synthetic subscription data as a Pandas DataFrame."""
    subscriptions = []
    for i in range(1, num_customers + 1):
        start_date = fake.date_between(start_date='-4y', end_date='-1y')
        status = random.choices(['Active', 'Inactive', 'Suspended'], weights=[8, 1, 1], k=1)[0]
        end_date = None
        if status == 'Inactive':
            end_date = fake.date_between(start_date=start_date, end_date='today')
        
        subscriptions.append({
            'subscription_id': i,
            'customer_id': i,
            'plan_id': random.randint(1, 6),
            'start_date': start_date,
            'end_date': end_date,
            'status': status
        })
    print(f"Generated {num_customers} subscriptions.")
    return pd.DataFrame(subscriptions)

def generate_usage_records(num_records=5000, num_customers=200):
    """Generates synthetic usage data as a Pandas DataFrame."""
    usage_records = []
    customer_ids = list(range(1, num_customers + 1))
    for i in range(1, num_records + 1):
        usage_records.append({
            'usage_id': i,
            'customer_id': random.choice(customer_ids),
            'usage_date': fake.date_time_between(start_date='-1y', end_date='now'),
            'data_used_mb': round(random.uniform(10, 1024), 2),
            'voice_minutes_used': random.randint(0, 60),
            'sms_sent': random.randint(0, 20)
        })
    print(f"Generated {num_records} usage records.")
    return pd.DataFrame(usage_records)

def generate_recharges(subscriptions_df, num_recharges=1000):
    """Generates synthetic recharge data as a Pandas DataFrame."""
    plans_df = generate_plans()
    prepaid_plan_ids = plans_df[plans_df['plan_type'] == 'Prepaid']['plan_id'].tolist()
    prepaid_customer_ids = subscriptions_df[subscriptions_df['plan_id'].isin(prepaid_plan_ids)]['customer_id'].tolist()
    
    if not prepaid_customer_ids:
        print("No prepaid customers found to generate recharges for.")
        return pd.DataFrame()

    recharges = []
    for i in range(1, num_recharges + 1):
        recharges.append({
            'recharge_id': i,
            'customer_id': random.choice(prepaid_customer_ids),
            'recharge_date': fake.date_between(start_date='-1y', end_date='today'),
            'amount': float(random.choice([10, 20, 30, 50, 100])),
            'payment_method': random.choice(['Credit Card', 'Debit Card', 'Online Wallet', 'Voucher'])
        })
    print(f"Generated {num_recharges} recharges.")
    return pd.DataFrame(recharges)

def setup_database_and_tables(cursor):
    """Creates the database and tables for the telco dataset."""
    print(f"Setting up database '{ICEBERG_DATABASE}'...")
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {ICEBERG_DATABASE}")
    cursor.execute(f"USE {ICEBERG_DATABASE}")

    # Drop tables if they exist to ensure a fresh start
    tables = ["customers", "plans", "subscriptions", "usage_records", "recharges"]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    print("Creating Iceberg tables...")
    # Define table schemas
    create_customers_sql = """
    CREATE TABLE customers (
        customer_id INT, name STRING, email STRING, phone_number STRING, 
        address STRING, registration_date DATE
    ) STORED AS ICEBERG
    """
    create_plans_sql = """
    CREATE TABLE plans (
        plan_id INT, plan_name STRING, plan_type STRING, monthly_fee DOUBLE, 
        data_allowance_gb INT, voice_minutes INT, sms_allowance INT
    ) STORED AS ICEBERG
    """
    create_subscriptions_sql = """
    CREATE TABLE subscriptions (
        subscription_id INT, customer_id INT, plan_id INT, 
        start_date DATE, end_date DATE, status STRING
    ) STORED AS ICEBERG
    """
    create_usage_records_sql = """
    CREATE TABLE usage_records (
        usage_id INT, customer_id INT, usage_date TIMESTAMP, data_used_mb DOUBLE, 
        voice_minutes_used INT, sms_sent INT
    ) STORED AS ICEBERG
    """
    create_recharges_sql = """
    CREATE TABLE recharges (
        recharge_id INT, customer_id INT, recharge_date DATE, 
        amount DOUBLE, payment_method STRING
    ) STORED AS ICEBERG
    """
    # Execute create table statements
    cursor.execute(create_customers_sql)
    cursor.execute(create_plans_sql)
    cursor.execute(create_subscriptions_sql)
    cursor.execute(create_usage_records_sql)
    cursor.execute(create_recharges_sql)
    print("All tables created successfully.")

def insert_dataframe(cursor, df, table_name):
    """
    Inserts a Pandas DataFrame into a specified Impala table using a single,
    batched INSERT statement to create only one Iceberg snapshot.
    """
    if df.empty:
        print(f"DataFrame for '{table_name}' is empty. Nothing to insert.")
        return

    print(f"Preparing batch insert for '{table_name}'...")
    # Replace numpy NaN/NaT with None for SQL NULL representation
    df_clean = df.where(pd.notnull(df), None)
    
    # Get column names, ensuring they are properly quoted
    columns = ', '.join(f'`{col}`' for col in df_clean.columns)
    
    # --- Value Formatting ---
    # This section carefully formats each value into a string suitable for a SQL query.
    # It handles NULLs, numbers, and strings (with basic single-quote escaping).
    value_strings = []
    for row in df_clean.itertuples(index=False, name=None):
        formatted_row = []
        for item in row:
            # Use pd.isna() to robustly check for any null-like value (None, np.nan, etc.)
            if pd.isna(item):
                formatted_row.append("NULL")
            # If a float has no fractional part (e.g., 100.0), cast to int before making it a string.
            # This prevents Impala from seeing it as a DECIMAL and causing a casting error.
            elif isinstance(item, float) and item.is_integer():
                formatted_row.append(str(int(item)))
            elif isinstance(item, (int, float)):
                formatted_row.append(str(item))
            else:
                # Escape single quotes for SQL compatibility
                escaped_item = str(item).replace("'", "''")
                formatted_row.append(f"'{escaped_item}'")
        value_strings.append(f"({', '.join(formatted_row)})")

    # Combine all row values into a single string for the VALUES clause
    values_sql = ",\n".join(value_strings)
    
    # Construct the final, single INSERT statement
    query = f"INSERT INTO {table_name} ({columns}) VALUES {values_sql}"
    
    print(f"Executing batch insert of {len(df)} records into '{table_name}'...")
    cursor.execute(query)
    print(f"Finished batch insert into '{table_name}'.")

if __name__ == '__main__':
    conn = None
    try:
        # Establish the connection
        print("Connecting to Impala...")
        conn = connect(
            host=IMPALA_HOST,
            port=IMPALA_PORT,
            user=USERNAME,
            password=PASSWORD,
            auth_mechanism='LDAP',
            use_http_transport=True,
            http_path=HTTP_PATH,
            use_ssl=True
        )
        cursor = conn.cursor()
        print("Connection successful.")

        # Create database and tables
        setup_database_and_tables(cursor)

        # Generate data in Pandas DataFrames
        customers_pd = generate_customers()
        plans_pd = generate_plans()
        subscriptions_pd = generate_subscriptions()
        usage_records_pd = generate_usage_records()
        recharges_pd = generate_recharges(subscriptions_pd)

        # Insert data into Impala
        insert_dataframe(cursor, customers_pd, "customers")
        insert_dataframe(cursor, plans_pd, "plans")
        insert_dataframe(cursor, subscriptions_pd, "subscriptions")
        insert_dataframe(cursor, usage_records_pd, "usage_records")
        insert_dataframe(cursor, recharges_pd, "recharges")

    except ImpalaError as e:
        print(f"An Impala error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\nImpala connection closed.")
        print("Synthetic telco data generation complete.")

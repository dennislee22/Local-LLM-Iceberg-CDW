import os
import random
import pandas as pd
from faker import Faker
from datetime import datetime
from impala.dbapi import connect
from impala.error import Error as ImpalaError

# Initialize Faker to generate synthetic data
fake = Faker()

IMPALA_HOST = 'coordinator-ares-impala-vw.apps.cdppvc.ares.olympus.cloudera.com'
IMPALA_PORT = 443
USERNAME = 'dennislee'
# IMPORTANT: Use an environment variable for the password for better security.
PASSWORD = os.environ.get('IMPALA_PASSWORD', 'blah')
HTTP_PATH = '/cliservice'
ICEBERG_DATABASE = "dlee_telco"


def generate_customers(num_customers=50, start_id=1):
    """Generates new synthetic customer data starting from a specific ID."""
    customers = []
    for i in range(start_id, start_id + num_customers):
        customers.append({
            'customer_id': i,
            'name': fake.name(),
            'email': fake.unique.email(),
            'phone_number': fake.unique.phone_number(),
            'address': fake.address(),
            'registration_date': fake.date_between(start_date='-30d', end_date='today')
        })
    print(f"Generated {num_customers} new customers (starting from ID {start_id}).")
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
    return pd.DataFrame(plans)

def generate_subscriptions(num_customers=50, start_cust_id=1, start_sub_id=1):
    """Generates new synthetic subscription data."""
    subscriptions = []
    for i in range(num_customers):
        customer_id = start_cust_id + i
        subscription_id = start_sub_id + i
        start_date = fake.date_between(start_date='-30d', end_date='today')
        
        subscriptions.append({
            'subscription_id': subscription_id,
            'customer_id': customer_id,
            'plan_id': random.randint(1, 6),
            'start_date': start_date,
            'end_date': None,
            'status': 'Active' # New subscriptions are always active
        })
    print(f"Generated {num_customers} new subscriptions (starting from ID {start_sub_id}).")
    return pd.DataFrame(subscriptions)

def generate_usage_records(num_records=1000, customer_ids=[], start_id=1):
    """Generates new synthetic usage data for a list of customers."""
    if not customer_ids:
        return pd.DataFrame()
        
    usage_records = []
    for i in range(start_id, start_id + num_records):
        usage_records.append({
            'usage_id': i,
            'customer_id': random.choice(customer_ids),
            'usage_date': fake.date_time_between(start_date='-7d', end_date='now'),
            'data_used_mb': round(random.uniform(10, 1024), 2),
            'voice_minutes_used': random.randint(0, 60),
            'sms_sent': random.randint(0, 20)
        })
    print(f"Generated {num_records} new usage records (starting from ID {start_id}).")
    return pd.DataFrame(usage_records)

def generate_recharges(subscriptions_df, num_recharges=200, start_id=1):
    """Generates new synthetic recharge data."""
    plans_df = generate_plans()
    prepaid_plan_ids = plans_df[plans_df['plan_type'] == 'Prepaid']['plan_id'].tolist()
    prepaid_customer_ids = subscriptions_df[subscriptions_df['plan_id'].isin(prepaid_plan_ids)]['customer_id'].tolist()
    
    if not prepaid_customer_ids:
        print("No new prepaid customers to generate recharges for.")
        return pd.DataFrame()

    recharges = []
    for i in range(start_id, start_id + num_recharges):
        recharges.append({
            'recharge_id': i,
            'customer_id': random.choice(prepaid_customer_ids),
            'recharge_date': fake.date_between(start_date='-7d', end_date='today'),
            'amount': float(random.choice([10, 20, 30, 50, 100])),
            'payment_method': random.choice(['Credit Card', 'Debit Card', 'Online Wallet', 'Voucher'])
        })
    print(f"Generated {num_recharges} new recharges (starting from ID {start_id}).")
    return pd.DataFrame(recharges)

def get_max_ids(cursor):
    """Fetches the maximum ID from each table to ensure new data is unique."""
    print("Fetching maximum existing IDs from tables...")
    ids = {}
    tables_with_ids = {
        'customers': 'customer_id',
        'subscriptions': 'subscription_id',
        'usage_records': 'usage_id',
        'recharges': 'recharge_id'
    }
    for table, id_column in tables_with_ids.items():
        try:
            cursor.execute(f"SELECT MAX({id_column}) FROM {table}")
            max_id = cursor.fetchone()[0]
            ids[table] = max_id if max_id is not None else 0
        except ImpalaError as e:
            print(f"Could not fetch max ID for {table}, assuming 0. Error: {e}")
            ids[table] = 0
    print(f"Current max IDs: {ids}")
    return ids

def insert_dataframe(cursor, df, table_name):
    """Inserts a Pandas DataFrame into a specified Impala table using a single batch INSERT."""
    if df.empty:
        print(f"DataFrame for '{table_name}' is empty. Nothing to insert.")
        return

    print(f"Preparing batch insert for '{table_name}'...")
    df_clean = df.where(pd.notnull(df), None)
    columns = ', '.join(f'`{col}`' for col in df_clean.columns)
    
    value_strings = []
    for row in df_clean.itertuples(index=False, name=None):
        formatted_row = []
        for item in row:
            if pd.isna(item):
                formatted_row.append("NULL")
            elif isinstance(item, float) and item.is_integer():
                formatted_row.append(str(int(item)))
            elif isinstance(item, (int, float)):
                formatted_row.append(str(item))
            else:
                escaped_item = str(item).replace("'", "''")
                formatted_row.append(f"'{escaped_item}'")
        value_strings.append(f"({', '.join(formatted_row)})")

    values_sql = ",\n".join(value_strings)
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

        # Ensure we are using the correct database
        cursor.execute(f"USE {ICEBERG_DATABASE}")

        # Get the current maximum IDs to start generating new data from
        max_ids = get_max_ids(cursor)
        
        # Define how much new data to generate
        NUM_NEW_CUSTOMERS = 50
        NUM_NEW_USAGE_RECORDS = 1000
        NUM_NEW_RECHARGES = 200

        # Generate new data starting after the max IDs
        new_customers_pd = generate_customers(
            num_customers=NUM_NEW_CUSTOMERS, 
            start_id=max_ids['customers'] + 1
        )
        new_subscriptions_pd = generate_subscriptions(
            num_customers=NUM_NEW_CUSTOMERS, 
            start_cust_id=max_ids['customers'] + 1,
            start_sub_id=max_ids['subscriptions'] + 1
        )
        # Generate usage for all customers, old and new
        cursor.execute("SELECT customer_id FROM customers")
        all_customer_ids = [row[0] for row in cursor.fetchall()]
        
        new_usage_records_pd = generate_usage_records(
            num_records=NUM_NEW_USAGE_RECORDS,
            customer_ids=all_customer_ids,
            start_id=max_ids['usage_records'] + 1
        )
        new_recharges_pd = generate_recharges(
            subscriptions_df=new_subscriptions_pd,
            num_recharges=NUM_NEW_RECHARGES,
            start_id=max_ids['recharges'] + 1
        )

        # Insert the new data into Impala
        insert_dataframe(cursor, new_customers_pd, "customers")
        insert_dataframe(cursor, new_subscriptions_pd, "subscriptions")
        insert_dataframe(cursor, new_usage_records_pd, "usage_records")
        insert_dataframe(cursor, new_recharges_pd, "recharges")

    except ImpalaError as e:
        print(f"An Impala error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\nImpala connection closed.")
        print("Script to append telco data has completed.")

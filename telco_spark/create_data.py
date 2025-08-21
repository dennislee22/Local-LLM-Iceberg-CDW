import os
import random
import pandas as pd
from faker import Faker
from datetime import date

fake = Faker()

OUTPUT_DIR = "telco_data"
NUM_CUSTOMERS = 100
NUM_USAGE_RECORDS = 2000
NUM_RECHARGES = 300

def generate_customers(num_records=100):
    """Generates synthetic customer data."""
    customers = []
    for i in range(1, num_records + 1):
        customers.append({
            'customer_id': i,
            'name': fake.name(),
            'email': fake.unique.email(),
            'phone_number': fake.unique.phone_number(),
            'address': fake.address().replace('\n', ', '),
            'registration_date': fake.date_between(start_date='-2y', end_date='today')
        })
    print(f"Generated {num_records} customers.")
    return pd.DataFrame(customers)

def generate_plans():
    """Generates predefined telco plans."""
    plans = [
        {'plan_id': 1, 'plan_name': 'Basic Prepaid', 'plan_type': 'Prepaid', 'monthly_fee': 10.00, 'data_allowance_gb': 5, 'voice_minutes': 100, 'sms_allowance': 50},
        {'plan_id': 2, 'plan_name': 'Standard Prepaid', 'plan_type': 'Prepaid', 'monthly_fee': 20.00, 'data_allowance_gb': 15, 'voice_minutes': 300, 'sms_allowance': 100},
        {'plan_id': 3, 'plan_name': 'Data Hog Prepaid', 'plan_type': 'Prepaid', 'monthly_fee': 35.00, 'data_allowance_gb': 50, 'voice_minutes': 50, 'sms_allowance': 50},
        {'plan_id': 4, 'plan_name': 'Basic Postpaid', 'plan_type': 'Postpaid', 'monthly_fee': 30.00, 'data_allowance_gb': 20, 'voice_minutes': 500, 'sms_allowance': 200},
        {'plan_id': 5, 'plan_name': 'Premium Postpaid', 'plan_type': 'Postpaid', 'monthly_fee': 50.00, 'data_allowance_gb': 100, 'voice_minutes': 1000, 'sms_allowance': 500},
        {'plan_id': 6, 'plan_name': 'Ultimate Postpaid', 'plan_type': 'Postpaid', 'monthly_fee': 80.00, 'data_allowance_gb': None, 'voice_minutes': None, 'sms_allowance': None} # Representing unlimited
    ]
    return pd.DataFrame(plans)

def generate_subscriptions(num_customers=100):
    """Generates synthetic subscription data."""
    subscriptions = []
    for i in range(1, num_customers + 1):
        subscriptions.append({
            'subscription_id': 1000 + i,
            'customer_id': i,
            'plan_id': random.randint(1, 6),
            'start_date': fake.date_between(start_date='-1y', end_date='today'),
            'end_date': None,
            'status': 'Active'
        })
    print(f"Generated {num_customers} subscriptions.")
    return pd.DataFrame(subscriptions)

def generate_usage_records(num_records=2000, num_customers=100):
    """Generates synthetic usage data."""
    usage_records = []
    for i in range(1, num_records + 1):
        usage_records.append({
            'usage_id': 20000 + i,
            'customer_id': random.randint(1, num_customers),
            'usage_date': fake.date_time_between(start_date='-30d', end_date='now').strftime('%Y-%m-%d %H:%M:%S'),
            'data_used_mb': round(random.uniform(10, 1024), 2),
            'voice_minutes_used': random.randint(0, 60),
            'sms_sent': random.randint(0, 20)
        })
    print(f"Generated {num_records} usage records.")
    return pd.DataFrame(usage_records)

def generate_recharges(subscriptions_df, plans_df, num_records=300):
    """Generates synthetic recharge data for prepaid customers."""
    prepaid_plan_ids = plans_df[plans_df['plan_type'] == 'Prepaid']['plan_id'].tolist()
    prepaid_customer_ids = subscriptions_df[subscriptions_df['plan_id'].isin(prepaid_plan_ids)]['customer_id'].tolist()
    
    if not prepaid_customer_ids:
        return pd.DataFrame()

    recharges = []
    for i in range(1, num_records + 1):
        recharges.append({
            'recharge_id': 5000 + i,
            'customer_id': random.choice(prepaid_customer_ids),
            'recharge_date': fake.date_between(start_date='-30d', end_date='today'),
            'amount': float(random.choice([10, 20, 30, 50, 100])),
            'payment_method': random.choice(['Credit Card', 'Online Wallet', 'Voucher'])
        })
    print(f"Generated {num_records} recharges.")
    return pd.DataFrame(recharges)

if __name__ == '__main__':
    customers_df = generate_customers(NUM_CUSTOMERS)
    plans_df = generate_plans()
    subscriptions_df = generate_subscriptions(NUM_CUSTOMERS)
    usage_records_df = generate_usage_records(NUM_USAGE_RECORDS, NUM_CUSTOMERS)
    recharges_df = generate_recharges(subscriptions_df, plans_df, NUM_RECHARGES)

    customers_df.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    plans_df.to_csv(os.path.join(OUTPUT_DIR, "plans.csv"), index=False)
    subscriptions_df.to_csv(os.path.join(OUTPUT_DIR, "subscriptions.csv"), index=False)
    usage_records_df.to_csv(os.path.join(OUTPUT_DIR, "usage_records.csv"), index=False)
    recharges_df.to_csv(os.path.join(OUTPUT_DIR, "recharges.csv"), index=False)
    
    print(f"\nAll data saved to CSV files in the '{OUTPUT_DIR}' directory.")

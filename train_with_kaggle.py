
import os
import sys
import json
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random

import numpy as np
import pandas as pd
import torch
import kagglehub

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'behavior-analysis-service'))

# Setup Django (dummy settings if needed)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'behavior_service.settings')

try:
    import django
    from django.conf import settings
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            DATABASES={},
            INSTALLED_APPS=['app'],
            CUSTOMER_SERVICE_URL='http://customer-service:8000',
            ORDER_SERVICE_URL='http://order-service:8000',
            COMMENT_RATE_SERVICE_URL='http://comment-rate-service:8000',
        )
    django.setup()
except Exception as e:
    print(f"Django setup skipped or failed: {e}")

from app.training.train import Trainer
from app.ml_models.data_processor import DataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KaggleDataMapper:
    """Maps Kaggle Ecommerce Customer Behavior Dataset to internal schema"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.categories = [
            'Technology', 'Fiction', 'Science', 'Business', 'Self-Help',
            'History', 'Arts', 'Education', 'Health', 'Travel'
        ]
        self.jobs = [
            'Software Engineer', 'Data Scientist', 'Teacher', 'Doctor', 
            'Business Owner', 'Student', 'Marketing Manager', 'Accountant',
            'Designer', 'Sales Representative', 'Freelancer', 'Retired'
        ]

    def map_to_internal(self) -> Dict[str, pd.DataFrame]:
        logger.info("Mapping Kaggle data to internal schema...")
        
        # 1. Customers
        customers = []
        for i, row in self.df.iterrows():
            customer_id = i + 1
            
            # Segment logic
            if row['Churned'] == 1:
                segment = 'Churned'
            elif row['Lifetime_Value'] > self.df['Lifetime_Value'].quantile(0.8):
                segment = 'VIP'
            elif row['Total_Purchases'] < 2:
                segment = 'New'
            else:
                segment = 'Regular'
                
            customers.append({
                'customer_id': customer_id,
                'age': row['Age'] if not pd.isna(row['Age']) else 35,
                'gender': 'M' if row['Gender'] == 'Male' else 'F',
                'job': random.choice(self.jobs),
                'location': f"{row['City']}, {row['Country']}",
                'registration_date': datetime.now() - timedelta(days=int(row['Membership_Years'] * 365)),
                'segment': segment,
                'email_verified': True,
                'phone_verified': True
            })
        customers_df = pd.DataFrame(customers)
        
        # 2. Orders
        orders = []
        order_id = 1
        for i, row in self.df.iterrows():
            customer_id = i + 1
            n_purchases = int(row['Total_Purchases'])
            avg_val = row['Average_Order_Value']
            days_since_last = int(row['Days_Since_Last_Purchase']) if not pd.isna(row['Days_Since_Last_Purchase']) else 30
            
            reg_date = customers[i]['registration_date']
            total_days = (datetime.now() - reg_date).days
            
            for j in range(n_purchases):
                # Distribute orders between registration and last purchase
                if total_days > days_since_last:
                    order_days_ago = random.randint(days_since_last, total_days)
                else:
                    order_days_ago = days_since_last
                    
                orders.append({
                    'order_id': order_id,
                    'customer_id': customer_id,
                    'order_date': datetime.now() - timedelta(days=order_days_ago),
                    'total_amount': max(0, np.random.normal(avg_val, avg_val * 0.2)),
                    'category': random.choice(self.categories),
                    'status': 'completed'
                })
                order_id += 1
        orders_df = pd.DataFrame(orders)
        
        # 3. Reviews
        reviews = []
        review_id = 1
        for i, row in self.df.iterrows():
            customer_id = i + 1
            n_reviews = int(row['Product_Reviews_Written']) if not pd.isna(row['Product_Reviews_Written']) else 0
            for j in range(n_reviews):
                reviews.append({
                    'review_id': review_id,
                    'customer_id': customer_id,
                    'rating': random.randint(3, 5) if row['Churned'] == 0 else random.randint(1, 4),
                    'review_date': datetime.now() - timedelta(days=random.randint(1, 365))
                })
                review_id += 1
        reviews_df = pd.DataFrame(reviews)
        
        # 4. Events
        events = []
        event_id = 1
        for i, row in self.df.iterrows():
            customer_id = i + 1
            # Login events
            n_logins = int(row['Login_Frequency'] * 30) # approx per month
            for _ in range(min(100, n_logins)):
                events.append({
                    'event_id': event_id,
                    'customer_id': customer_id,
                    'event_type': 'login',
                    'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
                    'category': 'System'
                })
                event_id += 1
            
            # View events based on pages per session
            n_views = int(row['Pages_Per_Session'] * row['Login_Frequency'])
            for _ in range(min(50, n_views)):
                events.append({
                    'event_id': event_id,
                    'customer_id': customer_id,
                    'event_type': 'product_view',
                    'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
                    'category': random.choice(self.categories)
                })
                event_id += 1
        events_df = pd.DataFrame(events)
        
        return {
            'customers': customers_df,
            'orders': orders_df,
            'reviews': reviews_df,
            'events': events_df
        }

class KaggleTrainer(Trainer):
    """Trainer that uses Kaggle dataset"""
    
    def load_data(self):
        logger.info("Downloading Kaggle dataset...")
        path = kagglehub.dataset_download("dhairyajeetsingh/ecommerce-customer-behavior-dataset")
        csv_path = Path(path) / 'ecommerce_customer_churn_dataset.csv'
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded Kaggle dataset with {len(df)} records")
        
        # Take a subset if too large for quick training (e.g., 10000)
        if len(df) > 10000:
            df = df.sample(10000, random_state=42).reset_index(drop=True)
            logger.info("Sampled 10000 records for training")
        
        # Fill NaNs
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Pages_Per_Session'] = df['Pages_Per_Session'].fillna(df['Pages_Per_Session'].median())
        df['Login_Frequency'] = df['Login_Frequency'].fillna(df['Login_Frequency'].median())
        df['Average_Order_Value'] = df['Average_Order_Value'].fillna(df['Average_Order_Value'].median())
        df['Total_Purchases'] = df['Total_Purchases'].fillna(0)
        df['Days_Since_Last_Purchase'] = df['Days_Since_Last_Purchase'].fillna(30)
        df['Membership_Years'] = df['Membership_Years'].fillna(df['Membership_Years'].median())
        df['Product_Reviews_Written'] = df['Product_Reviews_Written'].fillna(0)
            
        mapper = KaggleDataMapper(df)
        return mapper.map_to_internal()

def main():
    # Set model directory relative to behavior-analysis-service
    model_dir = Path('behavior-analysis-service/data/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = KaggleTrainer(
        model_dir=str(model_dir),
        epochs=20, # Keep it short for demonstration
        batch_size=64,
        learning_rate=1e-3
    )
    
    metrics = trainer.train()
    logger.info(f"Training completed with metrics: {metrics}")

if __name__ == '__main__':
    main()

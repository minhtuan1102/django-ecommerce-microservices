"""
Data Collector Module
Thu thập dữ liệu từ các microservices và tạo synthetic data
"""
import requests
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """Thu thập dữ liệu từ các microservices"""
    
    def __init__(self):
        self.customer_service_url = getattr(settings, 'CUSTOMER_SERVICE_URL', 'http://customer-service:8000')
        self.order_service_url = getattr(settings, 'ORDER_SERVICE_URL', 'http://order-service:8000')
        self.comment_rate_service_url = getattr(settings, 'COMMENT_RATE_SERVICE_URL', 'http://comment-rate-service:8000')
        self.timeout = 10
        
    def fetch_customers(self) -> List[Dict]:
        """Fetch customer data từ Customer Service"""
        try:
            response = requests.get(
                f"{self.customer_service_url}/customers/",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'results' in data:
                    return data['results']
                return data if isinstance(data, list) else []
            logger.warning(f"Customer service returned status {response.status_code}")
            return []
        except requests.RequestException as e:
            logger.error(f"Error fetching customers: {e}")
            return []
    
    def fetch_orders(self) -> List[Dict]:
        """Fetch orders từ Order Service"""
        try:
            response = requests.get(
                f"{self.order_service_url}/orders/",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'results' in data:
                    return data['results']
                return data if isinstance(data, list) else []
            logger.warning(f"Order service returned status {response.status_code}")
            return []
        except requests.RequestException as e:
            logger.error(f"Error fetching orders: {e}")
            return []
    
    def fetch_reviews(self) -> List[Dict]:
        """Fetch reviews từ Comment Rate Service"""
        try:
            response = requests.get(
                f"{self.comment_rate_service_url}/reviews/",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'results' in data:
                    return data['results']
                return data if isinstance(data, list) else []
            logger.warning(f"Comment service returned status {response.status_code}")
            return []
        except requests.RequestException as e:
            logger.error(f"Error fetching reviews: {e}")
            return []
    
    def fetch_all_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Fetch tất cả dữ liệu từ các services"""
        customers = self.fetch_customers()
        orders = self.fetch_orders()
        reviews = self.fetch_reviews()
        return customers, orders, reviews


class SyntheticDataGenerator:
    """Tạo synthetic data cho training khi không có dữ liệu thật"""
    
    # Định nghĩa các constants
    JOBS = [
        'Software Engineer', 'Data Scientist', 'Teacher', 'Doctor', 
        'Business Owner', 'Student', 'Marketing Manager', 'Accountant',
        'Designer', 'Sales Representative', 'Freelancer', 'Retired'
    ]
    
    CATEGORIES = [
        'Technology', 'Fiction', 'Science', 'Business', 'Self-Help',
        'History', 'Arts', 'Education', 'Health', 'Travel'
    ]
    
    LOCATIONS = [
        'Ha Noi', 'Ho Chi Minh', 'Da Nang', 'Hai Phong', 'Can Tho',
        'Nha Trang', 'Hue', 'Vung Tau', 'Bien Hoa', 'Other'
    ]
    
    SEGMENTS = ['VIP', 'Regular', 'New', 'Churned']
    
    EVENT_TYPES = [
        'page_view', 'product_view', 'add_to_cart', 'purchase',
        'search', 'review', 'wishlist_add', 'login', 'logout'
    ]
    
    def __init__(self, n_customers: int = 1000, random_seed: int = 42):
        self.n_customers = n_customers
        np.random.seed(random_seed)
        random.seed(random_seed)
        
    def generate_customers(self) -> pd.DataFrame:
        """Tạo synthetic customer data"""
        customers = []
        
        for i in range(self.n_customers):
            # Xác định segment trước để tạo data consistent
            segment_probs = [0.1, 0.5, 0.25, 0.15]  # VIP, Regular, New, Churned
            segment = np.random.choice(self.SEGMENTS, p=segment_probs)
            
            # Registration date dựa trên segment
            if segment == 'New':
                days_since_registration = random.randint(1, 60)
            elif segment == 'Churned':
                days_since_registration = random.randint(180, 730)
            else:
                days_since_registration = random.randint(30, 500)
            
            registration_date = datetime.now() - timedelta(days=days_since_registration)
            
            # Age distribution
            age = max(18, min(80, int(np.random.normal(35, 12))))
            
            customer = {
                'customer_id': i + 1,
                'age': age,
                'gender': random.choice(['M', 'F']),
                'job': random.choice(self.JOBS),
                'location': random.choice(self.LOCATIONS),
                'registration_date': registration_date,
                'segment': segment,
                'email_verified': random.random() > 0.1,
                'phone_verified': random.random() > 0.2,
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_orders(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Tạo synthetic order data dựa trên customer segments"""
        orders = []
        order_id = 1
        
        for _, customer in customers_df.iterrows():
            segment = customer['segment']
            customer_id = customer['customer_id']
            registration_date = customer['registration_date']
            
            # Số đơn hàng dựa trên segment
            if segment == 'VIP':
                n_orders = random.randint(15, 50)
            elif segment == 'Regular':
                n_orders = random.randint(3, 15)
            elif segment == 'New':
                n_orders = random.randint(0, 3)
            else:  # Churned
                n_orders = random.randint(1, 5)
            
            # Thời gian đơn hàng cuối cùng
            if segment == 'Churned':
                last_order_days_ago = random.randint(90, 365)
            elif segment == 'VIP':
                last_order_days_ago = random.randint(1, 14)
            elif segment == 'Regular':
                last_order_days_ago = random.randint(7, 45)
            else:  # New
                last_order_days_ago = random.randint(0, 30)
            
            days_since_registration = (datetime.now() - registration_date).days
            
            for j in range(n_orders):
                # Order date giữa registration và last order
                max_days = max(1, days_since_registration - last_order_days_ago)
                order_days_ago = last_order_days_ago + random.randint(0, max_days)
                order_date = datetime.now() - timedelta(days=order_days_ago)
                
                # Giá trị đơn hàng dựa trên segment
                if segment == 'VIP':
                    amount = max(50000, np.random.normal(500000, 200000))
                elif segment == 'Regular':
                    amount = max(30000, np.random.normal(200000, 100000))
                else:
                    amount = max(20000, np.random.normal(150000, 80000))
                
                n_items = random.randint(1, 5)
                
                order = {
                    'order_id': order_id,
                    'customer_id': customer_id,
                    'order_date': order_date,
                    'total_amount': round(amount, 0),
                    'n_items': n_items,
                    'category': random.choice(self.CATEGORIES),
                    'status': random.choices(
                        ['completed', 'pending', 'cancelled'],
                        weights=[0.85, 0.10, 0.05]
                    )[0],
                    'payment_method': random.choice(['credit_card', 'bank_transfer', 'cod', 'e_wallet']),
                    'discount_applied': random.random() > 0.7,
                }
                orders.append(order)
                order_id += 1
        
        return pd.DataFrame(orders)
    
    def generate_reviews(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Tạo synthetic review data"""
        reviews = []
        review_id = 1
        
        completed_orders = orders_df[orders_df['status'] == 'completed']
        
        for _, order in completed_orders.iterrows():
            # 30% khách hàng để lại review
            if random.random() > 0.3:
                continue
            
            # Rating distribution
            rating = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.05, 0.05, 0.15, 0.35, 0.40]
            )[0]
            
            review_date = order['order_date'] + timedelta(days=random.randint(1, 14))
            
            review = {
                'review_id': review_id,
                'customer_id': order['customer_id'],
                'order_id': order['order_id'],
                'rating': rating,
                'review_date': review_date,
                'helpful_votes': random.randint(0, 20) if random.random() > 0.5 else 0,
                'has_text': random.random() > 0.4,
            }
            reviews.append(review)
            review_id += 1
        
        return pd.DataFrame(reviews) if reviews else pd.DataFrame(columns=[
            'review_id', 'customer_id', 'order_id', 'rating', 
            'review_date', 'helpful_votes', 'has_text'
        ])
    
    def generate_behavior_events(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Tạo synthetic behavior events"""
        events = []
        event_id = 1
        
        for _, customer in customers_df.iterrows():
            segment = customer['segment']
            customer_id = customer['customer_id']
            
            # Số events dựa trên segment
            if segment == 'VIP':
                n_events = random.randint(100, 300)
            elif segment == 'Regular':
                n_events = random.randint(30, 100)
            elif segment == 'New':
                n_events = random.randint(5, 30)
            else:  # Churned
                n_events = random.randint(10, 50)
            
            # Event distribution
            if segment == 'VIP':
                event_weights = [0.15, 0.25, 0.15, 0.20, 0.08, 0.05, 0.05, 0.04, 0.03]
            elif segment == 'Churned':
                event_weights = [0.30, 0.20, 0.05, 0.02, 0.15, 0.01, 0.02, 0.15, 0.10]
            else:
                event_weights = [0.25, 0.25, 0.10, 0.10, 0.10, 0.03, 0.05, 0.07, 0.05]
            
            for _ in range(n_events):
                days_ago = random.randint(0, 365)
                event_date = datetime.now() - timedelta(
                    days=days_ago, 
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                event_type = random.choices(self.EVENT_TYPES, weights=event_weights)[0]
                
                event = {
                    'event_id': event_id,
                    'customer_id': customer_id,
                    'event_type': event_type,
                    'timestamp': event_date,
                    'category': random.choice(self.CATEGORIES),
                    'session_id': f"sess_{customer_id}_{days_ago}_{random.randint(1, 5)}",
                    'device': random.choice(['desktop', 'mobile', 'tablet']),
                }
                events.append(event)
                event_id += 1
        
        return pd.DataFrame(events)
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Tạo tất cả synthetic data"""
        logger.info(f"Generating synthetic data for {self.n_customers} customers...")
        
        customers_df = self.generate_customers()
        logger.info(f"Generated {len(customers_df)} customers")
        
        orders_df = self.generate_orders(customers_df)
        logger.info(f"Generated {len(orders_df)} orders")
        
        reviews_df = self.generate_reviews(orders_df)
        logger.info(f"Generated {len(reviews_df)} reviews")
        
        events_df = self.generate_behavior_events(customers_df)
        logger.info(f"Generated {len(events_df)} behavior events")
        
        return {
            'customers': customers_df,
            'orders': orders_df,
            'reviews': reviews_df,
            'events': events_df
        }


def collect_or_generate_data(use_synthetic: bool = True, n_customers: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    Thu thập dữ liệu từ services hoặc generate synthetic data
    
    Args:
        use_synthetic: Nếu True, sử dụng synthetic data
        n_customers: Số lượng customers khi generate synthetic data
        
    Returns:
        Dictionary chứa các DataFrames
    """
    if use_synthetic:
        generator = SyntheticDataGenerator(n_customers=n_customers)
        return generator.generate_all_data()
    
    # Thu thập từ services
    collector = DataCollector()
    customers, orders, reviews = collector.fetch_all_data()
    
    # Nếu không có data, fallback to synthetic
    if not customers or not orders:
        logger.warning("No data from services, falling back to synthetic data")
        generator = SyntheticDataGenerator(n_customers=n_customers)
        return generator.generate_all_data()
    
    return {
        'customers': pd.DataFrame(customers),
        'orders': pd.DataFrame(orders),
        'reviews': pd.DataFrame(reviews) if reviews else pd.DataFrame(),
        'events': pd.DataFrame()  # Events không có từ services
    }

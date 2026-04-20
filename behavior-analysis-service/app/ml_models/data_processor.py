"""
Data Processor Module
Preprocessing, Feature Engineering và Sequence Preparation
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """Encode categorical features"""
    
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str]):
        """Fit encoders và scaler"""
        # Fit label encoders cho categorical columns
        for col in categorical_cols:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                # Handle missing values
                values = df[col].fillna('unknown').astype(str)
                self.encoders[col].fit(values)
        
        # Fit scaler cho numerical columns
        numerical_data = []
        for col in numerical_cols:
            if col in df.columns:
                numerical_data.append(df[col].fillna(0).values.reshape(-1, 1))
        
        if numerical_data:
            combined = np.hstack(numerical_data)
            self.scaler.fit(combined)
        
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.fitted = True
        
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        # Transform categorical
        categorical_encoded = []
        for col in self.categorical_cols:
            if col in df.columns:
                values = df[col].fillna('unknown').astype(str)
                # Handle unseen labels
                encoded = np.zeros(len(values), dtype=np.int64)
                for i, v in enumerate(values):
                    try:
                        encoded[i] = self.encoders[col].transform([v])[0]
                    except ValueError:
                        encoded[i] = 0  # Unknown category
                categorical_encoded.append(encoded.reshape(-1, 1))
        
        categorical_array = np.hstack(categorical_encoded) if categorical_encoded else np.array([])
        
        # Transform numerical
        numerical_data = []
        for col in self.numerical_cols:
            if col in df.columns:
                numerical_data.append(df[col].fillna(0).values.reshape(-1, 1))
        
        if numerical_data:
            combined = np.hstack(numerical_data)
            numerical_array = self.scaler.transform(combined)
        else:
            numerical_array = np.array([])
        
        return categorical_array, numerical_array
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes cho embedding layers"""
        return {col: len(enc.classes_) + 1 for col, enc in self.encoders.items()}


class RFMCalculator:
    """Tính toán RFM (Recency, Frequency, Monetary) features"""
    
    def __init__(self, reference_date: Optional[datetime] = None):
        self.reference_date = reference_date or datetime.now()
        
    def calculate(self, orders_df: pd.DataFrame, customer_id_col: str = 'customer_id',
                 date_col: str = 'order_date', amount_col: str = 'total_amount') -> pd.DataFrame:
        """
        Tính RFM features cho mỗi customer
        
        Returns:
            DataFrame với columns: customer_id, recency, frequency, monetary, rfm_score
        """
        if orders_df.empty:
            return pd.DataFrame(columns=[customer_id_col, 'recency', 'frequency', 'monetary', 'rfm_score'])
        
        # Ensure datetime
        orders_df = orders_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(orders_df[date_col]):
            orders_df[date_col] = pd.to_datetime(orders_df[date_col])
        
        # Group by customer
        rfm = orders_df.groupby(customer_id_col).agg({
            date_col: lambda x: (self.reference_date - x.max()).days,  # Recency
            'order_id': 'count',  # Frequency (dùng order_id để count)
            amount_col: 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
        
        # Handle edge cases
        rfm['recency'] = rfm['recency'].clip(lower=0)
        rfm['frequency'] = rfm['frequency'].clip(lower=1)
        rfm['monetary'] = rfm['monetary'].clip(lower=0)
        
        # Calculate RFM scores (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(float)
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(float)
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(float)
        
        # Fill NaN scores with median
        rfm['r_score'] = rfm['r_score'].fillna(3)
        rfm['f_score'] = rfm['f_score'].fillna(3)
        rfm['m_score'] = rfm['m_score'].fillna(3)
        
        # Combined RFM score
        rfm['rfm_score'] = rfm['r_score'] * 100 + rfm['f_score'] * 10 + rfm['m_score']
        
        # Normalize features
        rfm['recency_norm'] = 1 / (1 + rfm['recency'] / 30)  # Higher is better (more recent)
        rfm['frequency_norm'] = np.log1p(rfm['frequency']) / np.log1p(rfm['frequency'].max())
        rfm['monetary_norm'] = np.log1p(rfm['monetary']) / np.log1p(rfm['monetary'].max() + 1)
        
        return rfm


class SequenceBuilder:
    """Xây dựng sequences cho LSTM"""
    
    def __init__(self, sequence_length: int = 20, padding_value: int = 0):
        self.sequence_length = sequence_length
        self.padding_value = padding_value
        
    def build_event_sequences(self, events_df: pd.DataFrame, 
                              customer_id_col: str = 'customer_id',
                              event_type_col: str = 'event_type',
                              timestamp_col: str = 'timestamp',
                              category_col: str = 'category') -> Dict[int, Dict[str, np.ndarray]]:
        """
        Build sequences từ behavior events
        
        Returns:
            Dict mapping customer_id to sequences
        """
        if events_df.empty:
            return {}
        
        events_df = events_df.copy()
        
        # Encode event types
        event_encoder = LabelEncoder()
        events_df['event_encoded'] = event_encoder.fit_transform(events_df[event_type_col].fillna('unknown'))
        
        # Encode categories
        category_encoder = LabelEncoder()
        events_df['category_encoded'] = category_encoder.fit_transform(events_df[category_col].fillna('unknown'))
        
        # Sort by customer and timestamp
        if not pd.api.types.is_datetime64_any_dtype(events_df[timestamp_col]):
            events_df[timestamp_col] = pd.to_datetime(events_df[timestamp_col])
        
        events_df = events_df.sort_values([customer_id_col, timestamp_col])
        
        sequences = {}
        
        for customer_id, group in events_df.groupby(customer_id_col):
            event_seq = group['event_encoded'].values
            category_seq = group['category_encoded'].values
            
            # Pad or truncate
            event_seq = self._pad_sequence(event_seq)
            category_seq = self._pad_sequence(category_seq)
            
            sequences[customer_id] = {
                'events': event_seq,
                'categories': category_seq,
                'length': min(len(group), self.sequence_length)
            }
        
        self.event_encoder = event_encoder
        self.category_encoder = category_encoder
        self.n_event_types = len(event_encoder.classes_) + 1
        self.n_categories = len(category_encoder.classes_) + 1
        
        return sequences
    
    def build_order_sequences(self, orders_df: pd.DataFrame,
                             customer_id_col: str = 'customer_id',
                             date_col: str = 'order_date',
                             amount_col: str = 'total_amount',
                             category_col: str = 'category') -> Dict[int, Dict[str, np.ndarray]]:
        """Build sequences từ order history"""
        if orders_df.empty:
            return {}
        
        orders_df = orders_df.copy()
        
        # Encode categories
        category_encoder = LabelEncoder()
        orders_df['category_encoded'] = category_encoder.fit_transform(orders_df[category_col].fillna('unknown'))
        
        # Normalize amounts
        max_amount = orders_df[amount_col].max()
        orders_df['amount_norm'] = orders_df[amount_col] / (max_amount + 1)
        
        # Sort
        if not pd.api.types.is_datetime64_any_dtype(orders_df[date_col]):
            orders_df[date_col] = pd.to_datetime(orders_df[date_col])
        
        orders_df = orders_df.sort_values([customer_id_col, date_col])
        
        sequences = {}
        
        for customer_id, group in orders_df.groupby(customer_id_col):
            category_seq = group['category_encoded'].values
            amount_seq = group['amount_norm'].values
            
            category_seq = self._pad_sequence(category_seq)
            amount_seq = self._pad_sequence(amount_seq, dtype=np.float32)
            
            sequences[customer_id] = {
                'categories': category_seq,
                'amounts': amount_seq,
                'length': min(len(group), self.sequence_length)
            }
        
        self.order_category_encoder = category_encoder
        self.n_order_categories = len(category_encoder.classes_) + 1
        
        return sequences
    
    def _pad_sequence(self, seq: np.ndarray, dtype=np.int64) -> np.ndarray:
        """Pad hoặc truncate sequence"""
        if len(seq) >= self.sequence_length:
            return seq[-self.sequence_length:].astype(dtype)
        
        padded = np.full(self.sequence_length, self.padding_value, dtype=dtype)
        padded[-len(seq):] = seq
        return padded


class BehaviorDataset(Dataset):
    """PyTorch Dataset cho behavior data"""
    
    def __init__(self, features: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
        self.features = features
        self.labels = labels
        self.n_samples = len(next(iter(features.values())))
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sample_features = {k: v[idx] for k, v in self.features.items()}
        sample_labels = {k: v[idx] for k, v in self.labels.items()}
        return sample_features, sample_labels


class DataProcessor:
    """Main processor kết hợp tất cả components"""
    
    SEGMENT_MAPPING = {'VIP': 0, 'Regular': 1, 'New': 2, 'Churned': 3}
    SEGMENT_REVERSE = {0: 'VIP', 1: 'Regular', 2: 'New', 3: 'Churned'}
    
    def __init__(self, sequence_length: int = 20):
        self.feature_encoder = FeatureEncoder()
        self.rfm_calculator = RFMCalculator()
        self.sequence_builder = SequenceBuilder(sequence_length=sequence_length)
        self.fitted = False
        
    def fit_transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Fit và transform data
        
        Args:
            data: Dictionary với keys 'customers', 'orders', 'reviews', 'events'
            
        Returns:
            Processed data dictionary
        """
        customers_df = data['customers']
        orders_df = data['orders']
        reviews_df = data.get('reviews', pd.DataFrame())
        events_df = data.get('events', pd.DataFrame())
        
        # Calculate RFM
        rfm_df = self.rfm_calculator.calculate(orders_df)
        
        # Merge RFM with customers
        if not rfm_df.empty:
            customers_df = customers_df.merge(rfm_df, on='customer_id', how='left')
            customers_df = customers_df.fillna({
                'recency': 999, 'frequency': 0, 'monetary': 0,
                'recency_norm': 0, 'frequency_norm': 0, 'monetary_norm': 0,
                'rfm_score': 111
            })
        else:
            for col in ['recency', 'frequency', 'monetary', 'recency_norm', 
                       'frequency_norm', 'monetary_norm', 'rfm_score']:
                customers_df[col] = 0
        
        # Add review features
        if not reviews_df.empty:
            review_agg = reviews_df.groupby('customer_id').agg({
                'review_id': 'count',
                'rating': 'mean'
            }).reset_index()
            review_agg.columns = ['customer_id', 'n_reviews', 'avg_rating']
            customers_df = customers_df.merge(review_agg, on='customer_id', how='left')
            customers_df['n_reviews'] = customers_df['n_reviews'].fillna(0)
            customers_df['avg_rating'] = customers_df['avg_rating'].fillna(3.0)
        else:
            customers_df['n_reviews'] = 0
            customers_df['avg_rating'] = 3.0
        
        # Fit feature encoder
        categorical_cols = ['job', 'location', 'gender']
        numerical_cols = ['age', 'recency_norm', 'frequency_norm', 'monetary_norm', 
                         'n_reviews', 'avg_rating']
        
        self.feature_encoder.fit(customers_df, categorical_cols, numerical_cols)
        
        # Transform features
        categorical_encoded, numerical_encoded = self.feature_encoder.transform(customers_df)
        
        # Build sequences
        event_sequences = {}
        if not events_df.empty:
            event_sequences = self.sequence_builder.build_event_sequences(events_df)
        
        order_sequences = {}
        if not orders_df.empty:
            order_sequences = self.sequence_builder.build_order_sequences(orders_df)
        
        # Build combined sequences for each customer
        sequence_data = self._build_combined_sequences(
            customers_df, event_sequences, order_sequences
        )
        
        # Prepare labels
        labels = self._prepare_labels(customers_df, orders_df)
        
        self.fitted = True
        self.n_customers = len(customers_df)
        self.vocab_sizes = self.feature_encoder.get_vocab_sizes()
        
        return {
            'categorical': categorical_encoded,
            'numerical': numerical_encoded,
            'sequences': sequence_data,
            'labels': labels,
            'customer_ids': customers_df['customer_id'].values,
            'customers_df': customers_df
        }
    
    def _build_combined_sequences(self, customers_df: pd.DataFrame,
                                  event_sequences: Dict,
                                  order_sequences: Dict) -> Dict[str, np.ndarray]:
        """Combine event và order sequences"""
        n_customers = len(customers_df)
        seq_len = self.sequence_builder.sequence_length
        
        # Initialize arrays
        event_seq = np.zeros((n_customers, seq_len), dtype=np.int64)
        category_seq = np.zeros((n_customers, seq_len), dtype=np.int64)
        amount_seq = np.zeros((n_customers, seq_len), dtype=np.float32)
        seq_lengths = np.ones(n_customers, dtype=np.int64)  # Minimum 1
        
        for i, customer_id in enumerate(customers_df['customer_id']):
            if customer_id in event_sequences:
                event_seq[i] = event_sequences[customer_id]['events']
                seq_lengths[i] = max(seq_lengths[i], event_sequences[customer_id]['length'])
            
            if customer_id in order_sequences:
                category_seq[i] = order_sequences[customer_id]['categories']
                amount_seq[i] = order_sequences[customer_id]['amounts']
                seq_lengths[i] = max(seq_lengths[i], order_sequences[customer_id]['length'])
        
        return {
            'event_seq': event_seq,
            'category_seq': category_seq,
            'amount_seq': amount_seq,
            'seq_lengths': seq_lengths
        }
    
    def _prepare_labels(self, customers_df: pd.DataFrame, 
                        orders_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare labels cho training"""
        n_customers = len(customers_df)
        
        # Segment labels
        segment_labels = np.array([
            self.SEGMENT_MAPPING.get(seg, 1) 
            for seg in customers_df['segment']
        ], dtype=np.int64)
        
        # Churn labels (1 if Churned, 0 otherwise)
        churn_labels = (customers_df['segment'] == 'Churned').astype(np.float32).values
        
        # Next category prediction (most frequent category)
        if not orders_df.empty:
            last_categories = orders_df.sort_values('order_date').groupby('customer_id')['category'].last()
            category_encoder = self.sequence_builder.order_category_encoder if hasattr(
                self.sequence_builder, 'order_category_encoder'
            ) else None
            
            next_category = np.zeros(n_customers, dtype=np.int64)
            for i, customer_id in enumerate(customers_df['customer_id']):
                if customer_id in last_categories.index:
                    cat = last_categories[customer_id]
                    if category_encoder:
                        try:
                            next_category[i] = category_encoder.transform([cat])[0]
                        except ValueError:
                            next_category[i] = 0
        else:
            next_category = np.zeros(n_customers, dtype=np.int64)
        
        return {
            'segment': segment_labels,
            'churn': churn_labels,
            'next_category': next_category
        }
    
    def create_dataloader(self, processed_data: Dict, batch_size: int = 32, 
                         shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader"""
        features = {
            'categorical': torch.tensor(processed_data['categorical'], dtype=torch.long),
            'numerical': torch.tensor(processed_data['numerical'], dtype=torch.float32),
            'event_seq': torch.tensor(processed_data['sequences']['event_seq'], dtype=torch.long),
            'category_seq': torch.tensor(processed_data['sequences']['category_seq'], dtype=torch.long),
            'amount_seq': torch.tensor(processed_data['sequences']['amount_seq'], dtype=torch.float32),
            'seq_lengths': torch.tensor(processed_data['sequences']['seq_lengths'], dtype=torch.long),
        }
        
        labels = {
            'segment': torch.tensor(processed_data['labels']['segment'], dtype=torch.long),
            'churn': torch.tensor(processed_data['labels']['churn'], dtype=torch.float32),
            'next_category': torch.tensor(processed_data['labels']['next_category'], dtype=torch.long),
        }
        
        dataset = BehaviorDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def transform_single(self, customer_data: Dict, 
                        order_history: List[Dict] = None,
                        events: List[Dict] = None) -> Dict[str, torch.Tensor]:
        """Transform single customer data cho inference"""
        if not self.fitted:
            raise ValueError("Processor not fitted")
        
        # Create DataFrame
        customer_df = pd.DataFrame([customer_data])
        
        # Add default values
        for col in ['recency_norm', 'frequency_norm', 'monetary_norm', 'n_reviews', 'avg_rating']:
            if col not in customer_df.columns:
                customer_df[col] = 0.0
        
        # Transform
        categorical, numerical = self.feature_encoder.transform(customer_df)
        
        # Create sequences
        seq_len = self.sequence_builder.sequence_length
        event_seq = np.zeros((1, seq_len), dtype=np.int64)
        category_seq = np.zeros((1, seq_len), dtype=np.int64)
        amount_seq = np.zeros((1, seq_len), dtype=np.float32)
        
        return {
            'categorical': torch.tensor(categorical, dtype=torch.long),
            'numerical': torch.tensor(numerical, dtype=torch.float32),
            'event_seq': torch.tensor(event_seq, dtype=torch.long),
            'category_seq': torch.tensor(category_seq, dtype=torch.long),
            'amount_seq': torch.tensor(amount_seq, dtype=torch.float32),
            'seq_lengths': torch.tensor([1], dtype=torch.long),
        }

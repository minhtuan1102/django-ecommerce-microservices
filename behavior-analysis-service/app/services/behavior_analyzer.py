"""
Behavior Analyzer Service
Load trained model và cung cấp inference methods
"""
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from django.conf import settings
import logging
import json
import pickle

from app.ml_models.behavior_model import BehaviorAnalysisModel, create_model
from app.ml_models.data_processor import DataProcessor, FeatureEncoder, RFMCalculator

logger = logging.getLogger(__name__)


class BehaviorAnalyzer:
    """Service để analyze customer behavior"""
    
    SEGMENT_NAMES = {0: 'VIP', 1: 'Regular', 2: 'New', 3: 'Churned'}
    SEGMENT_DESCRIPTIONS = {
        'VIP': 'Khách hàng VIP - Mua sắm thường xuyên, giá trị cao',
        'Regular': 'Khách hàng thường xuyên - Mua sắm ổn định',
        'New': 'Khách hàng mới - Vừa đăng ký gần đây',
        'Churned': 'Khách hàng ngừng hoạt động - Cần retention'
    }
    
    CATEGORY_NAMES = [
        'Technology', 'Fiction', 'Science', 'Business', 'Self-Help',
        'History', 'Arts', 'Education', 'Health', 'Travel'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            getattr(settings, 'MODEL_PATH', 'data/models'),
            'behavior_model.pt'
        )
        self.processor_path = os.path.join(
            os.path.dirname(self.model_path),
            'data_processor.pkl'
        )
        self.config_path = os.path.join(
            os.path.dirname(self.model_path),
            'model_config.json'
        )
        
        self.model: Optional[BehaviorAnalysisModel] = None
        self.processor: Optional[DataProcessor] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._loaded = False
        
        # Try to load model on init
        self._try_load_model()
    
    def _try_load_model(self):
        """Attempt to load model and processor"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.config_path):
                self.load_model()
                self._loaded = True
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model files not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def load_model(self):
        """Load trained model và processor"""
        # Load config
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        self.model = create_model(
            vocab_sizes=config['vocab_sizes'],
            n_event_types=config.get('n_event_types', 10),
            n_categories=config.get('n_categories', 10),
            embedding_dim=config.get('embedding_dim', 32),
            lstm_hidden_size=config.get('lstm_hidden_size', 128),
            lstm_layers=config.get('lstm_layers', 2),
            numerical_features_dim=config.get('numerical_features_dim', 6)
        )
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load processor
        if os.path.exists(self.processor_path):
            with open(self.processor_path, 'rb') as f:
                self.processor = pickle.load(f)
        else:
            logger.warning("Processor not found, creating new one")
            self.processor = DataProcessor()
        
        self._loaded = True
        logger.info(f"Model loaded from {self.model_path}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded and self.model is not None
    
    def analyze_customer(self, customer_data: Dict, 
                        order_history: Optional[List[Dict]] = None,
                        events: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Phân tích hành vi customer
        
        Args:
            customer_data: Dict với thông tin customer
            order_history: List các orders
            events: List các behavior events
            
        Returns:
            Dict với segment, predictions và confidence
        """
        if not self.is_loaded():
            return self._get_fallback_analysis(customer_data)
        
        try:
            # Prepare features
            features = self._prepare_features(customer_data, order_history, events)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model.predict(features)
            
            # Extract predictions
            segment_probs = outputs['segment_probs'].cpu().numpy()[0]
            segment_idx = int(np.argmax(segment_probs))
            segment_name = self.SEGMENT_NAMES.get(segment_idx, 'Unknown')
            
            category_probs = outputs['category_probs'].cpu().numpy()[0]
            top_categories = np.argsort(category_probs)[::-1][:3]
            
            churn_prob = float(outputs['churn_prob'].cpu().numpy()[0])
            
            return {
                'customer_id': customer_data.get('customer_id', customer_data.get('id')),
                'segment': segment_name,
                'segment_confidence': float(segment_probs[segment_idx]),
                'segment_probabilities': {
                    self.SEGMENT_NAMES[i]: float(p) 
                    for i, p in enumerate(segment_probs)
                },
                'churn_risk': churn_prob,
                'churn_level': self._get_churn_level(churn_prob),
                'predicted_categories': [
                    {
                        'category': self.CATEGORY_NAMES[i] if i < len(self.CATEGORY_NAMES) else f'Category_{i}',
                        'probability': float(category_probs[i])
                    }
                    for i in top_categories
                ],
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing customer: {e}")
            return self._get_fallback_analysis(customer_data)
    
    def get_customer_insights(self, customer_data: Dict,
                             order_history: Optional[List[Dict]] = None,
                             events: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Get detailed insights cho customer
        """
        base_analysis = self.analyze_customer(customer_data, order_history, events)
        
        # Calculate RFM nếu có order history
        rfm_insights = self._calculate_rfm_insights(order_history)
        
        # Behavior patterns
        behavior_patterns = self._analyze_behavior_patterns(events)
        
        # Recommendations
        recommendations = self._generate_recommendations(
            base_analysis, rfm_insights, behavior_patterns
        )
        
        return {
            **base_analysis,
            'rfm_analysis': rfm_insights,
            'behavior_patterns': behavior_patterns,
            'recommendations': recommendations,
            'engagement_score': self._calculate_engagement_score(
                base_analysis, rfm_insights, behavior_patterns
            )
        }
    
    def predict_churn_risk(self, customer_data: Dict,
                          order_history: Optional[List[Dict]] = None,
                          events: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Dự đoán xác suất churn
        """
        analysis = self.analyze_customer(customer_data, order_history, events)
        
        churn_prob = analysis.get('churn_risk', 0.5)
        
        # Identify risk factors
        risk_factors = []
        
        if order_history:
            last_order_date = self._get_last_order_date(order_history)
            if last_order_date:
                days_since_last = (datetime.now() - last_order_date).days
                if days_since_last > 90:
                    risk_factors.append({
                        'factor': 'inactive_period',
                        'description': f'Không có đơn hàng trong {days_since_last} ngày',
                        'impact': 'high'
                    })
                elif days_since_last > 30:
                    risk_factors.append({
                        'factor': 'reduced_activity',
                        'description': f'Đơn hàng cuối cách đây {days_since_last} ngày',
                        'impact': 'medium'
                    })
            
            # Declining purchase frequency
            if len(order_history) >= 4:
                recent_orders = len([o for o in order_history 
                                   if self._days_ago(o.get('order_date', '')) < 90])
                old_orders = len(order_history) - recent_orders
                if old_orders > 0 and recent_orders / (old_orders + 1) < 0.3:
                    risk_factors.append({
                        'factor': 'declining_frequency',
                        'description': 'Tần suất mua hàng giảm đáng kể',
                        'impact': 'medium'
                    })
        
        if analysis.get('segment') == 'Churned':
            risk_factors.append({
                'factor': 'churned_segment',
                'description': 'Đã được phân loại là khách hàng rời bỏ',
                'impact': 'high'
            })
        
        # Retention suggestions
        retention_suggestions = self._get_retention_suggestions(churn_prob, risk_factors)
        
        return {
            'customer_id': customer_data.get('customer_id', customer_data.get('id')),
            'churn_probability': churn_prob,
            'churn_level': analysis.get('churn_level', 'Unknown'),
            'risk_factors': risk_factors,
            'retention_suggestions': retention_suggestions,
            'predicted_at': datetime.now().isoformat()
        }
    
    def batch_analyze(self, customers: List[Dict], 
                     orders_by_customer: Optional[Dict[int, List[Dict]]] = None,
                     events_by_customer: Optional[Dict[int, List[Dict]]] = None) -> List[Dict]:
        """Batch analysis cho nhiều customers"""
        results = []
        
        for customer in customers:
            customer_id = customer.get('customer_id', customer.get('id'))
            
            orders = orders_by_customer.get(customer_id, []) if orders_by_customer else None
            events = events_by_customer.get(customer_id, []) if events_by_customer else None
            
            analysis = self.analyze_customer(customer, orders, events)
            results.append(analysis)
        
        return results
    
    def _prepare_features(self, customer_data: Dict,
                         order_history: Optional[List[Dict]],
                         events: Optional[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Prepare features cho model inference"""
        if self.processor and self.processor.fitted:
            features = self.processor.transform_single(customer_data, order_history, events)
            return {k: v.to(self.device) for k, v in features.items()}
        
        # Fallback: create simple features
        return self._create_simple_features(customer_data)
    
    def _create_simple_features(self, customer_data: Dict) -> Dict[str, torch.Tensor]:
        """Create simple features when processor is not available"""
        # Simple categorical encoding
        job_map = {'Software Engineer': 1, 'Data Scientist': 2, 'Teacher': 3, 'Doctor': 4}
        location_map = {'Ha Noi': 1, 'Ho Chi Minh': 2, 'Da Nang': 3}
        gender_map = {'M': 1, 'F': 2}
        
        categorical = torch.tensor([[
            job_map.get(customer_data.get('job', ''), 0),
            location_map.get(customer_data.get('location', ''), 0),
            gender_map.get(customer_data.get('gender', ''), 0)
        ]], dtype=torch.long, device=self.device)
        
        numerical = torch.tensor([[
            customer_data.get('age', 30) / 100.0,
            0.5, 0.5, 0.5, 0.0, 3.0
        ]], dtype=torch.float32, device=self.device)
        
        seq_len = 20
        return {
            'categorical': categorical,
            'numerical': numerical,
            'event_seq': torch.zeros(1, seq_len, dtype=torch.long, device=self.device),
            'category_seq': torch.zeros(1, seq_len, dtype=torch.long, device=self.device),
            'amount_seq': torch.zeros(1, seq_len, dtype=torch.float32, device=self.device),
            'seq_lengths': torch.tensor([1], dtype=torch.long, device=self.device)
        }
    
    def _get_fallback_analysis(self, customer_data: Dict) -> Dict[str, Any]:
        """Fallback analysis khi model không available"""
        # Simple rule-based analysis
        segment = 'Regular'
        churn_risk = 0.3
        
        registration_date = customer_data.get('registration_date')
        if registration_date:
            if isinstance(registration_date, str):
                try:
                    registration_date = datetime.fromisoformat(registration_date.replace('Z', '+00:00'))
                except:
                    registration_date = None
            
            if registration_date:
                days_since_reg = (datetime.now() - registration_date.replace(tzinfo=None)).days
                if days_since_reg < 30:
                    segment = 'New'
                    churn_risk = 0.2
        
        return {
            'customer_id': customer_data.get('customer_id', customer_data.get('id')),
            'segment': segment,
            'segment_confidence': 0.5,
            'segment_probabilities': {'VIP': 0.1, 'Regular': 0.5, 'New': 0.25, 'Churned': 0.15},
            'churn_risk': churn_risk,
            'churn_level': self._get_churn_level(churn_risk),
            'predicted_categories': [
                {'category': 'Technology', 'probability': 0.3},
                {'category': 'Fiction', 'probability': 0.2},
                {'category': 'Business', 'probability': 0.15}
            ],
            'analyzed_at': datetime.now().isoformat(),
            'note': 'Analysis performed with fallback rules (model not loaded)'
        }
    
    def _get_churn_level(self, probability: float) -> str:
        """Convert probability to churn level"""
        if probability < 0.2:
            return 'Very Low'
        elif probability < 0.4:
            return 'Low'
        elif probability < 0.6:
            return 'Medium'
        elif probability < 0.8:
            return 'High'
        else:
            return 'Very High'
    
    def _calculate_rfm_insights(self, order_history: Optional[List[Dict]]) -> Dict[str, Any]:
        """Calculate RFM insights từ order history"""
        if not order_history:
            return {'status': 'no_order_history'}
        
        calculator = RFMCalculator()
        orders_df = pd.DataFrame(order_history)
        
        if 'order_date' not in orders_df.columns:
            return {'status': 'invalid_order_data'}
        
        # Convert dates
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        
        # Calculate metrics
        recency = (datetime.now() - orders_df['order_date'].max()).days
        frequency = len(orders_df)
        monetary = orders_df.get('total_amount', orders_df.get('amount', pd.Series([0]))).sum()
        
        return {
            'recency_days': recency,
            'frequency': frequency,
            'monetary_total': float(monetary),
            'avg_order_value': float(monetary / frequency) if frequency > 0 else 0,
            'recency_score': 5 if recency < 7 else (4 if recency < 30 else (3 if recency < 60 else (2 if recency < 90 else 1))),
            'frequency_score': min(5, 1 + frequency // 3),
            'monetary_score': 5 if monetary > 1000000 else (4 if monetary > 500000 else (3 if monetary > 200000 else (2 if monetary > 50000 else 1)))
        }
    
    def _analyze_behavior_patterns(self, events: Optional[List[Dict]]) -> Dict[str, Any]:
        """Analyze behavior patterns từ events"""
        if not events:
            return {'status': 'no_events'}
        
        events_df = pd.DataFrame(events)
        
        # Count event types
        event_counts = events_df['event_type'].value_counts().to_dict() if 'event_type' in events_df.columns else {}
        
        # Device preferences
        device_pref = events_df['device'].value_counts().to_dict() if 'device' in events_df.columns else {}
        
        # Category interests
        category_interests = events_df['category'].value_counts().to_dict() if 'category' in events_df.columns else {}
        
        return {
            'total_events': len(events),
            'event_distribution': event_counts,
            'device_preferences': device_pref,
            'category_interests': category_interests,
            'engagement_level': 'High' if len(events) > 50 else ('Medium' if len(events) > 20 else 'Low')
        }
    
    def _generate_recommendations(self, analysis: Dict, 
                                 rfm: Dict, patterns: Dict) -> List[Dict]:
        """Generate personalized recommendations"""
        recommendations = []
        
        segment = analysis.get('segment', 'Regular')
        churn_risk = analysis.get('churn_risk', 0.5)
        
        # Segment-based recommendations
        if segment == 'VIP':
            recommendations.append({
                'type': 'loyalty',
                'title': 'VIP Benefits',
                'description': 'Cung cấp ưu đãi độc quyền VIP, early access sản phẩm mới',
                'priority': 'medium'
            })
        elif segment == 'New':
            recommendations.append({
                'type': 'onboarding',
                'title': 'Welcome Campaign',
                'description': 'Gửi welcome email, hướng dẫn sử dụng, ưu đãi đơn hàng đầu tiên',
                'priority': 'high'
            })
        elif segment == 'Churned':
            recommendations.append({
                'type': 'reactivation',
                'title': 'Win-back Campaign',
                'description': 'Gửi offer đặc biệt để khuyến khích quay lại',
                'priority': 'high'
            })
        
        # Churn-based recommendations
        if churn_risk > 0.6:
            recommendations.append({
                'type': 'retention',
                'title': 'Retention Alert',
                'description': 'Cần chiến dịch retention ngay: discount, personalized offer',
                'priority': 'urgent'
            })
        
        # Category-based recommendations
        predicted_cats = analysis.get('predicted_categories', [])
        if predicted_cats:
            top_cat = predicted_cats[0]['category']
            recommendations.append({
                'type': 'personalization',
                'title': f'Recommend {top_cat}',
                'description': f'Gợi ý sản phẩm category {top_cat} dựa trên behavior',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _calculate_engagement_score(self, analysis: Dict, 
                                   rfm: Dict, patterns: Dict) -> float:
        """Calculate overall engagement score (0-100)"""
        score = 50.0  # Base score
        
        # Segment bonus
        segment_bonus = {'VIP': 30, 'Regular': 15, 'New': 5, 'Churned': -20}
        score += segment_bonus.get(analysis.get('segment', 'Regular'), 0)
        
        # RFM bonus
        if rfm.get('status') != 'no_order_history':
            r_score = rfm.get('recency_score', 3)
            f_score = rfm.get('frequency_score', 3)
            m_score = rfm.get('monetary_score', 3)
            rfm_bonus = (r_score + f_score + m_score) * 2  # Max 30
            score += rfm_bonus - 18  # Normalize around 0
        
        # Engagement bonus
        if patterns.get('status') != 'no_events':
            engagement = patterns.get('engagement_level', 'Low')
            eng_bonus = {'High': 15, 'Medium': 5, 'Low': -5}
            score += eng_bonus.get(engagement, 0)
        
        # Churn penalty
        churn_risk = analysis.get('churn_risk', 0.5)
        score -= churn_risk * 20
        
        return max(0, min(100, score))
    
    def _get_last_order_date(self, order_history: List[Dict]) -> Optional[datetime]:
        """Get last order date từ history"""
        if not order_history:
            return None
        
        dates = []
        for order in order_history:
            date_str = order.get('order_date')
            if date_str:
                try:
                    if isinstance(date_str, str):
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        dates.append(dt.replace(tzinfo=None))
                    elif isinstance(date_str, datetime):
                        dates.append(date_str.replace(tzinfo=None) if date_str.tzinfo else date_str)
                except:
                    pass
        
        return max(dates) if dates else None
    
    def _days_ago(self, date_str: str) -> int:
        """Calculate days ago from date string"""
        if not date_str:
            return 999
        try:
            if isinstance(date_str, str):
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return (datetime.now() - dt.replace(tzinfo=None)).days
            elif isinstance(date_str, datetime):
                return (datetime.now() - date_str.replace(tzinfo=None)).days
        except:
            return 999
    
    def _get_retention_suggestions(self, churn_prob: float, 
                                  risk_factors: List[Dict]) -> List[Dict]:
        """Generate retention suggestions based on risk"""
        suggestions = []
        
        if churn_prob > 0.7:
            suggestions.extend([
                {
                    'action': 'urgent_outreach',
                    'description': 'Liên hệ trực tiếp qua phone/email với offer đặc biệt',
                    'urgency': 'immediate'
                },
                {
                    'action': 'special_discount',
                    'description': 'Cung cấp discount 30-50% cho đơn hàng tiếp theo',
                    'urgency': 'immediate'
                }
            ])
        elif churn_prob > 0.5:
            suggestions.extend([
                {
                    'action': 'email_campaign',
                    'description': 'Gửi email "We miss you" với personalized recommendations',
                    'urgency': 'within_week'
                },
                {
                    'action': 'loyalty_points',
                    'description': 'Tặng bonus loyalty points',
                    'urgency': 'within_week'
                }
            ])
        else:
            suggestions.append({
                'action': 'engagement_maintain',
                'description': 'Duy trì engagement qua newsletter và personalized content',
                'urgency': 'ongoing'
            })
        
        return suggestions


# Singleton instance
_analyzer_instance: Optional[BehaviorAnalyzer] = None

def get_analyzer() -> BehaviorAnalyzer:
    """Get or create analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = BehaviorAnalyzer()
    return _analyzer_instance

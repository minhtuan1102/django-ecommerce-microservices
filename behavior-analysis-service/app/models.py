"""
Django Models cho Behavior Analysis Service
"""
from django.db import models
from django.utils import timezone
import json


class CustomerBehavior(models.Model):
    """Lưu trữ phân tích hành vi customer"""
    
    SEGMENT_CHOICES = [
        ('VIP', 'VIP'),
        ('Regular', 'Regular'),
        ('New', 'New'),
        ('Churned', 'Churned'),
    ]
    
    customer_id = models.IntegerField(db_index=True, unique=True)
    segment = models.CharField(max_length=20, choices=SEGMENT_CHOICES, default='Regular')
    segment_confidence = models.FloatField(default=0.0)
    segment_probabilities = models.JSONField(default=dict, blank=True)
    
    churn_risk = models.FloatField(default=0.0)
    churn_level = models.CharField(max_length=20, default='Low')
    
    predicted_categories = models.JSONField(default=list, blank=True)
    engagement_score = models.FloatField(default=50.0)
    
    # RFM metrics
    rfm_recency = models.IntegerField(default=0)  # Days since last order
    rfm_frequency = models.IntegerField(default=0)  # Total orders
    rfm_monetary = models.FloatField(default=0.0)  # Total spend
    rfm_score = models.IntegerField(default=0)
    
    last_analyzed = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'customer_behaviors'
        verbose_name = 'Customer Behavior'
        verbose_name_plural = 'Customer Behaviors'
        ordering = ['-last_analyzed']
    
    def __str__(self):
        return f"Customer {self.customer_id} - {self.segment}"
    
    def get_segment_description(self):
        descriptions = {
            'VIP': 'Khách hàng VIP - Mua sắm thường xuyên với giá trị cao',
            'Regular': 'Khách hàng thường xuyên - Mua sắm ổn định',
            'New': 'Khách hàng mới - Vừa đăng ký gần đây',
            'Churned': 'Khách hàng ngừng hoạt động - Cần retention',
        }
        return descriptions.get(self.segment, 'Unknown segment')
    
    def update_from_analysis(self, analysis: dict):
        """Update from analysis result"""
        self.segment = analysis.get('segment', self.segment)
        self.segment_confidence = analysis.get('segment_confidence', 0.0)
        self.segment_probabilities = analysis.get('segment_probabilities', {})
        self.churn_risk = analysis.get('churn_risk', 0.0)
        self.churn_level = analysis.get('churn_level', 'Low')
        self.predicted_categories = analysis.get('predicted_categories', [])
        self.engagement_score = analysis.get('engagement_score', 50.0)
        
        rfm = analysis.get('rfm_analysis', {})
        if rfm and rfm.get('status') != 'no_order_history':
            self.rfm_recency = rfm.get('recency_days', 0)
            self.rfm_frequency = rfm.get('frequency', 0)
            self.rfm_monetary = rfm.get('monetary_total', 0.0)
        
        self.save()


class BehaviorEvent(models.Model):
    """Track behavior events"""
    
    EVENT_TYPE_CHOICES = [
        ('page_view', 'Page View'),
        ('product_view', 'Product View'),
        ('add_to_cart', 'Add to Cart'),
        ('remove_from_cart', 'Remove from Cart'),
        ('purchase', 'Purchase'),
        ('search', 'Search'),
        ('review', 'Review'),
        ('wishlist_add', 'Add to Wishlist'),
        ('wishlist_remove', 'Remove from Wishlist'),
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('signup', 'Sign Up'),
    ]
    
    DEVICE_CHOICES = [
        ('desktop', 'Desktop'),
        ('mobile', 'Mobile'),
        ('tablet', 'Tablet'),
    ]
    
    customer_id = models.IntegerField(db_index=True)
    event_type = models.CharField(max_length=50, choices=EVENT_TYPE_CHOICES)
    event_data = models.JSONField(default=dict, blank=True)
    
    # Context
    session_id = models.CharField(max_length=100, blank=True, null=True)
    device = models.CharField(max_length=20, choices=DEVICE_CHOICES, default='desktop')
    category = models.CharField(max_length=100, blank=True, null=True)
    product_id = models.IntegerField(blank=True, null=True)
    
    # Metadata
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.TextField(blank=True, null=True)
    referrer = models.URLField(blank=True, null=True)
    
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    
    class Meta:
        db_table = 'behavior_events'
        verbose_name = 'Behavior Event'
        verbose_name_plural = 'Behavior Events'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['customer_id', 'timestamp']),
            models.Index(fields=['event_type', 'timestamp']),
            models.Index(fields=['customer_id', 'event_type']),
        ]
    
    def __str__(self):
        return f"Customer {self.customer_id} - {self.event_type} at {self.timestamp}"
    
    @classmethod
    def track(cls, customer_id: int, event_type: str, **kwargs):
        """Convenience method to track an event"""
        return cls.objects.create(
            customer_id=customer_id,
            event_type=event_type,
            event_data=kwargs.pop('event_data', {}),
            session_id=kwargs.pop('session_id', None),
            device=kwargs.pop('device', 'desktop'),
            category=kwargs.pop('category', None),
            product_id=kwargs.pop('product_id', None),
            ip_address=kwargs.pop('ip_address', None),
            user_agent=kwargs.pop('user_agent', None),
            referrer=kwargs.pop('referrer', None),
        )


class AnalysisResult(models.Model):
    """Lưu trữ kết quả analysis"""
    
    ANALYSIS_TYPE_CHOICES = [
        ('full_analysis', 'Full Analysis'),
        ('segment', 'Segment Classification'),
        ('churn', 'Churn Prediction'),
        ('category', 'Category Prediction'),
        ('rfm', 'RFM Analysis'),
        ('insights', 'Customer Insights'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    customer_id = models.IntegerField(db_index=True)
    analysis_type = models.CharField(max_length=50, choices=ANALYSIS_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='completed')
    
    result = models.JSONField(default=dict)
    error_message = models.TextField(blank=True, null=True)
    
    # Performance metrics
    processing_time_ms = models.IntegerField(default=0)
    model_version = models.CharField(max_length=50, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        db_table = 'analysis_results'
        verbose_name = 'Analysis Result'
        verbose_name_plural = 'Analysis Results'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['customer_id', 'analysis_type']),
            models.Index(fields=['customer_id', 'created_at']),
        ]
    
    def __str__(self):
        return f"Customer {self.customer_id} - {self.analysis_type} at {self.created_at}"
    
    @classmethod
    def create_result(cls, customer_id: int, analysis_type: str, 
                     result: dict, processing_time_ms: int = 0,
                     model_version: str = None):
        """Create analysis result"""
        return cls.objects.create(
            customer_id=customer_id,
            analysis_type=analysis_type,
            status='completed',
            result=result,
            processing_time_ms=processing_time_ms,
            model_version=model_version
        )


class ModelMetadata(models.Model):
    """Track model versions và performance"""
    
    version = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    
    # File paths
    model_path = models.CharField(max_length=500)
    processor_path = models.CharField(max_length=500, blank=True, null=True)
    config_path = models.CharField(max_length=500, blank=True, null=True)
    
    # Training info
    training_samples = models.IntegerField(default=0)
    training_date = models.DateTimeField(blank=True, null=True)
    training_duration_seconds = models.IntegerField(default=0)
    
    # Performance metrics
    accuracy_segment = models.FloatField(default=0.0)
    accuracy_category = models.FloatField(default=0.0)
    auc_churn = models.FloatField(default=0.0)
    
    # Status
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'model_metadata'
        verbose_name = 'Model Metadata'
        verbose_name_plural = 'Model Metadata'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Model {self.version} {'(active)' if self.is_active else ''}"
    
    @classmethod
    def get_active_model(cls):
        """Get currently active model"""
        return cls.objects.filter(is_active=True).first()
    
    def activate(self):
        """Set this model as active"""
        ModelMetadata.objects.update(is_active=False)
        self.is_active = True
        self.save()

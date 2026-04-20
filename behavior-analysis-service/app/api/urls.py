"""
URL Configuration cho Behavior Analysis API
"""
from django.urls import path
from app.api.views import (
    CustomerAnalysisView,
    CustomerSegmentView,
    CustomerPredictionsView,
    TrackBehaviorView,
    HealthCheckView,
    BulkAnalysisView,
    CustomerEventsView,
    SegmentSummaryView,
    CustomerGraphRecommendationsView,
)

app_name = 'behavior_api'

urlpatterns = [
    # Customer-specific endpoints
    path('customer/<int:customer_id>/analysis/', CustomerAnalysisView.as_view(), name='customer-analysis'),
    path('customer/<int:customer_id>/segment/', CustomerSegmentView.as_view(), name='customer-segment'),
    path('customer/<int:customer_id>/predictions/', CustomerPredictionsView.as_view(), name='customer-predictions'),
    path('customer/<int:customer_id>/events/', CustomerEventsView.as_view(), name='customer-events'),
    path('customer/<int:customer_id>/graph-recommendations/', CustomerGraphRecommendationsView.as_view(), name='customer-graph-recommendations'),
    
    # Tracking
    path('track/', TrackBehaviorView.as_view(), name='track-behavior'),
    
    # Bulk operations
    path('bulk-analysis/', BulkAnalysisView.as_view(), name='bulk-analysis'),
    
    # Aggregations
    path('segments/summary/', SegmentSummaryView.as_view(), name='segment-summary'),
    
    # Health
    path('health/', HealthCheckView.as_view(), name='health'),
]

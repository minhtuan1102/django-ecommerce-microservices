"""
URL routing for Chat API
"""
from django.urls import path
from . import views

urlpatterns = [
    # Main chat endpoints
    path('message/', views.SendMessageView.as_view(), name='send_message'),
    path('history/<uuid:session_id>/', views.ChatHistoryView.as_view(), name='chat_history'),
    path('feedback/', views.SubmitFeedbackView.as_view(), name='submit_feedback'),
    path('health/', views.HealthCheckView.as_view(), name='health_check'),
    
    # Session management
    path('sessions/<str:customer_id>/', views.CustomerSessionsView.as_view(), name='customer_sessions'),
    path('session/<uuid:session_id>/', views.delete_session, name='delete_session'),
    path('session/<uuid:session_id>/clear/', views.clear_session, name='clear_session'),
    
    # Knowledge search
    path('search/', views.SearchKnowledgeView.as_view(), name='search_knowledge'),
]

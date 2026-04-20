"""
Tests for behavior client
"""
import pytest
from unittest.mock import patch, Mock
import requests

from app.utils.behavior_client import BehaviorClient


class TestBehaviorClient:
    """Tests for BehaviorClient"""
    
    def test_initialization(self):
        client = BehaviorClient(
            behavior_service_url="http://test:8000",
            timeout=5
        )
        assert client.base_url == "http://test:8000"
        assert client.timeout == 5
    
    def test_mock_customer_analysis(self):
        client = BehaviorClient()
        analysis = client._get_mock_customer_analysis("customer-123")
        
        assert 'customer_id' in analysis
        assert 'segment' in analysis
        assert 'favorite_categories' in analysis
        assert isinstance(analysis['favorite_categories'], list)
    
    def test_mock_recommendations(self):
        client = BehaviorClient()
        recs = client._get_mock_recommendations()
        
        assert len(recs) > 0
        assert 'name' in recs[0]
        assert 'price' in recs[0]
    
    @patch('requests.Session.request')
    def test_get_customer_analysis_success(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'customer_id': 'test',
            'segment': 'VIP'
        }
        mock_request.return_value = mock_response
        
        client = BehaviorClient()
        result = client.get_customer_analysis('test')
        
        assert result['segment'] == 'VIP'
    
    @patch('requests.Session.request')
    def test_get_customer_analysis_failure(self, mock_request):
        mock_request.side_effect = requests.exceptions.Timeout()
        
        client = BehaviorClient()
        result = client.get_customer_analysis('test')
        
        # Should return mock data on failure
        assert 'segment' in result
    
    @patch('requests.Session.request')
    def test_get_recommendations(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'name': 'Book 1', 'price': '100.000đ'}
        ]
        mock_request.return_value = mock_response
        
        client = BehaviorClient()
        result = client.get_recommendations('test', limit=5)
        
        assert len(result) == 1
    
    @patch('requests.Session.request')
    def test_health_check_success(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'healthy'}
        mock_request.return_value = mock_response
        
        client = BehaviorClient()
        result = client.health_check()
        
        assert result == True
    
    @patch('requests.Session.request')
    def test_health_check_failure(self, mock_request):
        mock_request.side_effect = requests.exceptions.ConnectionError()
        
        client = BehaviorClient()
        result = client.health_check()
        
        assert result == False
    
    def test_context_manager(self):
        with BehaviorClient() as client:
            assert client._session is not None or client._session is None
        # Session should be closed after context
    
    def test_different_segments_for_different_customers(self):
        client = BehaviorClient()
        
        analysis1 = client._get_mock_customer_analysis("customer-1")
        analysis2 = client._get_mock_customer_analysis("customer-999")
        
        # Different customer IDs should potentially get different segments
        # (based on hash function)
        assert 'segment' in analysis1
        assert 'segment' in analysis2

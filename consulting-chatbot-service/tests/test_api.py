"""
Tests for API views
"""
import json
import uuid
from unittest.mock import patch, Mock

from django.test import TestCase, Client
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from app.models import ChatSession, ChatMessage, ChatFeedback


class SendMessageAPITest(APITestCase):
    """Tests for send message endpoint"""
    
    def setUp(self):
        self.url = '/api/chat/message/'
        self.client = Client()
    
    def test_send_message_success(self):
        data = {
            'message': 'Xin chào, tôi muốn tìm sách'
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('response', response.json())
        self.assertIn('session_id', response.json())
    
    def test_send_message_with_customer_id(self):
        data = {
            'message': 'Tìm sách về lập trình',
            'customer_id': 'customer-123'
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()['customer_id'], 'customer-123')
    
    def test_send_message_with_session(self):
        # Create a session first
        session = ChatSession.objects.create(customer_id='test-customer')
        
        data = {
            'message': 'Test message',
            'session_id': str(session.id)
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()['session_id'], str(session.id))
    
    def test_send_message_empty(self):
        data = {
            'message': ''
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_send_message_with_category(self):
        data = {
            'message': 'Chính sách đổi trả',
            'category': 'policies'
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)


class ChatHistoryAPITest(APITestCase):
    """Tests for chat history endpoint"""
    
    def setUp(self):
        self.session = ChatSession.objects.create(customer_id='test-customer')
        ChatMessage.objects.create(
            session=self.session,
            role='user',
            content='Test question'
        )
        ChatMessage.objects.create(
            session=self.session,
            role='assistant',
            content='Test answer'
        )
    
    def test_get_history_success(self):
        url = f'/api/chat/history/{self.session.id}/'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()['total_messages'], 2)
    
    def test_get_history_not_found(self):
        fake_id = uuid.uuid4()
        url = f'/api/chat/history/{fake_id}/'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


class FeedbackAPITest(APITestCase):
    """Tests for feedback endpoint"""
    
    def setUp(self):
        self.session = ChatSession.objects.create()
        self.message = ChatMessage.objects.create(
            session=self.session,
            role='assistant',
            content='Test response'
        )
        self.url = '/api/chat/feedback/'
    
    def test_submit_feedback_success(self):
        data = {
            'message_id': str(self.message.id),
            'rating': 5,
            'comment': 'Great response!'
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(ChatFeedback.objects.count(), 1)
    
    def test_submit_feedback_invalid_rating(self):
        data = {
            'message_id': str(self.message.id),
            'rating': 10
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_update_existing_feedback(self):
        # Create initial feedback
        ChatFeedback.objects.create(
            message=self.message,
            rating=3
        )
        
        # Update feedback
        data = {
            'message_id': str(self.message.id),
            'rating': 5,
            'comment': 'Updated comment'
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        feedback = ChatFeedback.objects.get(message=self.message)
        self.assertEqual(feedback.rating, 5)


class HealthCheckAPITest(APITestCase):
    """Tests for health check endpoint"""
    
    def test_health_check(self):
        url = '/api/chat/health/'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.json())
        self.assertIn('service', response.json())


class SearchAPITest(APITestCase):
    """Tests for search endpoint"""
    
    def test_search_success(self):
        url = '/api/chat/search/?q=sách kinh tế'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.json())
    
    def test_search_with_category(self):
        url = '/api/chat/search/?q=đổi trả&category=policies'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_search_no_query(self):
        url = '/api/chat/search/'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

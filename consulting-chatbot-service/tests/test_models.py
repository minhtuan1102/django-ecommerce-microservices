"""
Tests for Django models
"""
import pytest
from datetime import datetime
from django.test import TestCase
from django.utils import timezone

from app.models import ChatSession, ChatMessage, ChatFeedback, ConversationContext


class ChatSessionModelTest(TestCase):
    """Tests for ChatSession model"""
    
    def test_create_session(self):
        session = ChatSession.objects.create(customer_id='test-customer')
        
        self.assertIsNotNone(session.id)
        self.assertEqual(session.customer_id, 'test-customer')
        self.assertTrue(session.is_active)
    
    def test_session_str(self):
        session = ChatSession.objects.create(customer_id='customer-123')
        
        self.assertIn('customer-123', str(session))
    
    def test_message_count_property(self):
        session = ChatSession.objects.create()
        
        ChatMessage.objects.create(
            session=session,
            role='user',
            content='Test 1'
        )
        ChatMessage.objects.create(
            session=session,
            role='assistant',
            content='Test 2'
        )
        
        self.assertEqual(session.message_count, 2)
    
    def test_get_conversation_history(self):
        session = ChatSession.objects.create()
        
        ChatMessage.objects.create(
            session=session,
            role='user',
            content='Question'
        )
        ChatMessage.objects.create(
            session=session,
            role='assistant',
            content='Answer'
        )
        
        history = session.get_conversation_history()
        
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[1]['role'], 'assistant')


class ChatMessageModelTest(TestCase):
    """Tests for ChatMessage model"""
    
    def setUp(self):
        self.session = ChatSession.objects.create()
    
    def test_create_message(self):
        message = ChatMessage.objects.create(
            session=self.session,
            role='user',
            content='Test message'
        )
        
        self.assertIsNotNone(message.id)
        self.assertEqual(message.role, 'user')
        self.assertEqual(message.content, 'Test message')
    
    def test_message_str(self):
        message = ChatMessage.objects.create(
            session=self.session,
            role='assistant',
            content='This is a test response'
        )
        
        self.assertIn('[assistant]', str(message))
    
    def test_message_to_dict(self):
        message = ChatMessage.objects.create(
            session=self.session,
            role='user',
            content='Test'
        )
        
        data = message.to_dict()
        
        self.assertEqual(data['role'], 'user')
        self.assertEqual(data['content'], 'Test')
        self.assertIn('timestamp', data)
    
    def test_message_ordering(self):
        msg1 = ChatMessage.objects.create(
            session=self.session,
            role='user',
            content='First'
        )
        msg2 = ChatMessage.objects.create(
            session=self.session,
            role='assistant',
            content='Second'
        )
        
        messages = list(self.session.messages.all())
        
        self.assertEqual(messages[0].content, 'First')
        self.assertEqual(messages[1].content, 'Second')


class ChatFeedbackModelTest(TestCase):
    """Tests for ChatFeedback model"""
    
    def setUp(self):
        self.session = ChatSession.objects.create()
        self.message = ChatMessage.objects.create(
            session=self.session,
            role='assistant',
            content='Response'
        )
    
    def test_create_feedback(self):
        feedback = ChatFeedback.objects.create(
            message=self.message,
            rating=5,
            comment='Great!'
        )
        
        self.assertIsNotNone(feedback.id)
        self.assertEqual(feedback.rating, 5)
        self.assertEqual(feedback.comment, 'Great!')
    
    def test_feedback_str(self):
        feedback = ChatFeedback.objects.create(
            message=self.message,
            rating=4
        )
        
        self.assertIn('4/5', str(feedback))
    
    def test_one_feedback_per_message(self):
        ChatFeedback.objects.create(
            message=self.message,
            rating=5
        )
        
        # Creating another feedback for same message should raise error
        # due to OneToOneField
        with self.assertRaises(Exception):
            ChatFeedback.objects.create(
                message=self.message,
                rating=3
            )


class ConversationContextModelTest(TestCase):
    """Tests for ConversationContext model"""
    
    def setUp(self):
        self.session = ChatSession.objects.create()
    
    def test_create_context(self):
        context = ConversationContext.objects.create(
            session=self.session,
            summary='Test summary'
        )
        
        self.assertEqual(context.summary, 'Test summary')
    
    def test_add_topic(self):
        context = ConversationContext.objects.create(session=self.session)
        
        context.add_topic('books')
        context.add_topic('shipping')
        
        self.assertIn('books', context.topics)
        self.assertIn('shipping', context.topics)
    
    def test_add_duplicate_topic(self):
        context = ConversationContext.objects.create(session=self.session)
        
        context.add_topic('books')
        context.add_topic('books')
        
        self.assertEqual(context.topics.count('books'), 1)
    
    def test_add_intent(self):
        context = ConversationContext.objects.create(session=self.session)
        
        context.add_intent('product_search')
        
        self.assertEqual(len(context.intent_history), 1)
        self.assertEqual(context.intent_history[0]['intent'], 'product_search')
    
    def test_update_entity(self):
        context = ConversationContext.objects.create(session=self.session)
        
        context.update_entity('product', 'Đắc Nhân Tâm')
        
        self.assertEqual(context.entities['product'], 'Đắc Nhân Tâm')

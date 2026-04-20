"""
Django Models for Consulting Chatbot Service
"""
import uuid
from django.db import models
from django.utils import timezone


class ChatSession(models.Model):
    """
    Model cho phiên chat
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    customer_id = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    title = models.CharField(max_length=255, blank=True, default='')
    is_active = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'chat_sessions'
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['customer_id', '-updated_at']),
            models.Index(fields=['-created_at']),
        ]
    
    def __str__(self):
        return f"Session {self.id} - Customer: {self.customer_id or 'Anonymous'}"
    
    @property
    def message_count(self) -> int:
        return self.messages.count()
    
    def get_last_messages(self, limit: int = 10):
        """Lấy n messages cuối cùng"""
        return self.messages.order_by('-timestamp')[:limit]
    
    def get_conversation_history(self) -> list:
        """Lấy toàn bộ lịch sử hội thoại"""
        messages = self.messages.order_by('timestamp')
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
            }
            for msg in messages
        ]


class ChatMessage(models.Model):
    """
    Model cho tin nhắn trong phiên chat
    """
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        ChatSession, 
        on_delete=models.CASCADE, 
        related_name='messages'
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    
    class Meta:
        db_table = 'chat_messages'
        ordering = ['timestamp']
        indexes = [
            models.Index(fields=['session', 'timestamp']),
            models.Index(fields=['role']),
        ]
    
    def __str__(self):
        content_preview = self.content[:50] + '...' if len(self.content) > 50 else self.content
        return f"[{self.role}] {content_preview}"
    
    def to_dict(self) -> dict:
        return {
            'id': str(self.id),
            'role': self.role,
            'content': self.content,
            'sources': self.sources,
            'timestamp': self.timestamp.isoformat()
        }


class ChatFeedback(models.Model):
    """
    Model cho feedback của tin nhắn
    """
    RATING_CHOICES = [
        (1, '1 - Rất không hài lòng'),
        (2, '2 - Không hài lòng'),
        (3, '3 - Bình thường'),
        (4, '4 - Hài lòng'),
        (5, '5 - Rất hài lòng'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.OneToOneField(
        ChatMessage,
        on_delete=models.CASCADE,
        related_name='feedback'
    )
    rating = models.IntegerField(choices=RATING_CHOICES)
    comment = models.TextField(blank=True, null=True)
    is_helpful = models.BooleanField(null=True, blank=True)
    feedback_type = models.CharField(
        max_length=50, 
        blank=True,
        null=True,
        help_text="Type of feedback: accuracy, relevance, speed, etc."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'chat_feedbacks'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback for Message {self.message.id}: {self.rating}/5"


class ConversationContext(models.Model):
    """
    Model để lưu context của cuộc hội thoại
    Dùng cho multi-turn conversations
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='context'
    )
    summary = models.TextField(blank=True, help_text="Tóm tắt cuộc hội thoại")
    topics = models.JSONField(default=list, blank=True, help_text="Các chủ đề đã thảo luận")
    entities = models.JSONField(default=dict, blank=True, help_text="Các entity được đề cập")
    intent_history = models.JSONField(default=list, blank=True, help_text="Lịch sử intent")
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'conversation_contexts'
    
    def __str__(self):
        return f"Context for Session {self.session.id}"
    
    def add_topic(self, topic: str):
        """Thêm topic vào danh sách"""
        if topic not in self.topics:
            self.topics.append(topic)
            self.save(update_fields=['topics', 'last_updated'])
    
    def add_intent(self, intent: str):
        """Thêm intent vào history"""
        self.intent_history.append({
            'intent': intent,
            'timestamp': timezone.now().isoformat()
        })
        # Giữ tối đa 20 intents gần nhất
        self.intent_history = self.intent_history[-20:]
        self.save(update_fields=['intent_history', 'last_updated'])
    
    def update_entity(self, entity_type: str, entity_value: str):
        """Cập nhật entity"""
        self.entities[entity_type] = entity_value
        self.save(update_fields=['entities', 'last_updated'])

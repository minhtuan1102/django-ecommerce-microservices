"""
Serializers for Chat API
"""
from rest_framework import serializers
from app.models import ChatSession, ChatMessage, ChatFeedback


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer cho ChatMessage"""
    
    class Meta:
        model = ChatMessage
        fields = ['id', 'role', 'content', 'sources', 'metadata', 'timestamp']
        read_only_fields = ['id', 'timestamp']


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer cho ChatSession"""
    messages = ChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ['id', 'customer_id', 'title', 'is_active', 'metadata', 
                  'created_at', 'updated_at', 'messages', 'message_count']
        read_only_fields = ['id', 'created_at', 'updated_at', 'message_count']


class ChatSessionListSerializer(serializers.ModelSerializer):
    """Serializer cho danh sách sessions (lightweight)"""
    message_count = serializers.IntegerField(read_only=True)
    last_message = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatSession
        fields = ['id', 'customer_id', 'title', 'is_active', 
                  'created_at', 'updated_at', 'message_count', 'last_message']
    
    def get_last_message(self, obj):
        last_msg = obj.messages.order_by('-timestamp').first()
        if last_msg:
            return {
                'content': last_msg.content[:100] + '...' if len(last_msg.content) > 100 else last_msg.content,
                'role': last_msg.role,
                'timestamp': last_msg.timestamp.isoformat()
            }
        return None


class ChatFeedbackSerializer(serializers.ModelSerializer):
    """Serializer cho ChatFeedback"""
    
    class Meta:
        model = ChatFeedback
        fields = ['id', 'message', 'rating', 'comment', 'is_helpful', 
                  'feedback_type', 'created_at']
        read_only_fields = ['id', 'created_at']


# Request/Response Serializers

class SendMessageRequestSerializer(serializers.Serializer):
    """Serializer cho request gửi tin nhắn"""
    message = serializers.CharField(
        required=True,
        max_length=4000,
        help_text="Nội dung tin nhắn"
    )
    customer_id = serializers.CharField(
        required=False,
        max_length=100,
        allow_null=True,
        help_text="ID khách hàng (optional)"
    )
    session_id = serializers.UUIDField(
        required=False,
        allow_null=True,
        help_text="ID session chat (optional, tạo mới nếu không có)"
    )
    category = serializers.ChoiceField(
        choices=['products', 'policies', 'faqs'],
        required=False,
        allow_null=True,
        help_text="Category để filter search (optional)"
    )
    personalized = serializers.BooleanField(
        default=False,
        help_text="Sử dụng personalization từ behavior analysis"
    )
    
    def validate_message(self, value):
        """Validate message content"""
        if not value or not value.strip():
            raise serializers.ValidationError("Tin nhắn không được để trống")
        return value.strip()


class SendMessageResponseSerializer(serializers.Serializer):
    """Serializer cho response gửi tin nhắn"""
    response = serializers.CharField(help_text="Câu trả lời từ chatbot")
    session_id = serializers.UUIDField(help_text="ID session chat")
    sources = serializers.ListField(
        child=serializers.DictField(),
        help_text="Danh sách sources được tham khảo"
    )
    customer_id = serializers.CharField(
        allow_null=True,
        help_text="ID khách hàng"
    )
    personalized = serializers.BooleanField(help_text="Response có được personalize không")
    timestamp = serializers.DateTimeField(help_text="Thời gian response")
    metadata = serializers.DictField(
        required=False,
        help_text="Metadata bổ sung"
    )


class FeedbackRequestSerializer(serializers.Serializer):
    """Serializer cho request feedback"""
    message_id = serializers.UUIDField(
        required=True,
        help_text="ID của tin nhắn cần feedback"
    )
    rating = serializers.IntegerField(
        required=True,
        min_value=1,
        max_value=5,
        help_text="Đánh giá từ 1-5"
    )
    comment = serializers.CharField(
        required=False,
        max_length=1000,
        allow_blank=True,
        allow_null=True,
        help_text="Nhận xét bổ sung"
    )
    is_helpful = serializers.BooleanField(
        required=False,
        allow_null=True,
        help_text="Câu trả lời có hữu ích không"
    )
    feedback_type = serializers.CharField(
        required=False,
        max_length=50,
        allow_blank=True,
        allow_null=True,
        help_text="Loại feedback"
    )


class HealthCheckResponseSerializer(serializers.Serializer):
    """Serializer cho health check response"""
    status = serializers.CharField()
    service = serializers.CharField()
    version = serializers.CharField()
    timestamp = serializers.DateTimeField()
    components = serializers.DictField(required=False)


class ConversationHistorySerializer(serializers.Serializer):
    """Serializer cho conversation history"""
    session = ChatSessionSerializer()
    messages = ChatMessageSerializer(many=True)
    total_messages = serializers.IntegerField()

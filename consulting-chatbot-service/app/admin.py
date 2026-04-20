"""
Django Admin configuration for Consulting Chatbot Service
"""
from django.contrib import admin
from .models import ChatSession, ChatMessage, ChatFeedback, ConversationContext


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'customer_id', 'title', 'is_active', 'message_count', 'created_at', 'updated_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['customer_id', 'title']
    readonly_fields = ['id', 'created_at', 'updated_at']
    ordering = ['-updated_at']
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'role', 'content_preview', 'timestamp']
    list_filter = ['role', 'timestamp']
    search_fields = ['content', 'session__customer_id']
    readonly_fields = ['id', 'timestamp']
    ordering = ['-timestamp']
    
    def content_preview(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content'


@admin.register(ChatFeedback)
class ChatFeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'message', 'rating', 'is_helpful', 'feedback_type', 'created_at']
    list_filter = ['rating', 'is_helpful', 'feedback_type', 'created_at']
    search_fields = ['comment']
    readonly_fields = ['id', 'created_at']
    ordering = ['-created_at']


@admin.register(ConversationContext)
class ConversationContextAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'topics_count', 'last_updated']
    readonly_fields = ['id', 'last_updated']
    ordering = ['-last_updated']
    
    def topics_count(self, obj):
        return len(obj.topics) if obj.topics else 0
    topics_count.short_description = 'Topics'

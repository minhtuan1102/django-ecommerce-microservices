"""
API Views for Consulting Chatbot Service
"""
import logging
from datetime import datetime

from django.utils import timezone
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from app.models import ChatSession, ChatMessage, ChatFeedback
from app.rag.chain import ConsultingChain
from app.rag.retriever import KnowledgeRetriever
from app.rag.generator import ResponseGenerator
from app.utils.behavior_client import BehaviorClient
from .serializers import (
    SendMessageRequestSerializer,
    SendMessageResponseSerializer,
    FeedbackRequestSerializer,
    ChatSessionSerializer,
    ChatMessageSerializer,
    ChatFeedbackSerializer,
    HealthCheckResponseSerializer,
)

logger = logging.getLogger(__name__)

# Initialize RAG chain (singleton)
_chain_instance = None


def get_chain() -> ConsultingChain:
    """Get or create ConsultingChain singleton"""
    global _chain_instance
    if _chain_instance is None:
        retriever = KnowledgeRetriever()
        generator = ResponseGenerator()
        behavior_client = BehaviorClient()
        _chain_instance = ConsultingChain(
            retriever=retriever,
            generator=generator,
            behavior_client=behavior_client
        )
    return _chain_instance


class SendMessageView(APIView):
    """
    POST /api/chat/message
    Gửi tin nhắn và nhận phản hồi từ chatbot
    """
    
    def post(self, request):
        # Validate request
        serializer = SendMessageRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid request', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        data = serializer.validated_data
        message = data['message']
        customer_id = data.get('customer_id')
        session_id = data.get('session_id')
        category = data.get('category')
        personalized = data.get('personalized', False)
        
        try:
            # Get or create session
            if session_id:
                try:
                    session = ChatSession.objects.get(id=session_id)
                except ChatSession.DoesNotExist:
                    session = ChatSession.objects.create(
                        id=session_id,
                        customer_id=customer_id
                    )
            else:
                session = ChatSession.objects.create(customer_id=customer_id)
            
            # Save user message
            user_message = ChatMessage.objects.create(
                session=session,
                role='user',
                content=message
            )
            
            # Get chain and process
            chain = get_chain()
            
            if personalized and customer_id:
                response_obj = chain.get_personalized_response(
                    query=message,
                    customer_id=customer_id,
                    session_id=str(session.id)
                )
            else:
                response_obj = chain.process(
                    query=message,
                    customer_id=customer_id,
                    session_id=str(session.id),
                    category=category
                )
            
            # Save assistant message
            assistant_message = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=response_obj.response,
                sources=response_obj.sources,
                metadata=response_obj.metadata
            )
            
            # Update session
            session.updated_at = timezone.now()
            if not session.title and message:
                session.title = message[:100]
            session.save()
            
            # Build response
            response_data = {
                'response': response_obj.response,
                'session_id': str(session.id),
                'sources': response_obj.sources,
                'customer_id': customer_id,
                'personalized': response_obj.personalized,
                'timestamp': assistant_message.timestamp,
                'metadata': response_obj.metadata
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return Response(
                {'error': 'Internal server error', 'message': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatHistoryView(APIView):
    """
    GET /api/chat/history/{session_id}
    Lấy lịch sử chat của session
    """
    
    def get(self, request, session_id):
        try:
            session = get_object_or_404(ChatSession, id=session_id)
            
            # Get messages
            messages = session.messages.order_by('timestamp')
            
            # Serialize
            session_serializer = ChatSessionSerializer(session)
            messages_serializer = ChatMessageSerializer(messages, many=True)
            
            return Response({
                'session': session_serializer.data,
                'messages': messages_serializer.data,
                'total_messages': messages.count()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return Response(
                {'error': 'Session not found or error occurred'},
                status=status.HTTP_404_NOT_FOUND
            )


class SubmitFeedbackView(APIView):
    """
    POST /api/chat/feedback
    Gửi feedback cho tin nhắn
    """
    
    def post(self, request):
        serializer = FeedbackRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid request', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        data = serializer.validated_data
        message_id = data['message_id']
        
        try:
            # Get message
            message = get_object_or_404(ChatMessage, id=message_id)
            
            # Check if feedback already exists
            if hasattr(message, 'feedback'):
                # Update existing feedback
                feedback = message.feedback
                feedback.rating = data['rating']
                feedback.comment = data.get('comment')
                feedback.is_helpful = data.get('is_helpful')
                feedback.feedback_type = data.get('feedback_type')
                feedback.save()
            else:
                # Create new feedback
                feedback = ChatFeedback.objects.create(
                    message=message,
                    rating=data['rating'],
                    comment=data.get('comment'),
                    is_helpful=data.get('is_helpful'),
                    feedback_type=data.get('feedback_type')
                )
            
            feedback_serializer = ChatFeedbackSerializer(feedback)
            return Response({
                'message': 'Feedback submitted successfully',
                'feedback': feedback_serializer.data
            }, status=status.HTTP_201_CREATED)
            
        except ChatMessage.DoesNotExist:
            return Response(
                {'error': 'Message not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return Response(
                {'error': 'Error submitting feedback'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HealthCheckView(APIView):
    """
    GET /api/chat/health
    Health check endpoint
    """
    
    def get(self, request):
        try:
            chain = get_chain()
            chain_health = chain.get_health_status()
            
            response_data = {
                'status': chain_health.get('status', 'healthy'),
                'service': 'consulting-chatbot-service',
                'version': '1.0.0',
                'timestamp': timezone.now(),
                'components': chain_health.get('components', {})
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return Response({
                'status': 'unhealthy',
                'service': 'consulting-chatbot-service',
                'version': '1.0.0',
                'timestamp': timezone.now(),
                'error': str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


class CustomerSessionsView(APIView):
    """
    GET /api/chat/sessions/{customer_id}
    Lấy danh sách sessions của khách hàng
    """
    
    def get(self, request, customer_id):
        try:
            sessions = ChatSession.objects.filter(
                customer_id=customer_id
            ).order_by('-updated_at')[:20]
            
            sessions_data = []
            for session in sessions:
                sessions_data.append({
                    'id': str(session.id),
                    'title': session.title or 'Cuộc hội thoại',
                    'message_count': session.messages.count(),
                    'created_at': session.created_at.isoformat(),
                    'updated_at': session.updated_at.isoformat(),
                    'is_active': session.is_active
                })
            
            return Response({
                'customer_id': customer_id,
                'sessions': sessions_data,
                'total': len(sessions_data)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting customer sessions: {e}")
            return Response(
                {'error': 'Error retrieving sessions'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SearchKnowledgeView(APIView):
    """
    GET /api/chat/search
    Tìm kiếm trong knowledge base
    """
    
    def get(self, request):
        query = request.query_params.get('q', '')
        category = request.query_params.get('category')
        limit = int(request.query_params.get('limit', 5))
        
        if not query:
            return Response(
                {'error': 'Query parameter "q" is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            chain = get_chain()
            
            if category:
                documents = chain.retriever.search_by_category(query, category, limit)
            else:
                documents = chain.retriever.retrieve(query, limit)
            
            results = []
            for doc in documents:
                results.append({
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'score': doc.score
                })
            
            return Response({
                'query': query,
                'category': category,
                'results': results,
                'total': len(results)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return Response(
                {'error': 'Error searching knowledge base'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Function-based views for simpler endpoints

@api_view(['DELETE'])
def delete_session(request, session_id):
    """
    DELETE /api/chat/session/{session_id}
    Xóa session chat
    """
    try:
        session = get_object_or_404(ChatSession, id=session_id)
        session.delete()
        return Response(
            {'message': 'Session deleted successfully'},
            status=status.HTTP_200_OK
        )
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return Response(
            {'error': 'Error deleting session'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def clear_session(request, session_id):
    """
    POST /api/chat/session/{session_id}/clear
    Xóa tất cả messages trong session
    """
    try:
        session = get_object_or_404(ChatSession, id=session_id)
        deleted_count = session.messages.all().delete()[0]
        return Response({
            'message': f'Cleared {deleted_count} messages from session',
            'session_id': str(session_id)
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return Response(
            {'error': 'Error clearing session'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

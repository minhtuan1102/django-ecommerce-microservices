"""
API Views cho Behavior Analysis Service
"""
import time
from datetime import datetime
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
import logging

from app.models import CustomerBehavior, BehaviorEvent, AnalysisResult
from app.services.behavior_analyzer import get_analyzer, BehaviorAnalyzer
from app.services.data_collector import DataCollector
from app.services.graph import get_graph_service

logger = logging.getLogger(__name__)


class CustomerAnalysisView(APIView):
    """API để phân tích hành vi customer"""
    
    def get(self, request, customer_id):
        """
        GET /api/behavior/customer/{id}/analysis
        Phân tích đầy đủ hành vi customer
        """
        start_time = time.time()
        
        try:
            # Get analyzer
            analyzer = get_analyzer()
            
            # Fetch customer data từ services
            collector = DataCollector()
            customers = collector.fetch_customers()
            orders = collector.fetch_orders()
            reviews = collector.fetch_reviews()
            
            # Find customer
            customer_data = None
            for c in customers:
                if c.get('id') == customer_id or c.get('customer_id') == customer_id:
                    customer_data = c
                    break
            
            # If not found in service, try to create mock data for demo
            if not customer_data:
                customer_data = {
                    'customer_id': customer_id,
                    'id': customer_id,
                    'age': 30,
                    'gender': 'M',
                    'job': 'Software Engineer',
                    'location': 'Ha Noi'
                }
            
            # Filter orders cho customer này
            customer_orders = [o for o in orders 
                            if o.get('customer_id') == customer_id or 
                               o.get('user_id') == customer_id]
            
            # Get insights
            result = analyzer.get_customer_insights(
                customer_data, 
                customer_orders
            )
            
            # Save to database
            processing_time = int((time.time() - start_time) * 1000)
            
            behavior, created = CustomerBehavior.objects.update_or_create(
                customer_id=customer_id,
                defaults={
                    'segment': result.get('segment', 'Regular'),
                    'segment_confidence': result.get('segment_confidence', 0.0),
                    'segment_probabilities': result.get('segment_probabilities', {}),
                    'churn_risk': result.get('churn_risk', 0.0),
                    'churn_level': result.get('churn_level', 'Low'),
                    'predicted_categories': result.get('predicted_categories', []),
                    'engagement_score': result.get('engagement_score', 50.0),
                }
            )
            
            # Save analysis result
            AnalysisResult.create_result(
                customer_id=customer_id,
                analysis_type='full_analysis',
                result=result,
                processing_time_ms=processing_time
            )
            
            return Response({
                'success': True,
                'data': result,
                'processing_time_ms': processing_time
            })
            
        except Exception as e:
            logger.error(f"Error analyzing customer {customer_id}: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CustomerSegmentView(APIView):
    """API để lấy segment của customer"""
    
    def get(self, request, customer_id):
        """
        GET /api/behavior/customer/{id}/segment
        Lấy segment classification
        """
        try:
            # Try to get from database first
            try:
                behavior = CustomerBehavior.objects.get(customer_id=customer_id)
                return Response({
                    'success': True,
                    'data': {
                        'customer_id': customer_id,
                        'segment': behavior.segment,
                        'segment_confidence': behavior.segment_confidence,
                        'segment_probabilities': behavior.segment_probabilities,
                        'description': behavior.get_segment_description(),
                        'last_analyzed': behavior.last_analyzed.isoformat()
                    }
                })
            except CustomerBehavior.DoesNotExist:
                pass
            
            # Run analysis
            analyzer = get_analyzer()
            customer_data = {'customer_id': customer_id, 'id': customer_id}
            
            result = analyzer.analyze_customer(customer_data)
            
            return Response({
                'success': True,
                'data': {
                    'customer_id': customer_id,
                    'segment': result.get('segment', 'Unknown'),
                    'segment_confidence': result.get('segment_confidence', 0.0),
                    'segment_probabilities': result.get('segment_probabilities', {}),
                    'analyzed_at': result.get('analyzed_at')
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting segment for customer {customer_id}: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CustomerPredictionsView(APIView):
    """API để lấy predictions cho customer"""
    
    def get(self, request, customer_id):
        """
        GET /api/behavior/customer/{id}/predictions
        Lấy churn prediction và category prediction
        """
        try:
            analyzer = get_analyzer()
            
            # Get customer data
            customer_data = {'customer_id': customer_id, 'id': customer_id}
            
            # Get churn prediction
            churn_result = analyzer.predict_churn_risk(customer_data)
            
            # Get full analysis for category predictions
            analysis = analyzer.analyze_customer(customer_data)
            
            return Response({
                'success': True,
                'data': {
                    'customer_id': customer_id,
                    'churn_prediction': {
                        'probability': churn_result.get('churn_probability', 0.0),
                        'level': churn_result.get('churn_level', 'Unknown'),
                        'risk_factors': churn_result.get('risk_factors', []),
                        'retention_suggestions': churn_result.get('retention_suggestions', [])
                    },
                    'category_prediction': {
                        'predicted_categories': analysis.get('predicted_categories', [])
                    },
                    'predicted_at': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting predictions for customer {customer_id}: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TrackBehaviorView(APIView):
    """API để track behavior events"""
    
    def post(self, request):
        """
        POST /api/behavior/track
        Track một behavior event
        
        Body:
        {
            "customer_id": 123,
            "event_type": "product_view",
            "event_data": {...},
            "session_id": "...",
            "device": "mobile",
            "category": "Technology",
            "product_id": 456
        }
        """
        try:
            data = request.data
            
            # Validate required fields
            customer_id = data.get('customer_id')
            event_type = data.get('event_type')
            
            if not customer_id:
                return Response({
                    'success': False,
                    'error': 'customer_id is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if not event_type:
                return Response({
                    'success': False,
                    'error': 'event_type is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate event_type
            valid_event_types = [choice[0] for choice in BehaviorEvent.EVENT_TYPE_CHOICES]
            if event_type not in valid_event_types:
                return Response({
                    'success': False,
                    'error': f'Invalid event_type. Must be one of: {valid_event_types}'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get client info
            ip_address = self._get_client_ip(request)
            user_agent = request.META.get('HTTP_USER_AGENT', '')
            referrer = request.META.get('HTTP_REFERER', '')
            
            # Create event
            event = BehaviorEvent.track(
                customer_id=customer_id,
                event_type=event_type,
                event_data=data.get('event_data', {}),
                session_id=data.get('session_id'),
                device=data.get('device', 'desktop'),
                category=data.get('category'),
                product_id=data.get('product_id'),
                ip_address=ip_address,
                user_agent=user_agent,
                referrer=referrer
            )
            
            # Log to Neo4j if product_id is available
            product_id = data.get('product_id')
            if product_id:
                try:
                    graph_service = get_graph_service()
                    graph_service.log_interaction(
                        user_id=customer_id,
                        product_id=product_id,
                        event_type=event_type
                    )
                except Exception as ge:
                    logger.warning(f"Failed to log to Neo4j: {ge}")
            
            return Response({
                'success': True,
                'data': {
                    'event_id': event.id,
                    'customer_id': event.customer_id,
                    'event_type': event.event_type,
                    'timestamp': event.timestamp.isoformat()
                }
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error tracking behavior: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _get_client_ip(self, request):
        """Get client IP from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class HealthCheckView(APIView):
    """Health check endpoint"""
    
    def get(self, request):
        """
        GET /api/behavior/health
        Health check với thông tin chi tiết
        """
        try:
            # Check analyzer
            analyzer = get_analyzer()
            model_loaded = analyzer.is_loaded()
            
            # Check database
            db_ok = False
            try:
                from django.db import connection
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                db_ok = True
            except:
                pass
            
            # Get stats
            stats = {}
            try:
                stats = {
                    'total_behaviors': CustomerBehavior.objects.count(),
                    'total_events': BehaviorEvent.objects.count(),
                    'total_analyses': AnalysisResult.objects.count(),
                }
            except:
                pass
            
            is_healthy = db_ok  # Model không bắt buộc phải loaded
            
            return Response({
                'status': 'healthy' if is_healthy else 'unhealthy',
                'service': 'behavior-analysis-service',
                'timestamp': datetime.now().isoformat(),
                'checks': {
                    'database': 'ok' if db_ok else 'error',
                    'model_loaded': model_loaded,
                },
                'stats': stats
            }, status=status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return Response({
                'status': 'unhealthy',
                'error': str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


class BulkAnalysisView(APIView):
    """API để phân tích hàng loạt customers"""
    
    def post(self, request):
        """
        POST /api/behavior/bulk-analysis
        Phân tích nhiều customers cùng lúc
        
        Body:
        {
            "customer_ids": [1, 2, 3, ...]
        }
        """
        try:
            customer_ids = request.data.get('customer_ids', [])
            
            if not customer_ids:
                return Response({
                    'success': False,
                    'error': 'customer_ids is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if len(customer_ids) > 100:
                return Response({
                    'success': False,
                    'error': 'Maximum 100 customers per request'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            analyzer = get_analyzer()
            results = []
            
            for customer_id in customer_ids:
                customer_data = {'customer_id': customer_id, 'id': customer_id}
                analysis = analyzer.analyze_customer(customer_data)
                results.append(analysis)
            
            return Response({
                'success': True,
                'data': {
                    'total': len(results),
                    'results': results
                }
            })
            
        except Exception as e:
            logger.error(f"Error in bulk analysis: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CustomerEventsView(APIView):
    """API để lấy events của customer"""
    
    def get(self, request, customer_id):
        """
        GET /api/behavior/customer/{id}/events
        Lấy danh sách events của customer
        """
        try:
            # Pagination
            page = int(request.query_params.get('page', 1))
            page_size = int(request.query_params.get('page_size', 20))
            
            # Filter
            event_type = request.query_params.get('event_type')
            
            # Query
            events = BehaviorEvent.objects.filter(customer_id=customer_id)
            
            if event_type:
                events = events.filter(event_type=event_type)
            
            total = events.count()
            
            # Paginate
            start = (page - 1) * page_size
            events = events[start:start + page_size]
            
            return Response({
                'success': True,
                'data': {
                    'customer_id': customer_id,
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'events': [
                        {
                            'event_id': e.id,
                            'event_type': e.event_type,
                            'event_data': e.event_data,
                            'category': e.category,
                            'device': e.device,
                            'timestamp': e.timestamp.isoformat()
                        }
                        for e in events
                    ]
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting events for customer {customer_id}: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SegmentSummaryView(APIView):
    """API để lấy tổng hợp theo segment"""
    
    def get(self, request):
        """
        GET /api/behavior/segments/summary
        Lấy thống kê tổng hợp theo segment
        """
        try:
            from django.db.models import Count, Avg
            
            # Aggregate by segment
            summary = CustomerBehavior.objects.values('segment').annotate(
                count=Count('id'),
                avg_churn_risk=Avg('churn_risk'),
                avg_engagement=Avg('engagement_score')
            )
            
            # Format result
            segments = {}
            for item in summary:
                segments[item['segment']] = {
                    'count': item['count'],
                    'avg_churn_risk': round(item['avg_churn_risk'] or 0, 4),
                    'avg_engagement_score': round(item['avg_engagement'] or 0, 2)
                }
            
            total = CustomerBehavior.objects.count()
            
            return Response({
                'success': True,
                'data': {
                    'total_customers': total,
                    'segments': segments,
                    'generated_at': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting segment summary: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CustomerGraphRecommendationsView(APIView):
    """API để lấy recommendations từ Neo4j Graph"""
    
    def get(self, request, customer_id):
        """
        GET /api/behavior/customer/{id}/graph-recommendations
        """
        try:
            limit = int(request.query_params.get('limit', 5))
            graph_service = get_graph_service()
            
            # Get recommendations from graph
            rec_ids = graph_service.get_recommendations(customer_id, limit=limit)
            
            # Get user context
            context = graph_service.get_user_context(customer_id)
            
            return Response({
                'success': True,
                'data': {
                    'customer_id': customer_id,
                    'recommendations': rec_ids,
                    'graph_context': context,
                    'generated_at': datetime.now().isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Error getting graph recommendations: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

"""
Behavior Client - Gọi API của behavior-analysis-service
"""
import logging
from typing import Dict, Any, List, Optional
import requests
from requests.exceptions import RequestException, Timeout

from django.conf import settings

logger = logging.getLogger(__name__)


class BehaviorClient:
    """
    Client để gọi API của behavior-analysis-service
    """
    
    def __init__(
        self,
        behavior_service_url: str = None,
        timeout: int = 10
    ):
        """
        Khởi tạo behavior client
        
        Args:
            behavior_service_url: URL của behavior-analysis-service
            timeout: Request timeout (seconds)
        """
        self.base_url = behavior_service_url or getattr(
            settings, 'BEHAVIOR_SERVICE_URL',
            'http://behavior-analysis-service:8000'
        )
        self.timeout = timeout
        self._session = None
        
        logger.info(f"BehaviorClient initialized with base_url: {self.base_url}")
    
    @property
    def session(self) -> requests.Session:
        """Lazy initialization của requests session"""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
        return self._session
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        params: dict = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to behavior service
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data or None if error
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Timeout:
            logger.error(f"Timeout calling behavior service: {url}")
            return None
        except RequestException as e:
            logger.error(f"Error calling behavior service: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def get_customer_analysis(self, customer_id: str) -> Dict[str, Any]:
        """
        Lấy phân tích hành vi đầy đủ của khách hàng
        """
        result = self._make_request(
            'GET',
            f'/api/behavior/customer/{customer_id}/analysis/'
        )
        
        if result and result.get('success'):
            return result.get('data', {})
        
        # Return mock data if service unavailable
        logger.warning(f"Using mock data for customer {customer_id}")
        return self._get_mock_customer_analysis(customer_id)
    
    def get_customer_segment(self, customer_id: str) -> str:
        """
        Lấy phân khúc khách hàng
        """
        result = self._make_request(
            'GET',
            f'/api/behavior/customer/{customer_id}/segment/'
        )
        
        if result and result.get('success'):
            return result['data'].get('segment', 'Regular')
        
        # Try getting from full analysis
        analysis = self.get_customer_analysis(customer_id)
        return analysis.get('segment', 'Regular')
    
    def get_recommendations(
        self,
        customer_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Lấy danh sách sản phẩm được đề xuất cho khách hàng (Graph-based)
        """
        result = self._make_request(
            'GET',
            f'/api/behavior/customer/{customer_id}/graph-recommendations/',
            params={'limit': limit}
        )
        
        if result and result.get('success'):
            return result['data'].get('recommendations', [])
        
        # Return mock recommendations
        return self._get_mock_recommendations()

    def get_graph_context(self, customer_id: str) -> Dict[str, Any]:
        """
        Lấy ngữ cảnh đồ thị cho GraphRAG
        """
        result = self._make_request(
            'GET',
            f'/api/behavior/customer/{customer_id}/graph-recommendations/'
        )
        
        if result and result.get('success'):
            return result['data'].get('graph_context', {})
            
        return {}
    
    def track_interaction(
        self,
        customer_id: str,
        event_type: str,
        product_id: str = None,
        event_data: Dict[str, Any] = None
    ) -> bool:
        """
        Ghi nhận tương tác của khách hàng
        """
        data = {
            'customer_id': customer_id,
            'event_type': event_type,
            'event_data': event_data or {}
        }
        if product_id:
            data['product_id'] = product_id
            
        result = self._make_request(
            'POST',
            '/api/behavior/track/',
            data=data
        )
        
        return result is not None and result.get('success', False)
    
    def health_check(self) -> bool:
        """
        Kiểm tra health của behavior service
        """
        result = self._make_request('GET', '/api/behavior/health/')
        return result is not None and result.get('status') == 'healthy'
    
    def _get_mock_customer_analysis(self, customer_id: str) -> Dict[str, Any]:
        """Get mock customer analysis data"""
        import hashlib
        hash_val = int(hashlib.md5(str(customer_id).encode()).hexdigest()[:8], 16)
        
        segments = ['VIP', 'Loyal', 'Regular', 'New', 'At-Risk']
        segment = segments[hash_val % len(segments)]
        
        categories = ['Văn học', 'Kinh tế', 'Self-help', 'Thiếu nhi', 'Khoa học', 'Lịch sử']
        favorite_cats = categories[hash_val % 3: (hash_val % 3) + 2]
        
        return {
            'customer_id': customer_id,
            'segment': segment,
            'favorite_categories': favorite_cats,
            'purchase_frequency': 'Cao' if segment == 'VIP' else 'Trung bình',
            'avg_order_value': '500.000đ' if segment == 'VIP' else '200.000đ',
            'total_orders': (hash_val % 50) + 1,
            'last_purchase_date': '2024-01-15',
            'preferred_authors': ['Nguyễn Nhật Ánh', 'Dale Carnegie'],
            'price_range': {
                'min': 50000,
                'max': 300000,
                'avg': 150000
            }
        }
    
    def _get_mock_recommendations(self) -> List[Dict[str, Any]]:
        """Get mock product recommendations"""
        return [
            {'id': 1, 'name': 'Đắc Nhân Tâm', 'score': 0.95},
            {'id': 2, 'name': 'Nhà Giả Kim', 'score': 0.92},
            {'id': 3, 'name': 'Tuổi Trẻ Đáng Giá Bao Nhiêu', 'score': 0.88}
        ]
    
    def close(self):
        """Close the session"""
        if self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

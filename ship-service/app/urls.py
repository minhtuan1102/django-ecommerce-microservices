from django.urls import path
from .views import ShipmentListCreate, ShipmentDetail, ShipmentByOrder

urlpatterns = [
    # Paths are now relative to /shipping/ gateway prefix
    path('', ShipmentListCreate.as_view()),
    path('<int:pk>/', ShipmentDetail.as_view()),
    path('order/<int:order_id>/', ShipmentByOrder.as_view()),
]

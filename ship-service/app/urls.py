from django.urls import path
from .views import ShipmentListCreate, ShipmentDetail, ShipmentByOrder

urlpatterns = [
    path('shipments/', ShipmentListCreate.as_view()),
    path('shipments/<int:pk>/', ShipmentDetail.as_view()),
    path('shipments/order/<int:order_id>/', ShipmentByOrder.as_view()),
]

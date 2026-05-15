from django.urls import path
from .views import OrderListCreate, OrderDetail

urlpatterns = [
    # Paths are now relative to /orders/ gateway prefix
    path('', OrderListCreate.as_view()),
    path('customer/<int:customer_id>/', OrderListCreate.as_view()),
    path('<int:pk>/', OrderDetail.as_view()),
]

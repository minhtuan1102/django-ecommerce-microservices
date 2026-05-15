from django.urls import path
from .views import PaymentListCreate, PaymentDetail

urlpatterns = [
    # Paths are now relative to /payment/ gateway prefix
    path('', PaymentListCreate.as_view()),
    path('<int:pk>/', PaymentDetail.as_view()),
]

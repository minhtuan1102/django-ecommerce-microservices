from django.urls import path
from .views import PaymentListCreate, PaymentDetail, PaymentByOrder, PaymentConfirm

urlpatterns = [
    path('payments/', PaymentListCreate.as_view()),
    path('payments/<int:pk>/', PaymentDetail.as_view()),
    path('payments/order/<int:order_id>/', PaymentByOrder.as_view()),
    path('payments/confirm-payment/', PaymentConfirm.as_view()),
]

from django.urls import path
from .views import CustomerLogin, CustomerDetail, JobList, CustomerListCreate

urlpatterns = [
    path('customers/', CustomerListCreate.as_view()),
    path('customers/login/', CustomerLogin.as_view()),
    path('customers/<int:pk>/', CustomerDetail.as_view()),
    path('jobs/', JobList.as_view()),
]

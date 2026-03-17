from django.urls import path
from .views import RecommendForCustomer, PopularBooks

urlpatterns = [
    path('recommendations/<int:customer_id>/', RecommendForCustomer.as_view()),
    path('popular/', PopularBooks.as_view()),
]

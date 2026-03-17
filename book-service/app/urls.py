from django.urls import path
from .views import BookListCreate, BookDetail, BookReduceStock, BookRestoreStock

urlpatterns = [
    path('books/', BookListCreate.as_view()),
    path('books/<int:pk>/', BookDetail.as_view()),
    path('books/<int:pk>/reduce-stock/', BookReduceStock.as_view()),
    path('books/<int:pk>/restore-stock/', BookRestoreStock.as_view()),
]

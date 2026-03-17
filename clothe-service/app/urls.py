from django.urls import path
from .views import ClotheListCreate, ClotheDetail, ClotheReduceStock, ClotheRestoreStock

urlpatterns = [
    path('clothes/', ClotheListCreate.as_view()),
    path('clothes/<int:pk>/', ClotheDetail.as_view()),
    path('clothes/<int:pk>/reduce-stock/', ClotheReduceStock.as_view()),
    path('clothes/<int:pk>/restore-stock/', ClotheRestoreStock.as_view()),
]

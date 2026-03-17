from django.urls import path
from .views import CategoryListCreate, CategoryDetail, BookCatalogListCreate

urlpatterns = [
    path('categories/', CategoryListCreate.as_view()),
    path('categories/<int:pk>/', CategoryDetail.as_view()),
    path('book-catalogs/', BookCatalogListCreate.as_view()),
]

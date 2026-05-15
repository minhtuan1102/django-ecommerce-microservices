from django.urls import path
from .views import (
    HealthCheck,
    CategoryListCreate,
    CategoryDetail,
    BookCatalogListCreate,
    ProductCatalogListCreate,
    ProductCatalogDetail,
    ProductStockAdjust,
    SeedCatalogData,
    SimilarProducts,
    SyncExternalProducts,
)

urlpatterns = [
    path('health/', HealthCheck.as_view()),
    path('categories/', CategoryListCreate.as_view()),
    path('categories/<int:pk>/', CategoryDetail.as_view()),
    path('book-catalogs/', BookCatalogListCreate.as_view()),
    
    # Paths are now relative to /products/ gateway prefix
    path('', ProductCatalogListCreate.as_view()),
    path('<int:pk>/', ProductCatalogDetail.as_view()),
    path('<int:pk>/similar/', SimilarProducts.as_view()),
    path('<int:pk>/reduce-stock/', ProductStockAdjust.as_view(), {'action': 'reduce'}),
    path('<int:pk>/restore-stock/', ProductStockAdjust.as_view(), {'action': 'restore'}),
    
    path('seed/sample-data/', SeedCatalogData.as_view()),
    path('integrations/sync/', SyncExternalProducts.as_view()),
]

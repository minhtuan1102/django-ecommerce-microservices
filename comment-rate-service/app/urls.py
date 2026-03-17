from django.urls import path
from .views import ReviewListCreate, ReviewsByBook, ReviewDetail

urlpatterns = [
    path('reviews/', ReviewListCreate.as_view()),
    path('reviews/book/<int:book_id>/', ReviewsByBook.as_view()),
    path('reviews/<int:pk>/', ReviewDetail.as_view()),
]

from django.urls import path
from .views import RegisterView, LoginView, ValidateTokenView, UserListCreate, UserDetail, JobList

urlpatterns = [
    path('register/', RegisterView.as_view()),
    path('login/', LoginView.as_view()),
    path('validate/', ValidateTokenView.as_view()),
    path('users/', UserListCreate.as_view()),
    path('users/<int:pk>/', UserDetail.as_view()),
    path('jobs/', JobList.as_view()),
]

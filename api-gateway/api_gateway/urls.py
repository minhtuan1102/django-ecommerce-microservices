"""
URL configuration for api_gateway project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse, JsonResponse


def health(request):
    return JsonResponse({'status': 'healthy', 'service': 'api-gateway'})


def metrics(request):
    payload = [
        '# HELP gateway_up Gateway process availability',
        '# TYPE gateway_up gauge',
        'gateway_up 1',
    ]
    return HttpResponse('\n'.join(payload) + '\n', content_type='text/plain; version=0.0.4')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health),
    path('metrics/', metrics),
    path('', include('app.urls')),
]

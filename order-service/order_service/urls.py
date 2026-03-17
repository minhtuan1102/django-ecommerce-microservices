from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse, JsonResponse

def health_check(request):
    return JsonResponse({"status": "healthy", "service": "order-service"})


def metrics(request):
    payload = [
        '# HELP order_service_up Service up indicator',
        '# TYPE order_service_up gauge',
        'order_service_up 1',
    ]
    return HttpResponse('\n'.join(payload) + '\n', content_type='text/plain; version=0.0.4')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health_check),
    path('metrics/', metrics),
    path('', include('app.urls')),
]

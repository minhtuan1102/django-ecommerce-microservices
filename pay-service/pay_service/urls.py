from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse, JsonResponse


def health(request):
    return JsonResponse({'status': 'healthy', 'service': 'pay-service'})


def metrics(request):
    payload = [
        '# HELP pay_service_up Service up indicator',
        '# TYPE pay_service_up gauge',
        'pay_service_up 1',
    ]
    return HttpResponse('\n'.join(payload) + '\n', content_type='text/plain; version=0.0.4')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health),
    path('metrics/', metrics),
    path('', include('app.urls')),
]

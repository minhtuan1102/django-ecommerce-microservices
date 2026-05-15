from django.contrib import admin
from django.http import HttpResponse, JsonResponse
from django.urls import include, path
from django.conf import settings


def health(request):
    return JsonResponse({'status': 'healthy', 'service': 'auth-service'})


def metrics(request):
    m = settings.METRICS
    payload = [
        '# HELP auth_tokens_issued_total Total issued access tokens',
        '# TYPE auth_tokens_issued_total counter',
        f"auth_tokens_issued_total {m['tokens_issued']}",
        '# HELP auth_token_validation_success_total Successful validations',
        '# TYPE auth_token_validation_success_total counter',
        f"auth_token_validation_success_total {m['token_validation_success']}",
        '# HELP auth_token_validation_failed_total Failed validations',
        '# TYPE auth_token_validation_failed_total counter',
        f"auth_token_validation_failed_total {m['token_validation_failed']}",
        '# HELP auth_login_failed_total Failed login attempts',
        '# TYPE auth_login_failed_total counter',
        f"auth_login_failed_total {m['logins_failed']}",
    ]
    return HttpResponse('\n'.join(payload) + '\n', content_type='text/plain; version=0.0.4')


urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health),
    path('metrics/', metrics),
    path('', include('app.urls')),
]

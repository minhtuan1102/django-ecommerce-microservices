from datetime import datetime, timedelta, timezone

import jwt
from django.conf import settings
from django.contrib.auth.hashers import check_password, make_password
from django.core.cache import cache
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import AuthUser
from .serializers import AuthUserSerializer, LoginSerializer, RegisterSerializer


class RateLimitMixin:
    def _rate_limit(self, request):
        ip = request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip() or request.META.get('REMOTE_ADDR', 'unknown')
        key = f'auth:rl:{ip}'
        current = cache.get(key, 0) + 1
        cache.set(key, current, timeout=settings.RATE_LIMIT_WINDOW_SECONDS)
        return current <= settings.RATE_LIMIT_MAX_REQUESTS


def _issue_token(user):
    now = datetime.now(timezone.utc)
    payload = {
        'sub': str(user.id),
        'email': user.email,
        'role': user.role,
        'iat': int(now.timestamp()),
        'exp': int((now + timedelta(minutes=settings.JWT_ACCESS_MINUTES)).timestamp()),
        'iss': settings.JWT_ISSUER,
        'aud': settings.JWT_AUDIENCE,
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    settings.METRICS['tokens_issued'] += 1
    return token


def _decode_token(token):
    return jwt.decode(
        token,
        settings.JWT_SECRET,
        algorithms=[settings.JWT_ALGORITHM],
        audience=settings.JWT_AUDIENCE,
        issuer=settings.JWT_ISSUER,
    )


class RegisterView(RateLimitMixin, APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        if not self._rate_limit(request):
            return Response({'error': 'Rate limit exceeded'}, status=429)

        serializer = RegisterSerializer(data=request.data)
        if not serializer.is_valid():
            settings.METRICS['register_failed'] += 1
            return Response(serializer.errors, status=400)

        email = serializer.validated_data['email'].strip().lower()
        password = serializer.validated_data['password']
        role = serializer.validated_data.get('role', 'customer')

        if AuthUser.objects.filter(email=email).exists():
            settings.METRICS['register_failed'] += 1
            return Response({'error': 'Email already exists'}, status=409)

        user = AuthUser.objects.create(
            email=email,
            password=make_password(password),
            role=role,
            is_active=True,
        )
        settings.METRICS['register_success'] += 1

        token = _issue_token(user)
        return Response(
            {
                'user': AuthUserSerializer(user).data,
                'access': token,
                'token_type': 'Bearer',
                'expires_in_minutes': settings.JWT_ACCESS_MINUTES,
            },
            status=status.HTTP_201_CREATED,
        )


class LoginView(RateLimitMixin, APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        if not self._rate_limit(request):
            return Response({'error': 'Rate limit exceeded'}, status=429)

        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            settings.METRICS['logins_failed'] += 1
            return Response(serializer.errors, status=400)

        email = serializer.validated_data['email'].strip().lower()
        password = serializer.validated_data['password']

        try:
            user = AuthUser.objects.get(email=email, is_active=True)
        except AuthUser.DoesNotExist:
            settings.METRICS['logins_failed'] += 1
            return Response({'error': 'Invalid credentials'}, status=401)

        if not check_password(password, user.password):
            settings.METRICS['logins_failed'] += 1
            return Response({'error': 'Invalid credentials'}, status=401)

        token = _issue_token(user)
        return Response(
            {
                'user': AuthUserSerializer(user).data,
                'access': token,
                'token_type': 'Bearer',
                'expires_in_minutes': settings.JWT_ACCESS_MINUTES,
            }
        )


class ValidateTokenView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        token = request.data.get('token')
        if not token:
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ', 1)[1]

        if not token:
            settings.METRICS['token_validation_failed'] += 1
            return Response({'valid': False, 'error': 'Missing token'}, status=400)

        try:
            payload = _decode_token(token)
            settings.METRICS['token_validation_success'] += 1
            return Response({'valid': True, 'claims': payload})
        except jwt.InvalidTokenError as exc:
            settings.METRICS['token_validation_failed'] += 1
            return Response({'valid': False, 'error': str(exc)}, status=401)

from django.test import TestCase, Client
from django.contrib.auth.hashers import make_password
from .models import AuthUser
import jwt
from django.conf import settings

class JWTAuthTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = '/auth/register/'
        self.login_url = '/auth/login/'
        self.validate_url = '/auth/validate/'
        self.user_data = {
            'email': 'testuser@example.com',
            'password': 'testpassword123',
            'role': 'customer'
        }

    def test_register_and_get_jwt(self):
        response = self.client.post(self.register_url, self.user_data, content_type='application/json')
        self.assertEqual(response.status_code, 201)
        self.assertIn('access', response.data)
        self.assertIn('user', response.data)
        self.assertEqual(response.data['user']['email'], self.user_data['email'])
        
        # Verify the token
        token = response.data['access']
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET, 
            algorithms=[settings.JWT_ALGORITHM], 
            audience=settings.JWT_AUDIENCE, 
            issuer=settings.JWT_ISSUER
        )
        self.assertEqual(payload['email'], self.user_data['email'])
        self.assertEqual(payload['role'], self.user_data['role'])

    def test_login_and_get_jwt(self):
        # First register
        AuthUser.objects.create(
            email=self.user_data['email'],
            password=make_password(self.user_data['password']),
            role=self.user_data['role']
        )
        
        # Then login
        login_data = {
            'email': self.user_data['email'],
            'password': self.user_data['password']
        }
        response = self.client.post(self.login_url, login_data, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('access', response.data)
        
        token = response.data['access']
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET, 
            algorithms=[settings.JWT_ALGORITHM], 
            audience=settings.JWT_AUDIENCE, 
            issuer=settings.JWT_ISSUER
        )
        self.assertEqual(payload['email'], self.user_data['email'])

    def test_validate_token(self):
        # Register to get a token
        response = self.client.post(self.register_url, self.user_data, content_type='application/json')
        token = response.data['access']
        
        # Validate the token
        validate_response = self.client.post(
            self.validate_url, 
            {'token': token}, 
            content_type='application/json'
        )
        self.assertEqual(validate_response.status_code, 200)
        self.assertTrue(validate_response.data['valid'])
        self.assertEqual(validate_response.data['claims']['email'], self.user_data['email'])

    def test_validate_invalid_token(self):
        invalid_token = "this.is.an.invalid.token"
        response = self.client.post(
            self.validate_url, 
            {'token': invalid_token}, 
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 401)
        self.assertFalse(response.data['valid'])
        self.assertIn('error', response.data)

    def test_validate_expired_token(self):
        # We can't easily test expiration without mocking time or setting a very short expiry
        # But we can test the logic if we manually create an expired token
        import time
        now = int(time.time())
        payload = {
            'sub': '1',
            'email': 'expired@example.com',
            'role': 'customer',
            'iat': now - 3600,
            'exp': now - 1800, # expired 30 mins ago
            'iss': settings.JWT_ISSUER,
            'aud': settings.JWT_AUDIENCE,
        }
        expired_token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
        
        response = self.client.post(
            self.validate_url, 
            {'token': expired_token}, 
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 401)
        self.assertFalse(response.data['valid'])
        self.assertEqual(response.data['error'], 'Signature has expired')

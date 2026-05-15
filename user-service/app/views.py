from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.hashers import check_password, make_password
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.authentication import JWTAuthentication

from .models import User, Job
from .serializers import UserSerializer, RegisterSerializer, LoginSerializer, JobSerializer

DEFAULT_JOB_ENUMS = [
    {"title": "Student", "industry": "Education"},
    {"title": "Software Engineer", "industry": "IT"},
    {"title": "Designer", "industry": "Creative"},
    {"title": "Accountant", "industry": "Finance"},
    {"title": "Doctor", "industry": "Healthcare"},
    {"title": "Teacher", "industry": "Education"},
    {"title": "Sales Executive", "industry": "Retail"},
    {"title": "Marketing Specialist", "industry": "Marketing"},
    {"title": "Business Owner", "industry": "Business"},
    {"title": "Freelancer", "industry": "Services"},
]

def ensure_default_jobs():
    for job in DEFAULT_JOB_ENUMS:
        Job.objects.get_or_create(
            title=job["title"],
            defaults={"industry": job["industry"]}
        )

class RegisterView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        email = serializer.validated_data['email'].strip().lower()
        password = serializer.validated_data['password']
        username = serializer.validated_data.get('username', email)
        role = serializer.validated_data.get('role', 'customer')
        phone = serializer.validated_data.get('phone', '')
        staff_role = serializer.validated_data.get('staff_role', None)

        if User.objects.filter(email=email).exists():
            return Response({'error': 'Email already exists'}, status=status.HTTP_409_CONFLICT)

        user = User.objects.create(
            email=email,
            username=username,
            password=make_password(password),
            role=role,
            phone=phone,
            staff_role=staff_role,
            is_active=True,
        )

        refresh = RefreshToken.for_user(user)
        # Add custom claims
        refresh['email'] = user.email
        refresh['role'] = user.role

        return Response(
            {
                'user': UserSerializer(user).data,
                'access': str(refresh.access_token),
                'refresh': str(refresh),
                'token_type': 'Bearer',
            },
            status=status.HTTP_201_CREATED,
        )

class LoginView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        email = serializer.validated_data['email'].strip().lower()
        password = serializer.validated_data['password']

        try:
            user = User.objects.get(email=email, is_active=True)
        except User.DoesNotExist:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

        if not check_password(password, user.password):
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

        refresh = RefreshToken.for_user(user)
        refresh['email'] = user.email
        refresh['role'] = user.role

        return Response(
            {
                'user': UserSerializer(user).data,
                'access': str(refresh.access_token),
                'refresh': str(refresh),
                'token_type': 'Bearer',
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
            return Response({'valid': False, 'error': 'Missing token'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            jwt_auth = JWTAuthentication()
            validated_token = jwt_auth.get_validated_token(token)
            user = jwt_auth.get_user(validated_token)
            
            payload = {
                'sub': str(user.id),
                'email': user.email,
                'role': user.role,
                'exp': validated_token['exp'],
            }
            return Response({'valid': True, 'claims': payload})
        except (InvalidToken, TokenError) as exc:
            return Response({'valid': False, 'error': str(exc)}, status=status.HTTP_401_UNAUTHORIZED)

class UserListCreate(APIView):
    def get(self, request):
        role_filter = request.query_params.get('role', None)
        if role_filter:
            users = User.objects.filter(role=role_filter).order_by('-id')
        else:
            users = User.objects.all().order_by('-id')
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserDetail(APIView):
    def get(self, request, pk):
        try:
            user = User.objects.get(pk=pk)
            serializer = UserSerializer(user)
            return Response(serializer.data)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    def patch(self, request, pk):
        try:
            user = User.objects.get(pk=pk)
            job_id = request.data.get("job_id")
            if job_id is not None:
                if str(job_id).strip() == "":
                    user.job = None
                else:
                    try:
                        job = Job.objects.get(pk=job_id)
                        user.job = job
                    except Job.DoesNotExist:
                        pass
            
            serializer = UserSerializer(user, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        try:
            user = User.objects.get(pk=pk)
            user.delete()
            return Response({"message": "Deleted"}, status=status.HTTP_204_NO_CONTENT)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

class JobList(APIView):
    def get(self, request):
        ensure_default_jobs()
        jobs = Job.objects.all()
        serializer = JobSerializer(jobs, many=True)
        return Response(serializer.data)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.hashers import check_password
from .models import Customer
from .serializers import CustomerSerializer
from .publisher import publish_customer_created
import requests

# CART_SERVICE_URL = "http://cart-service:8000"

from .models import Customer, Job, Address
from .serializers import CustomerSerializer, JobSerializer, AddressSerializer


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


class CustomerListCreate(APIView):
    def get(self, request):
        customers = Customer.objects.all().order_by('-id')
        serializer = CustomerSerializer(customers, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = CustomerSerializer(data=request.data)
        if serializer.is_valid():
            customer = serializer.save()
            return Response(CustomerSerializer(customer).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CustomerDetail(APIView):
    def get(self, request, pk):
        try:
            customer = Customer.objects.get(pk=pk)
            serializer = CustomerSerializer(customer)
            return Response(serializer.data)
        except Customer.DoesNotExist:
            return Response({"error": "Không tìm thấy người dùng"}, status=status.HTTP_404_NOT_FOUND)

    def patch(self, request, pk):
        try:
            customer = Customer.objects.get(pk=pk)
            # Handle job_id separately if provided
            job_id = request.data.get("job_id")
            if job_id is not None:
                if str(job_id).strip() == "":
                    customer.job = None
                else:
                    try:
                        job = Job.objects.get(pk=job_id)
                        customer.job = job
                    except Job.DoesNotExist:
                        pass
            
            serializer = CustomerSerializer(customer, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Customer.DoesNotExist:
            return Response({"error": "Không tìm thấy người dùng"}, status=status.HTTP_404_NOT_FOUND)

class JobList(APIView):
    def get(self, request):
        ensure_default_jobs()
        jobs = Job.objects.all()
        serializer = JobSerializer(jobs, many=True)
        return Response(serializer.data)


from rest_framework_simplejwt.tokens import RefreshToken

class CustomerLogin(APIView):
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")
        
        try:
            customer = Customer.objects.get(email=email)
            if check_password(password, customer.password):
                refresh = RefreshToken.for_user(customer)
                return Response({
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                    "id": customer.id,
                    "name": customer.name,
                    "email": customer.email,
                })
            else:
                return Response({"error": "Mật khẩu không chính xác."}, status=status.HTTP_401_UNAUTHORIZED)
        except Customer.DoesNotExist:
            return Response({"error": "Email không tồn tại."}, status=status.HTTP_404_NOT_FOUND)

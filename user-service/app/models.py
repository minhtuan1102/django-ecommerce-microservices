from django.contrib.auth.models import AbstractUser
from django.db import models

class Job(models.Model):
    title = models.CharField(max_length=255)
    industry = models.CharField(max_length=255, blank=True, null=True)
    company = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.title

class User(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('staff', 'Staff'),
        ('customer', 'Customer'),
        ('service', 'Service'),
    )
    STAFF_ROLE_CHOICES = (
        ('sales', 'Nhân viên bán hàng'),
        ('warehouse', 'Nhân viên kho'),
        ('support', 'Nhân viên hỗ trợ'),
    )

    email = models.EmailField(unique=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='customer')
    phone = models.CharField(max_length=20, blank=True)
    
    # Staff specific fields
    staff_role = models.CharField(max_length=20, choices=STAFF_ROLE_CHOICES, blank=True, null=True)
    
    # Customer specific fields
    job = models.ForeignKey(Job, on_delete=models.SET_NULL, null=True, blank=True, related_name='users')

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'role']

    def __str__(self):
        return f"{self.email} ({self.role})"

class Address(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='addresses')
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    province = models.CharField(max_length=100)
    country = models.CharField(max_length=100, default='Vietnam')
    is_default = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.street}, {self.city}, {self.province}"

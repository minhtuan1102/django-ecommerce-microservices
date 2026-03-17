from django.db import models

class Job(models.Model):
    title = models.CharField(max_length=255) # Ví dụ: Software Engineer, Doctor, Student
    industry = models.CharField(max_length=255, blank=True, null=True) # Ví dụ: IT, Healthcare, Education
    company = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.title

class Customer(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255, null=True, blank=True)
    job = models.ForeignKey(Job, on_delete=models.SET_NULL, null=True, blank=True, related_name='customers')
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

class Address(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='addresses')
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    province = models.CharField(max_length=100)
    country = models.CharField(max_length=100, default='Vietnam')
    is_default = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.street}, {self.city}, {self.province}"

from django.db import models


class Manager(models.Model):
    DEPARTMENT_CHOICES = (
        ('general', 'Quản lý chung'),
        ('sales', 'Quản lý kinh doanh'),
        ('warehouse', 'Quản lý kho'),
    )
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)
    department = models.CharField(max_length=20, choices=DEPARTMENT_CHOICES, default='general')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.get_department_display()})"

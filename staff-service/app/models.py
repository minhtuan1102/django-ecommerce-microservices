from django.db import models


class Staff(models.Model):
    ROLE_CHOICES = (
        ('sales', 'Nhân viên bán hàng'),
        ('warehouse', 'Nhân viên kho'),
        ('support', 'Nhân viên hỗ trợ'),
    )
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='sales')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.get_role_display()})"

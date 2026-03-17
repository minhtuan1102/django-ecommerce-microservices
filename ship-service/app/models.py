from django.db import models
import uuid


class Shipment(models.Model):
    STATUS_CHOICES = [
        ('preparing', 'Preparing'),
        ('shipped', 'Shipped'),
        ('in_transit', 'In Transit'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    CARRIER_CHOICES = [
        ('ghn', 'Giao Hang Nhanh'),
        ('ghtk', 'Giao Hang Tiet Kiem'),
        ('viettel_post', 'Viettel Post'),
        ('jt_express', 'J&T Express'),
    ]

    order_id = models.IntegerField(unique=True)
    tracking_code = models.CharField(max_length=50, unique=True, blank=True)
    carrier = models.CharField(max_length=30, choices=CARRIER_CHOICES, default='ghn')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='preparing')
    shipping_address = models.TextField()
    estimated_days = models.IntegerField(default=3)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.tracking_code:
            self.tracking_code = f"SHIP-{uuid.uuid4().hex[:10].upper()}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Shipment#{self.pk} Order#{self.order_id} - {self.status}"

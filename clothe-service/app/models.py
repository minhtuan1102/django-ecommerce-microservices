from django.db import models

class Clothe(models.Model):
    name = models.CharField(max_length=255)
    material = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.IntegerField()

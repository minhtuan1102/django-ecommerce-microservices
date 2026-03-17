from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class Review(models.Model):
    customer_id = models.IntegerField()
    book_id = models.IntegerField()
    rating = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('customer_id', 'book_id')
        ordering = ['-created_at']

    def __str__(self):
        return f"Review by Customer#{self.customer_id} for Book#{self.book_id} - {self.rating}/5"

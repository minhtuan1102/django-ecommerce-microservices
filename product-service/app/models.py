from django.db import models


class Category(models.Model):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    parent = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        related_name='children',
        on_delete=models.CASCADE,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "categories"

    def __str__(self):
        return self.name


class ProductCatalog(models.Model):
    """Catalog products managed inside product-service."""
    sku = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    item_type = models.CharField(max_length=50)
    category = models.ForeignKey(Category, related_name='products', on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    stock = models.IntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ('name',)

    def __str__(self):
        return f"{self.sku} - {self.name}"


class BookCatalog(models.Model):
    """Maps book_id from book-service to categories."""
    book_id = models.IntegerField()
    category = models.ForeignKey(Category, related_name='books', on_delete=models.CASCADE)

    class Meta:
        unique_together = ('book_id', 'category')

    def __str__(self):
        return f"Book#{self.book_id} -> {self.category.name}"

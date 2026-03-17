from django.db import models


class Category(models.Model):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "categories"

    def __str__(self):
        return self.name


class BookCatalog(models.Model):
    """Maps book_id from book-service to categories."""
    book_id = models.IntegerField()
    category = models.ForeignKey(Category, related_name='books', on_delete=models.CASCADE)

    class Meta:
        unique_together = ('book_id', 'category')

    def __str__(self):
        return f"Book#{self.book_id} -> {self.category.name}"

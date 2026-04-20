from django.contrib import admin
from .models import Category, BookCatalog, ProductCatalog
admin.site.register(Category)
admin.site.register(BookCatalog)
admin.site.register(ProductCatalog)

from rest_framework import serializers
from .models import Category, BookCatalog, ProductCatalog


class CategorySerializer(serializers.ModelSerializer):
    parent_id = serializers.IntegerField(source='parent.id', read_only=True)

    class Meta:
        model = Category
        fields = ('id', 'name', 'description', 'parent', 'parent_id', 'created_at')


class BookCatalogSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)

    class Meta:
        model = BookCatalog
        fields = '__all__'


class ProductCatalogSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    category_id = serializers.IntegerField(source='category.id', read_only=True)

    class Meta:
        model = ProductCatalog
        fields = (
            'id',
            'sku',
            'name',
            'item_type',
            'category',
            'category_id',
            'category_name',
            'price',
            'stock',
            'metadata',
            'is_active',
            'created_at',
            'updated_at',
        )

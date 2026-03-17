from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Category, BookCatalog
from .serializers import CategorySerializer, BookCatalogSerializer


class CategoryListCreate(APIView):
    def get(self, request):
        categories = Category.objects.all()
        return Response(CategorySerializer(categories, many=True).data)

    def post(self, request):
        serializer = CategorySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class CategoryDetail(APIView):
    def get(self, request, pk):
        try:
            cat = Category.objects.get(pk=pk)
            data = CategorySerializer(cat).data
            # Include book_ids in this category
            book_ids = list(BookCatalog.objects.filter(category=cat).values_list('book_id', flat=True))
            data['book_ids'] = book_ids
            return Response(data)
        except Category.DoesNotExist:
            return Response({"error": "Category not found"}, status=404)


class BookCatalogListCreate(APIView):
    def get(self, request):
        book_id = request.query_params.get('book_id')
        if book_id:
            items = BookCatalog.objects.filter(book_id=book_id).select_related('category')
        else:
            items = BookCatalog.objects.all().select_related('category')
        return Response(BookCatalogSerializer(items, many=True).data)

    def post(self, request):
        serializer = BookCatalogSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

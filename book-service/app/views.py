from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Book
from .serializers import BookSerializer

class BookListCreate(APIView):
    def get(self, request):
        books = Book.objects.all()
        serializer = BookSerializer(books, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = BookSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors)


class BookDetail(APIView):
    def get(self, request, pk):
        try:
            book = Book.objects.get(pk=pk)
            serializer = BookSerializer(book)
            return Response(serializer.data)
        except Book.DoesNotExist:
            return Response({"error": "Book not found"}, status=404)

    def patch(self, request, pk):
        try:
            book = Book.objects.get(pk=pk)
            serializer = BookSerializer(book, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=400)
        except Book.DoesNotExist:
            return Response({"error": "Book not found"}, status=404)

class BookReduceStock(APIView):
    def post(self, request, pk):
        try:
            book = Book.objects.get(pk=pk)
            quantity = int(request.data.get("quantity", 0))
            if book.stock < quantity:
                return Response({"error": f"Không đủ hàng cho '{book.title}' (Cần: {quantity}, Hiện có: {book.stock})"}, status=400)
            
            book.stock -= quantity
            book.save()
            return Response({"success": True, "new_stock": book.stock})
        except Book.DoesNotExist:
            return Response({"error": "Sách không tồn tại"}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)

class BookRestoreStock(APIView):
    def post(self, request, pk):
        try:
            book = Book.objects.get(pk=pk)
            quantity = int(request.data.get("quantity", 0))
            book.stock += quantity
            book.save()
            return Response({"success": True, "new_stock": book.stock})
        except Book.DoesNotExist:
            return Response({"error": "Sách không tồn tại"}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)

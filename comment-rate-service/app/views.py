from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Avg
from .models import Review
from .serializers import ReviewSerializer


class ReviewListCreate(APIView):
    def get(self, request):
        reviews = Review.objects.all()
        return Response(ReviewSerializer(reviews, many=True).data)

    def post(self, request):
        serializer = ReviewSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class ReviewsByBook(APIView):
    def get(self, request, book_id):
        reviews = Review.objects.filter(book_id=book_id)
        avg_rating = reviews.aggregate(avg=Avg('rating'))['avg']
        return Response({
            'book_id': book_id,
            'average_rating': round(avg_rating, 1) if avg_rating else 0,
            'total_reviews': reviews.count(),
            'reviews': ReviewSerializer(reviews, many=True).data,
        })


class ReviewDetail(APIView):
    def delete(self, request, pk):
        try:
            review = Review.objects.get(pk=pk)
            review.delete()
            return Response(status=204)
        except Review.DoesNotExist:
            return Response({"error": "Review not found"}, status=404)

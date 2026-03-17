from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Clothe
from .serializers import ClotheSerializer

class ClotheListCreate(APIView):
    def get(self, request):
        clothes = Clothe.objects.all()
        serializer = ClotheSerializer(clothes, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = ClotheSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors)


class ClotheDetail(APIView):
    def get(self, request, pk):
        try:
            clothe = Clothe.objects.get(pk=pk)
            serializer = ClotheSerializer(clothe)
            return Response(serializer.data)
        except Clothe.DoesNotExist:
            return Response({'error': 'Clothe not found'}, status=404)

    def patch(self, request, pk):
        try:
            clothe = Clothe.objects.get(pk=pk)
            serializer = ClotheSerializer(clothe, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=400)
        except Clothe.DoesNotExist:
            return Response({'error': 'Clothe not found'}, status=404)

class ClotheReduceStock(APIView):
    def post(self, request, pk):
        try:
            clothe = Clothe.objects.get(pk=pk)
            quantity = int(request.data.get('quantity', 0))
            if clothe.stock < quantity:
                return Response({'error': f'Không đủ hàng cho {clothe.name}'}, status=400)

            clothe.stock -= quantity
            clothe.save()
            return Response({'success': True, 'new_stock': clothe.stock})
        except Clothe.DoesNotExist:
            return Response({'error': 'Quần áo không tồn tại'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class ClotheRestoreStock(APIView):
    def post(self, request, pk):
        try:
            clothe = Clothe.objects.get(pk=pk)
            quantity = int(request.data.get('quantity', 0))
            clothe.stock += quantity
            clothe.save()
            return Response({'success': True, 'new_stock': clothe.stock})
        except Clothe.DoesNotExist:
            return Response({'error': 'Quần áo không tồn tại'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Staff
from .serializers import StaffSerializer


class StaffListCreate(APIView):
    def get(self, request):
        staff = Staff.objects.all()
        serializer = StaffSerializer(staff, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = StaffSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class StaffDetail(APIView):
    def get(self, request, pk):
        try:
            staff = Staff.objects.get(pk=pk)
            return Response(StaffSerializer(staff).data)
        except Staff.DoesNotExist:
            return Response({"error": "Staff not found"}, status=404)

    def patch(self, request, pk):
        try:
            staff = Staff.objects.get(pk=pk)
            serializer = StaffSerializer(staff, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=400)
        except Staff.DoesNotExist:
            return Response({"error": "Staff not found"}, status=404)

    def delete(self, request, pk):
        try:
            staff = Staff.objects.get(pk=pk)
            staff.delete()
            return Response({"message": "Deleted"}, status=204)
        except Staff.DoesNotExist:
            return Response({"error": "Staff not found"}, status=404)

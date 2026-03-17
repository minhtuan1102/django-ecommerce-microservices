from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Manager
from .serializers import ManagerSerializer


class ManagerListCreate(APIView):
    def get(self, request):
        managers = Manager.objects.all()
        return Response(ManagerSerializer(managers, many=True).data)

    def post(self, request):
        serializer = ManagerSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class ManagerDetail(APIView):
    def get(self, request, pk):
        try:
            return Response(ManagerSerializer(Manager.objects.get(pk=pk)).data)
        except Manager.DoesNotExist:
            return Response({"error": "Manager not found"}, status=404)

    def patch(self, request, pk):
        try:
            mgr = Manager.objects.get(pk=pk)
            serializer = ManagerSerializer(mgr, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=400)
        except Manager.DoesNotExist:
            return Response({"error": "Manager not found"}, status=404)

    def delete(self, request, pk):
        try:
            Manager.objects.get(pk=pk).delete()
            return Response({"message": "Deleted"}, status=204)
        except Manager.DoesNotExist:
            return Response({"error": "Manager not found"}, status=404)

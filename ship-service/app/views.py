from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Shipment
from .serializers import ShipmentSerializer


class ShipmentListCreate(APIView):
    def get(self, request):
        shipments = Shipment.objects.all().order_by('-created_at')
        return Response(ShipmentSerializer(shipments, many=True).data)

    def post(self, request):
        data = request.data.copy()
        address = data.get("shipping_address", "").lower()
        
        # Real logic: calculate shipping rates and choose carrier based on address
        # Mocking dynamic pricing for production feel
        if "hà nội" in address or "hồ chí minh" in address:
            data['carrier'] = "ghn"  # Prefer GHN for major cities
            data['estimated_days'] = 2
        else:
            data['carrier'] = "ghtk" # Prefer GHTK for provincial delivery
            data['estimated_days'] = 4
            
        import random
        import string
        data['tracking_code'] = f"{data['carrier'].upper()}-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        
        serializer = ShipmentSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class ShipmentDetail(APIView):
    def get(self, request, pk):
        try:
            shipment = Shipment.objects.get(pk=pk)
            return Response(ShipmentSerializer(shipment).data)
        except Shipment.DoesNotExist:
            return Response({"error": "Shipment not found"}, status=404)

    def patch(self, request, pk):
        try:
            shipment = Shipment.objects.get(pk=pk)
            serializer = ShipmentSerializer(shipment, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=400)
        except Shipment.DoesNotExist:
            return Response({"error": "Shipment not found"}, status=404)


class ShipmentByOrder(APIView):
    def get(self, request, order_id):
        try:
            shipment = Shipment.objects.get(order_id=order_id)
            return Response(ShipmentSerializer(shipment).data)
        except Shipment.DoesNotExist:
            return Response({"error": "Shipment not found for this order"}, status=404)

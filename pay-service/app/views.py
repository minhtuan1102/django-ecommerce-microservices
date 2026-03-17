from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Payment
from .serializers import PaymentSerializer


import uuid
import requests
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Payment
from .serializers import PaymentSerializer

ORDER_SERVICE_URL = "http://order-service:8000"

class PaymentListCreate(APIView):
    def get(self, request):
        payments = Payment.objects.all().order_by('-created_at')
        return Response(PaymentSerializer(payments, many=True).data)

    def post(self, request):
        data = request.data.copy()
        # Generate a real transaction ID
        data['transaction_id'] = f"TRANS_{uuid.uuid4().hex[:12].upper()}"
        
        serializer = PaymentSerializer(data=data)
        if serializer.is_valid():
            payment = serializer.save()
            # If COD, we still mark completed but for online methods, it stays pending
            if payment.method == 'cod':
                payment.status = 'completed'
                payment.save()
            return Response(PaymentSerializer(payment).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PaymentConfirm(APIView):
    """
    Simulation of a Payment Gateway Webhook (Callback).
    This endpoint is called by the payment provider (Stripe/PayPal/MoMo) 
    after successful user interaction.
    """
    def post(self, request):
        order_id = request.data.get("order_id")
        transaction_id = request.data.get("transaction_id")
        secure_token = request.data.get("secure_token") # Mocking signature verification
        
        if secure_token != "SECRET_PAYMENT_TOKEN": # Simulated security check
             return Response({"error": "Invalid signature"}, status=403)

        try:
            payment = Payment.objects.get(order_id=order_id, transaction_id=transaction_id)
            payment.status = 'completed'
            payment.save()
            
            # NOTIFY Order Service that payment is SUCCESSFUL
            try:
                requests.patch(f"{ORDER_SERVICE_URL}/orders/{order_id}/", 
                               json={"status": "paid", "payment_status": "paid"}, timeout=3)
            except Exception:
                pass
                
            return Response({"status": "Payment confirmed and order updated"})
        except Payment.DoesNotExist:
            return Response({"error": "Payment record not found"}, status=404)

class PaymentDetail(APIView):
    # ... keep existing methods
    def get(self, request, pk):
        try:
            payment = Payment.objects.get(pk=pk)
            return Response(PaymentSerializer(payment).data)
        except Payment.DoesNotExist:
            return Response({"error": "Payment not found"}, status=404)

    def patch(self, request, pk):
        try:
            payment = Payment.objects.get(pk=pk)
            serializer = PaymentSerializer(payment, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=400)
        except Payment.DoesNotExist:
            return Response({"error": "Payment not found"}, status=404)


class PaymentByOrder(APIView):
    def get(self, request, order_id):
        payments = Payment.objects.filter(order_id=order_id).order_by('-created_at')
        return Response(PaymentSerializer(payments, many=True).data)

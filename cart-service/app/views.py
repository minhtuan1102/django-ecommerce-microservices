from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Cart, CartItem
from .serializers import CartSerializer, CartItemSerializer
import requests

PRODUCT_SERVICE_URL = "http://product-service:8000"

class CartCreate(APIView):
    def post(self, request):
        serializer = CartSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors)

class AddCartItem(APIView):
    def post(self, request):
        customer_id = request.data.get("customer_id")
        product_id = request.data.get("product_id")
        quantity = request.data.get("quantity", 1)

        if not customer_id or not product_id:
            return Response({"error": "customer_id and product_id are required"}, status=status.HTTP_400_BAD_REQUEST)

        # 1. Get or create cart for customer
        cart, _ = Cart.objects.get_or_create(customer_id=customer_id)

        # 2. Check stock from product-service
        try:
            r = requests.get(f"{PRODUCT_SERVICE_URL}/products/products/{product_id}/", timeout=3)
            if r.status_code == 200:
                item_data = r.json()
                if int(item_data.get("stock", 0)) < int(quantity):
                    return Response({"error": "Insufficient stock"}, status=status.HTTP_400_BAD_REQUEST)
        except:
            pass # Degraded mode

        # 3. Add or update item
        item, created = CartItem.objects.get_or_create(
            cart=cart, 
            book_id=product_id,
            defaults={"quantity": quantity}
        )
        if not created:
            item.quantity += int(quantity)
            item.save()

        return Response(CartItemSerializer(item).data, status=status.HTTP_201_CREATED)


class DeleteCartItem(APIView):
    def delete(self, request, cart_id, book_id):
        try:
            CartItem.objects.filter(cart_id=cart_id, book_id=book_id).delete()
            return Response({"message": "Item removed from cart"})
        except Exception as e:
            return Response({"error": str(e)}, status=400)


class ClearCart(APIView):
    def delete(self, request, customer_id):
        try:
            cart = Cart.objects.get(customer_id=customer_id)
            CartItem.objects.filter(cart=cart).delete()
            return Response({"message": "Cart cleared successfully"})
        except Cart.DoesNotExist:
            return Response({"error": "Cart not found"}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class CartView(APIView):
    def get(self, request, customer_id):
        try:
            cart, created = Cart.objects.get_or_create(customer_id=customer_id)
            items = CartItem.objects.filter(cart=cart)
            serializer = CartItemSerializer(items, many=True)
            return Response({"cart_id": cart.id, "items": serializer.data})
        except Exception as e:
            return Response({"error": str(e)}, status=500)


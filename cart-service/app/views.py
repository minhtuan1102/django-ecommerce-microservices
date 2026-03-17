from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Cart, CartItem
from .serializers import CartSerializer, CartItemSerializer
import requests

BOOK_SERVICE_URL = "http://book-service:8000"
CLOTHE_SERVICE_URL = "http://clothe-service:8000"

class CartCreate(APIView):
    def post(self, request):
        serializer = CartSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors)

class AddCartItem(APIView):
    def post(self, request):
        cart_id = request.data.get("cart")
        book_id = request.data.get("book_id")
        quantity = request.data.get("quantity")

        # Validate required fields and numeric constraints.
        try:
            cart_id = int(cart_id)
            book_id = int(book_id)
            quantity = int(quantity)
        except (TypeError, ValueError):
            return Response({"error": "Invalid cart, book_id or quantity"}, status=status.HTTP_400_BAD_REQUEST)

        if quantity <= 0:
            return Response({"error": "Quantity must be greater than 0"}, status=status.HTTP_400_BAD_REQUEST)

        if not Cart.objects.filter(id=cart_id).exists():
            return Response({"error": "Cart not found"}, status=status.HTTP_404_NOT_FOUND)

                # Check if book/clothe exists
        try:
            if book_id > 1000000:
                real_id = book_id - 1000000
                r = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/{real_id}/", timeout=3)
                if r.status_code != 200:
                    return Response({"error": "Clothe not found"}, status=status.HTTP_404_NOT_FOUND)
            else:
                r = requests.get(f"{BOOK_SERVICE_URL}/books/{book_id}/", timeout=3)
                if r.status_code != 200:
                    return Response({"error": "Book not found"}, status=status.HTTP_404_NOT_FOUND)

            item_data = r.json()
            if int(item_data.get("stock", 0) or 0) < quantity:
                return Response({"error": "Insufficient stock"}, status=status.HTTP_400_BAD_REQUEST)

            # Merge duplicate rows into one logical line item.
            existing = CartItem.objects.filter(cart_id=cart_id, book_id=book_id).first()
            if existing:
                merged_quantity = existing.quantity + quantity
                if int(item_data.get("stock", 0) or 0) < merged_quantity:
                    return Response({"error": "Insufficient stock for requested quantity"}, status=status.HTTP_400_BAD_REQUEST)
                existing.quantity = merged_quantity
                existing.save(update_fields=["quantity"])
                return Response(CartItemSerializer(existing).data, status=status.HTTP_200_OK)

        except requests.exceptions.RequestException:
            # If dependency is down, still allow cart operation in degraded mode.
            pass

        serializer = CartItemSerializer(data={
            "cart": cart_id,
            "book_id": book_id,
            "quantity": quantity,
        })
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


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


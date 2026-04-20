import requests
import random
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings


from .inference import predict_next_action

class PredictBehavior(APIView):
    """Predicts the next action of a user based on their behavior sequence."""
    def post(self, request):
        sequence = request.data.get('sequence', [])
        if not sequence:
            return Response({"error": "Sequence is required"}, status=400)
        
        # Example sequence: [0, 1, 2, 0, 1] representing action IDs
        action_idx = predict_next_action(sequence)
        
        if action_idx is not None:
            # You can map back action_idx to action name if you have the label encoder
            actions = ["view", "click", "add_to_cart", "purchase", "wishlist", "remove_from_cart", "share", "review"]
            action_name = actions[action_idx] if action_idx < len(actions) else "unknown"
            
            return Response({
                "next_action_index": action_idx,
                "next_action_name": action_name
            })
        else:
            return Response({"error": "Prediction failed or model not loaded"}, status=500)


class RecommendForCustomer(APIView):
    def get(self, request, customer_id):
        try:
            # 1. Fetch deep customer profile from Customer Service
            customer_profile = {}
            try:
                # Assuming CUSTOMER_SERVICE_URL is defined in settings
                resp = requests.get(f"{settings.CUSTOMER_SERVICE_URL}/customers/{customer_id}/", timeout=5)
                if resp.status_code == 200:
                    customer_profile = resp.json()
            except Exception: pass

            job_title = customer_profile.get("job_info", {}).get("title", "").lower()
            industry = customer_profile.get("job_info", {}).get("industry", "").lower()

            # 2. Get history
            ordered_book_ids = set()
            try:
                resp = requests.get(f"{settings.ORDER_SERVICE_URL}/orders/customer/{customer_id}/", timeout=5)
                if resp.status_code == 200:
                    for order in resp.json():
                        for item in order.get('items', []):
                            ordered_book_ids.add(item.get('book_id'))
            except Exception: pass

            # 3. Get all books
            all_books = []
            try:
                resp = requests.get(f"{settings.BOOK_SERVICE_URL}/books/", timeout=5)
                if resp.status_code == 200: all_books = resp.json()
            except Exception: return Response({"recommendations": [], "strategy": "error"})

            # 4. Filter and SCORE based on PERSONA
            scored = []
            for book in all_books:
                if book.get('id') in ordered_book_ids: continue
                
                score = 0
                title = book.get("title", "").lower()
                author = book.get("author", "").lower()

                # PERSONA LOGIC (Complex AI Simulation)
                if "engineer" in job_title or "it" in industry:
                    if any(x in title for x in ["clean code", "java", "python", "system", "data", "algorithm"]):
                        score += 50 # High priority for Tech professionals
                
                if "doctor" in job_title or "medical" in industry:
                    if any(x in title for x in ["health", "medical", "anatomy", "life"]):
                        score += 50

                if "student" in job_title:
                    if any(x in title for x in ["study", "guide", "learn", "science"]):
                        score += 30

                # Reputation scoring
                try:
                    r_rev = requests.get(f"{settings.COMMENT_RATE_SERVICE_URL}/reviews/book/{book['id']}/", timeout=2)
                    if r_rev.status_code == 200:
                        score += r_rev.json().get('average_rating', 0) * 5
                except Exception: pass

                scored.append((book, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            recommendations = [item[0] for item in scored[:5]]

            return Response({
                "customer_id": customer_id,
                "persona": job_title or "generic",
                "recommendations": recommendations,
            })
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class PopularBooks(APIView):
    """Returns most popular books based on ratings."""
    def get(self, request):
        try:
            all_books = []
            resp = requests.get(f"{settings.BOOK_SERVICE_URL}/books/", timeout=5)
            if resp.status_code == 200:
                all_books = resp.json()

            scored = []
            for book in all_books:
                try:
                    resp = requests.get(
                        f"{settings.COMMENT_RATE_SERVICE_URL}/reviews/book/{book['id']}/",
                        timeout=3
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        scored.append({
                            **book,
                            'average_rating': data.get('average_rating', 0),
                            'total_reviews': data.get('total_reviews', 0),
                        })
                    else:
                        scored.append({**book, 'average_rating': 0, 'total_reviews': 0})
                except requests.RequestException:
                    scored.append({**book, 'average_rating': 0, 'total_reviews': 0})

            scored.sort(key=lambda x: (x['average_rating'], x['total_reviews']), reverse=True)
            return Response(scored[:10])

        except Exception:
            return Response([], status=500)

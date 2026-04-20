from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import user_passes_test
from django.http import JsonResponse
import requests
import os
from math import ceil
from urllib.parse import quote

PRODUCT_SERVICE_URL = os.environ.get("PRODUCT_SERVICE_URL", "http://product-service:8000")
BOOK_SERVICE_URL = PRODUCT_SERVICE_URL
CART_SERVICE_URL = "http://cart-service:8000"
CUSTOMER_SERVICE_URL = "http://customer-service:8000"
ORDER_SERVICE_URL = "http://order-service:8000"
STAFF_SERVICE_URL = "http://staff-service:8000"
MANAGER_SERVICE_URL = "http://manager-service:8000"
CLOTHE_SERVICE_URL = PRODUCT_SERVICE_URL
PRODUCT_CATEGORY_SERVICE_URL = PRODUCT_SERVICE_URL
PAY_SERVICE_URL = "http://pay-service:8000"
SHIP_SERVICE_URL = "http://ship-service:8000"
COMMENT_RATE_SERVICE_URL = "http://comment-rate-service:8000"
RECOMMENDER_SERVICE_URL = "http://recommender-ai-service:8000"
AUTH_SERVICE_URL = os.environ.get("AUTH_SERVICE_URL", "http://auth-service:8000")
LAPTOP_SERVICE_URL = ""
MOBILE_SERVICE_URL = ""
# AI Services
BEHAVIOR_ANALYSIS_SERVICE_URL = os.environ.get(
    "BEHAVIOR_ANALYSIS_SERVICE_URL", "http://behavior-analysis-service:8000"
)
CHATBOT_SERVICE_URL = os.environ.get(
    "CHATBOT_SERVICE_URL", "http://consulting-chatbot-service:8000"
)
UNIFIED_AI_SERVICE_URL = os.environ.get(
    "UNIFIED_AI_SERVICE_URL", "http://unified-ai-service:8000"
)


# ── HELPERS ──────────────────────────────────────────────────

def is_staff_check(user):
    return user.is_staff


def _get_store_customer(request):
    cid = request.session.get("customer_id")
    if cid:
        return {"id": cid, "name": request.session.get("customer_name", "")}
    return None


def _get_cart_id(customer_id):
    try:
        r = requests.get(f"{CART_SERVICE_URL}/carts/{customer_id}/", timeout=3)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "cart_id" in data:
                return data["cart_id"]

        # Fallback for first-time customers or transient desync: explicitly create cart.
        r_create = requests.post(
            f"{CART_SERVICE_URL}/carts/",
            json={"customer_id": customer_id},
            timeout=3,
        )
        if r_create.status_code in (200, 201):
            created = r_create.json()
            if isinstance(created, dict) and "id" in created:
                return created["id"]

        # Final retry to handle race conditions.
        r_retry = requests.get(f"{CART_SERVICE_URL}/carts/{customer_id}/", timeout=3)
        if r_retry.status_code == 200:
            data = r_retry.json()
            if isinstance(data, dict) and "cart_id" in data:
                return data["cart_id"]
    except Exception:
        pass
    return None


def _safe_get_json(url, timeout=4):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            payload = r.json()
            if isinstance(payload, list):
                return payload
    except Exception:
        pass
    return []


def _normalize_multi_item_products(items, item_type, source):
    normalized = []
    for raw in items:
        if not isinstance(raw, dict):
            continue

        if item_type == "book":
            item_id = raw.get("id")
            title = raw.get("title", "")
            category = raw.get("category", "books")
            brand_or_author = raw.get("author", "")
            material = ""
        elif item_type == "clothe":
            item_id = raw.get("id")
            title = raw.get("name", "")
            category = raw.get("category", "fashion")
            brand_or_author = raw.get("brand", "")
            material = raw.get("material", "")
        else:
            # Generic mapping for optional item services (laptop/mobile/...)
            item_id = raw.get("id")
            title = raw.get("name") or raw.get("title") or ""
            category = raw.get("category", item_type)
            brand_or_author = raw.get("brand") or raw.get("manufacturer") or ""
            material = raw.get("material", "")

        if not item_id or not title:
            continue

        try:
            price = float(raw.get("price", 0) or 0)
        except (TypeError, ValueError):
            price = 0.0

        try:
            stock = int(raw.get("stock", 0) or 0)
        except (TypeError, ValueError):
            stock = 0

        normalized.append({
            "id": item_id,
            "sku": f"{item_type}-{item_id}",
            "title": title,
            "item_type": item_type,
            "category": str(category),
            "brand_or_author": str(brand_or_author),
            "material": str(material),
            "price": price,
            "stock": stock,
            "in_stock": stock > 0,
            "source_service": source,
            "raw": raw,
        })

    return normalized


def _fetch_ecommerce_products():
    products = []
    catalog_payload = []
    try:
        r = requests.get(f"{PRODUCT_SERVICE_URL}/products/", timeout=5)
        if r.status_code == 200:
            body = r.json()
            if isinstance(body, dict):
                catalog_payload = body.get("products", [])
            elif isinstance(body, list):
                catalog_payload = body
    except Exception:
        catalog_payload = []

    for item in catalog_payload:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        title = item.get("name") or item.get("title")
        raw_item_type = str(item.get("item_type") or "general").lower()
        item_type = "clothe" if raw_item_type == "fashion" else raw_item_type
        category = item.get("category_name") or item.get("category") or item_type
        sku = item.get("sku") or f"{item_type}-{item_id}"
        metadata_obj = item.get("metadata") or {}

        try:
            price = float(item.get("price", 0) or 0)
        except (TypeError, ValueError):
            price = 0.0

        try:
            stock = int(item.get("stock", 0) or 0)
        except (TypeError, ValueError):
            stock = 0

        if not item_id or not title:
            continue

        products.append(
            {
                "id": item_id,
                "sku": sku,
                "title": str(title),
                "item_type": item_type,
                "category": str(category),
                "brand_or_author": str(metadata_obj.get("brand_or_author", "")),
                "material": str(metadata_obj.get("material", "")),
                "description": str(metadata_obj.get("description", "")),
                "price": price,
                "stock": stock,
                "in_stock": stock > 0,
                "image_url": metadata_obj.get("image_url") or _build_recommendation_image(item_type, item_id, title),
                "detail_url": f"/store/item/{item_type}/{item_id}/",
                "source_service": "product-service",
                "raw": item,
            }
        )

    return products


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fetch_behavior_analysis(customer_id):
    """Fetch behavior analysis snapshot for current customer."""
    customer_id_int = _safe_int(customer_id, None)
    if customer_id_int is None:
        return {}

    try:
        r = requests.get(
            f"{BEHAVIOR_ANALYSIS_SERVICE_URL}/api/behavior/customer/{customer_id_int}/analysis/",
            timeout=6,
        )
        if r.status_code != 200:
            return {}

        payload = r.json()
        if isinstance(payload, dict):
            if payload.get("success") and isinstance(payload.get("data"), dict):
                return payload["data"]
            if isinstance(payload.get("data"), dict):
                return payload["data"]
            if payload.get("segment"):
                return payload
    except Exception:
        pass

    return {}


def _normalize_behavior_categories(behavior_data):
    categories = []
    raw = behavior_data.get("predicted_categories", [])
    if not isinstance(raw, list):
        return categories

    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("category", "")).strip()
            probability = item.get("probability")
        else:
            name = str(item).strip()
            probability = None

        if not name:
            continue

        categories.append(
            {
                "category": name,
                "probability": float(probability) if isinstance(probability, (int, float)) else None,
            }
        )

    return categories


def _build_recommendation_image(item_type, item_id, title):
    """Create a robust inline SVG thumbnail for recommendation cards with fixed encoding."""
    icon_map = {
        "book": "BOOK",
        "clothe": "FASHION",
        "laptop": "LAPTOP",
        "mobile": "MOBILE",
    }
    label = icon_map.get(str(item_type or "").lower(), "PRODUCT")
    safe_title = (str(title or "Product")[:28]).replace("&", "and")
    
    # Improved SVG styling for MVP
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='360' height='220' viewBox='0 0 360 220'>"
        f"<rect width='360' height='220' fill='#f0f4f8'/>"
        f"<rect x='20' y='20' width='320' height='180' rx='15' fill='white' stroke='#d1d9e6' stroke-width='2'/>"
        f"<text x='40' y='60' fill='#4361ee' font-size='18' font-family='sans-serif' font-weight='bold'>{label}</text>"
        f"<text x='40' y='100' fill='#2d3748' font-size='16' font-family='sans-serif'>{safe_title}</text>"
        f"<text x='40' y='140' fill='#718096' font-size='14' font-family='sans-serif'>ID: #{item_id}</text>"
        f"<rect x='40' y='160' width='80' height='4' rx='2' fill='#4361ee' fill-opacity='0.3'/>"
        f"</svg>"
    )
    # Proper percent-encoding
    return f"data:image/svg+xml,{quote(svg)}"


def _fetch_single_item_detail(item_type, item_id):
    """Fetch item detail from unified product-service."""
    try:
        r = requests.get(f"{PRODUCT_SERVICE_URL}/products/{item_id}/", timeout=4)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _build_ai_recommendations(products, behavior_data, limit=8, item_type="all", query=""):
    """Score and return personalized multi-item recommendations."""
    normalized_categories = _normalize_behavior_categories(behavior_data)
    category_keywords = {
        c["category"].strip().lower()
        for c in normalized_categories
        if c.get("category")
    }
    segment = str(behavior_data.get("segment", "Regular") or "Regular")
    query_lower = str(query or "").strip().lower()

    scored = []
    for product in products:
        if item_type != "all" and product.get("item_type") != item_type:
            continue

        title = str(product.get("title", ""))
        category = str(product.get("category", ""))
        search_space = f"{title} {category} {product.get('brand_or_author', '')}".lower()
        if query_lower and query_lower not in search_space:
            continue

        in_stock = bool(product.get("in_stock"))
        price = float(product.get("price", 0) or 0)
        stock = _safe_int(product.get("stock"), 0)
        item_type_value = str(product.get("item_type", "general"))
        if item_type_value == "fashion":
            item_type_value = "clothe"

        score = 0.0
        score += 30.0 if in_stock else -60.0
        score += min(max(stock, 0), 30) * 0.5

        if segment == "VIP":
            if price >= 300000:
                score += 10.0
        elif segment == "New":
            if 50000 <= price <= 700000:
                score += 10.0
        elif segment == "Churned":
            if price <= 300000:
                score += 12.0
        else:
            if 70000 <= price <= 900000:
                score += 8.0

        if category_keywords:
            lowered_title = title.lower()
            lowered_category = category.lower()
            if any(k in lowered_title or k in lowered_category for k in category_keywords):
                score += 25.0

        if item_type_value in ("book", "clothe"):
            score += 4.0

        recommendation = {
            **product,
            "ai_score": round(score, 2),
            "detail_url": f"/store/item/{item_type_value}/{product.get('id')}/",
            "image_url": _build_recommendation_image(item_type_value, product.get("id"), title),
            "add_to_cart_payload": None,
        }

        if item_type_value == "book":
            recommendation["add_to_cart_payload"] = {"book_id": product.get("id")}
        elif item_type_value == "clothe":
            recommendation["add_to_cart_payload"] = {"clothe_id": product.get("id")}

        scored.append(recommendation)

    scored.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
    return scored[: max(1, min(limit, 24))]


def ecommerce_products_api(request):
    """Unified multi-item catalog API for ecommerce and AI ingestion."""
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    q = request.GET.get("q", "").strip().lower()
    item_type = request.GET.get("type", "all").strip().lower()
    stock_filter = request.GET.get("stock", "all").strip().lower()

    products = _fetch_ecommerce_products()
    filtered = []

    for p in products:
        if item_type != "all" and p.get("item_type") != item_type:
            continue

        if stock_filter == "in_stock" and not p.get("in_stock"):
            continue
        if stock_filter == "out_of_stock" and p.get("in_stock"):
            continue

        if q:
            haystack = " ".join([
                str(p.get("title", "")),
                str(p.get("category", "")),
                str(p.get("brand_or_author", "")),
            ]).lower()
            if q not in haystack:
                continue

        filtered.append(p)

    return JsonResponse({
        "total": len(filtered),
        "item_types": sorted({p.get("item_type") for p in filtered if p.get("item_type")}),
        "products": filtered,
    })


# ── ADMIN VIEWS ──────────────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def home(request):
    try:
        books = requests.get(f"{BOOK_SERVICE_URL}/books/", timeout=3).json()
    except Exception:
        books = []
    try:
        customers = requests.get(f"{CUSTOMER_SERVICE_URL}/customers/", timeout=3).json()
    except Exception:
        customers = []
    try:
        orders = requests.get(f"{ORDER_SERVICE_URL}/orders/", timeout=3).json()
    except Exception:
        orders = []
    try:
        staff = requests.get(f"{STAFF_SERVICE_URL}/staff/", timeout=3).json()
    except Exception:
        staff = []
    try:
        managers = requests.get(f"{MANAGER_SERVICE_URL}/managers/", timeout=3).json()
    except Exception:
        managers = []
    try:
        payments = requests.get(f"{PAY_SERVICE_URL}/payments/", timeout=3).json()
    except Exception:
        payments = []
    try:
        shipments = requests.get(f"{SHIP_SERVICE_URL}/shipments/", timeout=3).json()
    except Exception:
        shipments = []
    return render(request, "home.html", {
        "total_books": len(books) if isinstance(books, list) else 0,
        "total_customers": len(customers) if isinstance(customers, list) else 0,
        "total_orders": len(orders) if isinstance(orders, list) else 0,
        "total_staff": len(staff) if isinstance(staff, list) else 0,
        "total_managers": len(managers) if isinstance(managers, list) else 0,
        "total_payments": len(payments) if isinstance(payments, list) else 0,
        "total_shipments": len(shipments) if isinstance(shipments, list) else 0,
    })


@user_passes_test(is_staff_check, login_url='/admin/login/')
def book_list(request):
    error = None
    books = []
    if request.method == "POST":
        data = {
            "title": request.POST.get("title"),
            "author": request.POST.get("author"),
            "price": request.POST.get("price"),
            "stock": request.POST.get("stock"),
        }
        try:
            r = requests.post(f"{BOOK_SERVICE_URL}/books/", json=data, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Thêm sách thành công!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Không kết nối được book-service: {e}")
        return redirect("book_list")
    try:
        r = requests.get(f"{BOOK_SERVICE_URL}/books/", timeout=3)
        books = r.json()
        if not isinstance(books, list):
            books = []
    except Exception as e:
        error = str(e)
    return render(request, "books.html", {"books": books, "error": error})


@user_passes_test(is_staff_check, login_url='/admin/login/')
def customer_list(request):
    error = None
    customers = []
    if request.method == "POST":
        data = {
            "name": request.POST.get("name"),
            "email": request.POST.get("email"),
        }
        try:
            r = requests.post(f"{CUSTOMER_SERVICE_URL}/customers/", json=data, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Thêm khách hàng thành công!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Không kết nối được customer-service: {e}")
        return redirect("customer_list")
    try:
        r = requests.get(f"{CUSTOMER_SERVICE_URL}/customers/", timeout=3)
        customers = r.json()
        if not isinstance(customers, list):
            customers = []
    except Exception as e:
        error = str(e)
    return render(request, "customers.html", {"customers": customers, "error": error})


@user_passes_test(is_staff_check, login_url='/admin/login/')
def view_cart(request, customer_id):
    error = None
    items = []
    cart_id = None
    cart_error = None
    if request.method == "POST":
        data = {
            "cart": request.POST.get("cart_id"),
            "book_id": request.POST.get("book_id"),
            "quantity": request.POST.get("quantity"),
        }
        try:
            r = requests.post(f"{CART_SERVICE_URL}/cart-items/", json=data, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Thêm vào giỏ hàng thành công!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Không kết nối được cart-service: {e}")
        return redirect("view_cart", customer_id=customer_id)
    try:
        r = requests.get(f"{CART_SERVICE_URL}/carts/{customer_id}/", timeout=3)
        data = r.json()
        if isinstance(data, dict) and "cart_id" in data:
            cart_id = data["cart_id"]
            items = data.get("items", [])
        elif isinstance(data, dict) and "error" in data:
            cart_error = data["error"]
    except Exception as e:
        error = str(e)
    try:
        books = requests.get(f"{BOOK_SERVICE_URL}/books/", timeout=3).json()
        if not isinstance(books, list):
            books = []
    except Exception:
        books = []
    return render(request, "cart.html", {
        "items": items, "customer_id": customer_id, "cart_id": cart_id,
        "books": books, "error": error, "cart_error": cart_error,
    })


# ── STOREFRONT VIEWS ─────────────────────────────────────────

def store_home(request):
    products = []
    recommendations = []
    ai_recommendations = []
    behavior_data = {}
    chatbot_health_status = "unknown"
    q = request.GET.get("q", "").strip()
    item_type = request.GET.get("item_type", request.GET.get("author", "")).strip().lower()
    stock = request.GET.get("stock", "all").strip()
    sort = request.GET.get("sort", "featured").strip()
    min_price_raw = request.GET.get("min_price", "").strip()
    max_price_raw = request.GET.get("max_price", "").strip()
    page_raw = request.GET.get("page", "1").strip()

    min_price = None
    max_price = None
    try:
        if min_price_raw:
            min_price = float(min_price_raw)
    except ValueError:
        min_price = None
    try:
        if max_price_raw:
            max_price = float(max_price_raw)
    except ValueError:
        max_price = None

    try:
        page = max(1, int(page_raw))
    except ValueError:
        page = 1

    products = _fetch_ecommerce_products()
    all_item_types = sorted({str(p.get("item_type", "")).strip().lower() for p in products if p.get("item_type")})

    filtered_products = []
    q_lower = q.lower()
    for product in products:
        title = str(product.get("title", ""))
        category = str(product.get("category", ""))
        item_type_value = str(product.get("item_type", "")).lower()
        stock_value = int(product.get("stock", 0) or 0)
        try:
            price_value = float(product.get("price", 0) or 0)
        except (TypeError, ValueError):
            price_value = 0.0

        if q_lower and q_lower not in title.lower() and q_lower not in category.lower() and q_lower not in item_type_value:
            continue
        if item_type and item_type != item_type_value:
            continue
        if stock == "in_stock" and stock_value <= 0:
            continue
        if stock == "out_of_stock" and stock_value > 0:
            continue
        if min_price is not None and price_value < min_price:
            continue
        if max_price is not None and price_value > max_price:
            continue
        filtered_products.append(product)

    if sort == "price_asc":
        filtered_products.sort(key=lambda x: float(x.get("price", 0) or 0))
    elif sort == "price_desc":
        filtered_products.sort(key=lambda x: float(x.get("price", 0) or 0), reverse=True)
    elif sort == "title_asc":
        filtered_products.sort(key=lambda x: str(x.get("title", "")).lower())
    elif sort == "title_desc":
        filtered_products.sort(key=lambda x: str(x.get("title", "")).lower(), reverse=True)
    elif sort == "newest":
        filtered_products.sort(key=lambda x: int(x.get("id", 0) or 0), reverse=True)
    else:
        # Featured: in-stock first, then high stock, then newest.
        filtered_products.sort(
            key=lambda x: (
                int((x.get("stock", 0) or 0) <= 0),
                -int(x.get("stock", 0) or 0),
                -int(x.get("id", 0) or 0),
            )
        )

    page_size = 12
    total_results = len(filtered_products)
    total_pages = max(1, ceil(total_results / page_size))
    if page > total_pages:
        page = total_pages

    start = (page - 1) * page_size
    end = start + page_size
    paginated_products = filtered_products[start:end]

    base_filters = request.GET.copy()
    if "page" in base_filters:
        del base_filters["page"]
    query_without_page = base_filters.urlencode()

    page_numbers = [
        n for n in range(max(1, page - 2), min(total_pages, page + 2) + 1)
    ]

    current_querystring = request.get_full_path()

    customer = _get_store_customer(request)
    if customer:
        try:
            r = requests.get(f"{RECOMMENDER_SERVICE_URL}/recommendations/{customer['id']}/", timeout=5)
            if r.status_code == 200:
                recommendations = r.json().get("recommendations", [])
        except Exception:
            pass

        behavior_data = _fetch_behavior_analysis(customer.get("id"))
        ai_recommendations = _build_ai_recommendations(
            _fetch_ecommerce_products(),
            behavior_data,
            limit=8,
            item_type="all",
            query=q,
        )

        try:
            r_health = requests.get(f"{CHATBOT_SERVICE_URL}/api/chat/health/", timeout=4)
            if r_health.status_code == 200:
                payload = r_health.json()
                chatbot_health_status = str(payload.get("status", "healthy"))
            else:
                chatbot_health_status = "unhealthy"
        except Exception:
            chatbot_health_status = "unreachable"

    return render(request, "store_home.html", {
        "books": paginated_products,
        "customer": customer,
        "recommendations": recommendations,
        "ai_recommendations": ai_recommendations,
        "behavior_data": behavior_data,
        "predicted_categories": _normalize_behavior_categories(behavior_data),
        "chatbot_health_status": chatbot_health_status,
        "authors": all_item_types,
        "total_books": len(products),
        "total_results": total_results,
        "in_stock_count": sum(1 for b in products if int(b.get("stock", 0) or 0) > 0),
        "filters": {
            "q": q,
            "author": item_type,
            "stock": stock,
            "sort": sort,
            "min_price": min_price_raw,
            "max_price": max_price_raw,
        },
        "page": page,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1,
        "next_page": page + 1,
        "page_numbers": page_numbers,
        "query_without_page": query_without_page,
        "current_querystring": current_querystring,
    })


def store_ai_assistant(request):
    """Full AI assistant storefront: behavior insights + chatbot + multi-item recommendations."""
    customer = _get_store_customer(request)
    if not customer:
        messages.error(request, "Vui lòng đăng nhập để dùng trợ lý AI.")
        return redirect("store_login")

    behavior_data = _fetch_behavior_analysis(customer.get("id"))
    predicted_categories = _normalize_behavior_categories(behavior_data)
    products = _fetch_ecommerce_products()
    ai_recommendations = _build_ai_recommendations(
        products,
        behavior_data,
        limit=10,
        item_type="all",
        query="",
    )

    chatbot_health_status = "unknown"
    try:
        r = requests.get(f"{CHATBOT_SERVICE_URL}/api/chat/health/", timeout=4)
        if r.status_code == 200:
            payload = r.json()
            chatbot_health_status = str(payload.get("status", "healthy"))
        else:
            chatbot_health_status = "unhealthy"
    except Exception:
        chatbot_health_status = "unreachable"

    return render(
        request,
        "store_ai_assistant.html",
        {
            "customer": customer,
            "behavior_data": behavior_data,
            "predicted_categories": predicted_categories,
            "chatbot_health_status": chatbot_health_status,
            "ai_recommendations": ai_recommendations,
            "recommendation_count": len(ai_recommendations),
        },
    )


def store_login(request):
    if _get_store_customer(request):
        return redirect("store_home")
    if request.method == "POST":
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")
        try:
            r = requests.post(f"{CUSTOMER_SERVICE_URL}/customers/login/", 
                              json={"email": email, "password": password}, timeout=3)
            if r.status_code == 200:
                try:
                    found = r.json()
                    request.session["customer_id"] = found["id"]
                    request.session["customer_name"] = found["name"]

                    # Central auth-service token issuance.
                    try:
                        r_auth = requests.post(
                            f"{AUTH_SERVICE_URL}/auth/login/",
                            json={"email": email, "password": password},
                            timeout=3,
                        )
                        if r_auth.status_code == 200:
                            request.session["access_token"] = r_auth.json().get("access", "")
                    except Exception:
                        pass

                    messages.success(request, f"Xin chào, {found['name']}!")
                    return redirect("store_home")
                except ValueError:
                    messages.error(request, "Lỗi phản hồi từ hệ thống xác thực (Invalid JSON).")
            else:
                try:
                    error_msg = r.json().get("error", "Email hoặc mật khẩu không đúng.")
                except ValueError:
                    error_msg = f"Lỗi hệ thống ({r.status_code}). Vui lòng thử lại sau."
                messages.error(request, error_msg)
        except Exception as e:
            messages.error(request, f"Lỗi kết nối: {e}")
    return render(request, "store_login.html", {"customer": None})


def store_register(request):
    if _get_store_customer(request):
        return redirect("store_home")
    if request.method == "POST":
        data = {
            "name": request.POST.get("name", "").strip(),
            "email": request.POST.get("email", "").strip(),
            "password": request.POST.get("password", ""),
        }
        try:
            r = requests.post(f"{CUSTOMER_SERVICE_URL}/customers/", json=data, timeout=3)
            if r.status_code in (200, 201):
                customer = r.json()

                # Sync identity to central auth-service.
                try:
                    r_auth = requests.post(
                        f"{AUTH_SERVICE_URL}/auth/register/",
                        json={
                            "email": data["email"],
                            "password": data["password"],
                            "role": "customer",
                        },
                        timeout=3,
                    )
                    if r_auth.status_code in (200, 201):
                        request.session["access_token"] = r_auth.json().get("access", "")
                except Exception:
                    pass

                # Log them in automatically
                request.session["customer_id"] = customer["id"]
                request.session["customer_name"] = customer["name"]
                messages.success(request, f"Đăng ký thành công! Xin chào, {customer['name']}!")
                return redirect("store_home")
            else:
                resp = r.json()
                if "email" in resp:
                    messages.error(request, "Email này đã được đăng ký.")
                else:
                    messages.error(request, f"Lỗi: {resp}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối: {e}")
    return render(request, "store_register.html", {"customer": None})


def store_profile(request):
    customer = _get_store_customer(request)
    if not customer:
        return redirect("store_login")
    
    if request.method == "POST":
        data = {
            "name": request.POST.get("name"),
            "phone": request.POST.get("phone"),
            "job_id": request.POST.get("job_id"),
        }
        try:
            r = requests.patch(f"{CUSTOMER_SERVICE_URL}/customers/{customer['id']}/", json=data, timeout=3)
            if r.status_code == 200:
                messages.success(request, "Cập nhật hồ sơ thành công!")
                request.session["customer_name"] = data["name"] # Sync name in session
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối: {e}")
        return redirect("store_profile")

    # Fetch customer full info and available jobs
    full_info = {}
    jobs = []
    try:
        r_cust = requests.get(f"{CUSTOMER_SERVICE_URL}/customers/{customer['id']}/", timeout=3)
        if r_cust.status_code == 200:
            full_info = r_cust.json()
        
        r_jobs = requests.get(f"{CUSTOMER_SERVICE_URL}/jobs/", timeout=3)
        if r_jobs.status_code == 200:
            jobs = r_jobs.json()
    except Exception as e:
        messages.error(request, f"Lỗi lấy thông tin: {e}")

    return render(request, "store_profile.html", {
        "customer": full_info,
        "jobs": jobs,
    })


def store_logout(request):
    request.session.flush()
    messages.success(request, "Đã đăng xuất thành công.")
    return redirect("store_home")


def store_cart(request):
    customer = _get_store_customer(request)
    if not customer:
        messages.error(request, "Vui lòng đăng nhập để xem giỏ hàng.")
        return redirect("store_login")
    items = []
    cart_id = None
    error = None
    try:
        r = requests.get(f"{CART_SERVICE_URL}/carts/{customer['id']}/", timeout=3)
        data = r.json()
        if isinstance(data, dict) and "cart_id" in data:
            cart_id = data["cart_id"]
            items = data.get("items", [])
        elif isinstance(data, dict) and "error" in data:
            error = data["error"]
    except Exception as e:
        error = str(e)
        
    books_map = {}
    clothes_map = {}
    try:
        books = requests.get(f"{BOOK_SERVICE_URL}/books/", timeout=3).json()
        if isinstance(books, list):
            books_map = {b["id"]: b for b in books}
            
        clothes = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/", timeout=3).json()
        if isinstance(clothes, list):
            clothes_map = {c["id"]: c for c in clothes}
    except Exception:
        pass
        
    total = 0
    enriched = []
    for item in items:
        actual_id = item["book_id"]
        if actual_id > 1000000:
            clothe_id = actual_id - 1000000
            clothe = clothes_map.get(clothe_id, {"name": f"Áo #{clothe_id}", "price": 0})
            subtotal = float(clothe.get("price", 0)) * item["quantity"]
            total += subtotal
            enriched.append({
                **item, 
                "book": {"title": clothe.get("name"), "author": "Thời trang", "price": clothe.get("price")}, 
                "subtotal": subtotal,
                "is_clothe": True
            })
        else:
            book = books_map.get(actual_id, {"title": f"Sách #{actual_id}", "author": "", "price": 0})
            subtotal = float(book.get("price", 0)) * item["quantity"]
            total += subtotal
            enriched.append({**item, "book": book, "subtotal": subtotal, "is_clothe": False})
            
    return render(request, "store_cart.html", {
        "items": enriched, "cart_id": cart_id,
        "total": total, "error": error, "customer": customer,
    })


def store_add_to_cart(request):
    if request.method != "POST":
        return redirect("store_home")

    next_url = request.POST.get("next", "/store/")
    if not next_url.startswith("/store"):
        next_url = "/store/"

    customer = _get_store_customer(request)
    if not customer:
        messages.error(request, "Vui lòng đăng nhập để thêm vào giỏ hàng.")
        return redirect("store_login")

    book_id = request.POST.get("book_id")
    clothe_id = request.POST.get("clothe_id")
    
    try:
        quantity = int(request.POST.get("quantity", 1))
    except ValueError:
        quantity = 1
    quantity = max(1, min(quantity, 99))

    cart_id = _get_cart_id(customer["id"])
    if not cart_id:
        messages.error(request, "Không tìm thấy giỏ hàng.")
        return redirect(next_url)

    if book_id and book_id != "None" and book_id != "":
        try:
            r_book = requests.get(f"{BOOK_SERVICE_URL}/books/{int(book_id)}/", timeout=3)
            if r_book.status_code == 200:
                book = r_book.json()
                if int(book.get("stock", 0) or 0) <= 0:
                    messages.error(request, f"Sách '{book.get('title', '')}' đã hết hàng.")
                    return redirect(next_url)
            else:
                messages.error(request, "Không thể kiểm tra tồn kho hiện tại.")
                return redirect(next_url)
        except Exception as e:
            messages.error(request, f"Lỗi kết nối kiểm tra tồn kho: {e}")
            return redirect(next_url)
            
        try:
            r = requests.post(f"{CART_SERVICE_URL}/cart-items/", json={
                "cart": cart_id,
                "book_id": int(book_id),
                "quantity": quantity,
            }, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Đã thêm Sách vào giỏ hàng!")
            else:
                messages.error(request, f"Lỗi thêm vào giỏ: {r.text}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối giỏ hàng: {e}")
            
    elif clothe_id and clothe_id != "None" and clothe_id != "":
        try:
            r_clothe = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/{int(clothe_id)}/", timeout=3)
            if r_clothe.status_code == 200:
                clothe = r_clothe.json()
                if int(clothe.get("stock", 0) or 0) <= 0:
                    messages.error(request, f"Sản phẩm '{clothe.get('name', '')}' đã hết hàng.")
                    return redirect(next_url)
            else:
                messages.error(request, "Không thể kiểm tra tồn kho hiện tại.")
                return redirect(next_url)
        except Exception as e:
            messages.error(request, f"Lỗi kết nối kiểm tra tồn kho: {e}")
            return redirect(next_url)
            
        try:
            r = requests.post(f"{CART_SERVICE_URL}/cart-items/", json={
                "cart": cart_id,
                "book_id": int(clothe_id) + 1000000,
                "quantity": quantity,
            }, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Đã thêm Quần áo vào giỏ hàng!")
            else:
                messages.error(request, f"Lỗi thêm vào giỏ: {r.text}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối giỏ hàng: {e}")
    else:
        messages.error(request, "Dữ liệu sản phẩm không hợp lệ.")

    return redirect(next_url)

    if book_id:
        try:
            r_book = requests.get(f"{BOOK_SERVICE_URL}/books/{int(book_id)}/", timeout=3)
            if r_book.status_code == 200:
                book = r_book.json()
                if int(book.get("stock", 0) or 0) <= 0:
                    messages.error(request, f"Sách '{book.get('title', '')}' đã hết hàng.")
                    return redirect(next_url)
            else:
                messages.error(request, "Không thể kiểm tra tồn kho hiện tại.")
                return redirect(next_url)
        except Exception as e:
            messages.error(request, f"Lỗi kết nối kiểm tra tồn kho: {e}")
            return redirect(next_url)
            
        try:
            r = requests.post(f"{CART_SERVICE_URL}/cart-items/", json={
                "cart": cart_id,
                "book_id": int(book_id),
                "quantity": quantity,
            }, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Đã thêm Sách vào giỏ hàng!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối: {e}")
            
    elif clothe_id:
        try:
            r_clothe = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/{int(clothe_id)}/", timeout=3)
            if r_clothe.status_code == 200:
                clothe = r_clothe.json()
                if int(clothe.get("stock", 0) or 0) <= 0:
                    messages.error(request, f"Sản phẩm '{clothe.get('name', '')}' đã hết hàng.")
                    return redirect(next_url)
            else:
                messages.error(request, "Không thể kiểm tra tồn kho hiện tại.")
                return redirect(next_url)
        except Exception as e:
            messages.error(request, f"Lỗi kết nối kiểm tra tồn kho: {e}")
            return redirect(next_url)
            
        try:
            r = requests.post(f"{CART_SERVICE_URL}/cart-items/", json={
                "cart": cart_id,
                "book_id": int(clothe_id) + 1000000,
                "quantity": quantity,
            }, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Đã thêm Quần áo vào giỏ hàng!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối: {e}")

    return redirect(next_url)

    # Basic stock guard before writing cart item.
    try:
        r_book = requests.get(f"{BOOK_SERVICE_URL}/books/{int(book_id)}/", timeout=3)
        if r_book.status_code == 200:
            book = r_book.json()
            if int(book.get("stock", 0) or 0) <= 0:
                messages.error(request, f"Sách '{book.get('title', '')}' đã hết hàng.")
                return redirect(next_url)
        else:
            messages.error(request, "Không thể kiểm tra tồn kho hiện tại.")
            return redirect(next_url)
    except Exception as e:
        messages.error(request, f"Lỗi kết nối kiểm tra tồn kho: {e}")
        return redirect(next_url)

    try:
        r = requests.post(f"{CART_SERVICE_URL}/cart-items/", json={
            "cart": cart_id,
            "book_id": int(book_id),
            "quantity": quantity,
        }, timeout=3)
        if r.status_code in (200, 201):
            messages.success(request, "Đã thêm vào giỏ hàng!")
        else:
            messages.error(request, f"Lỗi: {r.text}")
    except Exception as e:
        messages.error(request, f"Lỗi kết nối: {e}")
    return redirect(next_url)


def store_book_detail(request, book_id):
    book = None
    reviews_data = {"reviews": [], "average_rating": 0, "total_reviews": 0}
    try:
        # Fetch single book directly from book-service
        r = requests.get(f"{BOOK_SERVICE_URL}/books/{book_id}/", timeout=3)
        if r.status_code == 200:
            book = r.json()
    except Exception as e:
        print(f"Error fetching book detail: {e}")
    
    try:
        r = requests.get(f"{COMMENT_RATE_SERVICE_URL}/reviews/book/{book_id}/", timeout=3)
        if r.status_code == 200:
            reviews_data = r.json()
    except Exception:
        pass
    
    return render(request, "store_book_detail.html", {
        "book": book,
        "customer": _get_store_customer(request),
        "reviews": reviews_data.get("reviews", []),
        "average_rating": reviews_data.get("average_rating", 0),
        "total_reviews": reviews_data.get("total_reviews", 0),
    })


def store_remove_from_cart(request, book_id):
    customer = _get_store_customer(request)
    if not customer:
        return redirect("store_login")
    cart_id = _get_cart_id(customer["id"])
    if cart_id:
        try:
            requests.delete(f"{CART_SERVICE_URL}/cart-items/{cart_id}/{book_id}/", timeout=3)
            messages.success(request, "Đã xóa sản phẩm khỏi giỏ hàng.")
        except Exception as e:
            messages.error(request, f"Lỗi: {e}")
    return redirect("store_cart")


def store_checkout(request):
    customer = _get_store_customer(request)
    if not customer:
        return redirect("store_login")
    
    if request.method != "POST":
        return redirect("store_cart")

    # 1. Get cart items
    try:
        r_cart = requests.get(f"{CART_SERVICE_URL}/carts/{customer['id']}/", timeout=3)
        cart_data = r_cart.json()
        raw_items = cart_data.get("items", [])
    except Exception as e:
        messages.error(request, f"Lỗi lấy thông tin giỏ hàng: {e}")
        return redirect("store_cart")

    if not raw_items:
        messages.error(request, "Giỏ hàng trống.")
        return redirect("store_home")

    # 2. Pre-check stock and gather prices
    items_to_order = []
    total_price = 0
    try:
        books_r = requests.get(f"{BOOK_SERVICE_URL}/books/", timeout=3)
        books_map = {b["id"]: b for b in books_r.json()}
        
        clothes_r = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/", timeout=3)
        clothes_map = {c["id"]: c for c in clothes_r.json()}
        
        for ri in raw_items:
            actual_id = ri["book_id"]
            if actual_id > 1000000:
                c_id = actual_id - 1000000
                clothe = clothes_map.get(c_id)
                if not clothe:
                    messages.error(request, f"Quần áo #{c_id} không tồn tại.")
                    return redirect("store_cart")
                if clothe["stock"] < ri["quantity"]:
                    messages.error(request, f"Sản phẩm '{clothe['name']}' không đủ hàng (Còn lại: {clothe['stock']}).")
                    return redirect("store_cart")
                price = float(clothe["price"])
                items_to_order.append({
                    "book_id": actual_id, # Giữ nguyên ID mapping để order lưu
                    "quantity": ri["quantity"],
                    "price": price,
                    "title": clothe["name"],
                    "is_clothe": True,
                    "real_id": c_id
                })
                total_price += price * ri["quantity"]
            else:
                book = books_map.get(actual_id)
                if not book:
                    messages.error(request, f"Sách #{actual_id} không tồn tại.")
                    return redirect("store_cart")
                if book["stock"] < ri["quantity"]:
                    messages.error(request, f"Sách '{book['title']}' không đủ hàng (Còn lại: {book['stock']}).")
                    return redirect("store_cart")
                price = float(book["price"])
                items_to_order.append({
                    "book_id": actual_id,
                    "quantity": ri["quantity"],
                    "price": price,
                    "title": book["title"],
                    "is_clothe": False,
                    "real_id": actual_id
                })
                total_price += price * ri["quantity"]
    except Exception as e:
        messages.error(request, f"Lỗi kiểm tra kho: {e}")
        return redirect("store_cart")

    # 3. Reduce stock first
    reduced_items = []
    stock_failed = False
    for item in items_to_order:
        try:
            if item.get("is_clothe"):
                res = requests.post(f"{CLOTHE_SERVICE_URL}/clothes/{item['real_id']}/reduce-stock/", 
                                    json={"quantity": item["quantity"]}, timeout=3)
            else:
                res = requests.post(f"{BOOK_SERVICE_URL}/books/{item['real_id']}/reduce-stock/", 
                                    json={"quantity": item["quantity"]}, timeout=3)
                                    
            if res.status_code == 200:
                reduced_items.append(item)
            else:
                stock_failed = True
                error_msg = res.json().get("error", "Lỗi không xác định")
                messages.error(request, f"Không thể trừ kho cho '{item['title']}': {error_msg}")
                break
        except Exception as e:
            stock_failed = True
            messages.error(request, f"Lỗi kết nối khi trừ kho: {e}")
            break

    if stock_failed:
        # Rollback
        for ri in reduced_items:
            try:
                if ri.get("is_clothe"):
                    requests.post(f"{CLOTHE_SERVICE_URL}/clothes/{ri['real_id']}/restore-stock/", json={"quantity": ri["quantity"]}, timeout=3)
                else:
                    requests.post(f"{BOOK_SERVICE_URL}/books/{ri['real_id']}/restore-stock/", json={"quantity": ri["quantity"]}, timeout=3)
            except Exception:
                pass
        return redirect("store_cart")

    # 4. Create the Order
    province = request.POST.get("province", "Khác")
    address_detail = request.POST.get("address_detail", "")
    full_address = f"{address_detail}, {province}"
    payment_method = request.POST.get("payment_method", "cod")
    
    shipping_fee = 0
    if province not in ["Hà Nội", "Hồ Chí Minh"]:
        shipping_fee = 30000
    
    order_data = {
        "customer_id": customer["id"],
        "total_price": total_price,
        "shipping_fee": shipping_fee,
        "shipping_address": full_address,
        "payment_method": payment_method,
        "items": [{"book_id": i["book_id"], "quantity": i["quantity"], "price": i["price"]} for i in items_to_order]
    }
    
    try:
        r_order = requests.post(f"{ORDER_SERVICE_URL}/orders/", json=order_data, timeout=5)
        if r_order.status_code == 201:
            order_resp = r_order.json()

            # 5. Clear cart
            try:
                requests.delete(f"{CART_SERVICE_URL}/carts/{customer['id']}/clear/", timeout=3)
            except Exception:
                pass
            
            # 6. VNPay Simulation
            if payment_method == 'vnpay':
                try:
                    pay_res = requests.post(f"{PAY_SERVICE_URL}/payments/", json={
                        "order_id": order_resp.get("id"),
                        "customer_id": customer["id"],
                        "amount": order_resp.get("grand_total"),
                        "method": "vnpay",
                    }, timeout=3)
                    if pay_res.status_code == 201:
                        pay_data = pay_res.json()
                        return render(request, "store_vnpay_sim.html", {
                            "order": order_resp,
                            "payment": pay_data
                        })
                except Exception:
                    pass

            messages.success(request, "Đặt hàng thành công! Hệ thống Saga đang xử lý đơn hàng của bạn.")
            return render(request, "store_success.html", {"customer": customer, "order": order_resp})

        else:
            # Rollback
            for ri in reduced_items:
                try:
                    if ri.get("is_clothe"):
                        requests.post(f"{CLOTHE_SERVICE_URL}/clothes/{ri['real_id']}/restore-stock/", json={"quantity": ri["quantity"]}, timeout=3)
                    else:
                        requests.post(f"{BOOK_SERVICE_URL}/books/{ri['real_id']}/restore-stock/", json={"quantity": ri["quantity"]}, timeout=3)
                except Exception:
                    pass
            messages.error(request, f"Lỗi tạo đơn hàng: {r_order.text}")
    except Exception as e:
        for ri in reduced_items:
            try:
                if ri.get("is_clothe"):
                    requests.post(f"{CLOTHE_SERVICE_URL}/clothes/{ri['real_id']}/restore-stock/", json={"quantity": ri["quantity"]}, timeout=3)
                else:
                    requests.post(f"{BOOK_SERVICE_URL}/books/{ri['real_id']}/restore-stock/", json={"quantity": ri["quantity"]}, timeout=3)
            except Exception:
                pass
        messages.error(request, f"Lỗi kết nối order-service: {e}")
    
    return redirect("store_cart")


def store_orders(request):
    customer = _get_store_customer(request)
    if not customer:
        messages.error(request, "Vui lòng đăng nhập để xem đơn hàng.")
        return redirect("store_login")
    orders = []
    try:
        r = requests.get(f"{ORDER_SERVICE_URL}/orders/customer/{customer['id']}/", timeout=5)
        orders = r.json()
        if not isinstance(orders, list):
            orders = []
    except Exception:
        pass
    return render(request, "store_orders.html", {
        "orders": orders,
        "customer": customer,
    })


def store_order_detail(request, order_id):
    customer = _get_store_customer(request)
    if not customer:
        return redirect("store_login")
    order = None
    shipment = None
    payment = None
    try:
        r = requests.get(f"{ORDER_SERVICE_URL}/orders/{order_id}/", timeout=5)
        if r.status_code == 200:
            order = r.json()
            if order.get("customer_id") != customer["id"]:
                messages.error(request, "Bạn không có quyền xem đơn hàng này.")
                return redirect("store_orders")
            # Enrich items with book info
            books_map = {}
            clothes_map = {}
            try:
                books = requests.get(f"{BOOK_SERVICE_URL}/books/", timeout=3).json()
                if isinstance(books, list):
                    books_map = {b["id"]: b for b in books}
            except Exception:
                pass
            try:
                clothes = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/", timeout=3).json()
                if isinstance(clothes, list):
                    clothes_map = {c["id"]: c for c in clothes}
            except Exception:
                pass
            for item in order.get("items", []):
                actual_id = item.get("book_id")
                if isinstance(actual_id, int) and actual_id > 1000000:
                    clothe_id = actual_id - 1000000
                    clothe = clothes_map.get(clothe_id, {})
                    item["book_title"] = clothe.get("name", f"Quần áo #{clothe_id}")
                    item["book_author"] = clothe.get("material", "Thời trang")
                    item["image_url"] = f"https://loremflickr.com/100/140/fashion,clothes?lock={clothe_id}"
                else:
                    book = books_map.get(actual_id, {})
                    item["book_title"] = book.get("title", f"Sách #{actual_id}")
                    item["book_author"] = book.get("author", "")
                    item["image_url"] = f"https://loremflickr.com/100/140/book?lock={actual_id}"
            # Get shipment info
            try:
                r_ship = requests.get(f"{SHIP_SERVICE_URL}/shipments/order/{order_id}/", timeout=3)
                if r_ship.status_code == 200:
                    shipment = r_ship.json()
            except Exception:
                pass
            # Get payment info
            try:
                r_pay = requests.get(f"{PAY_SERVICE_URL}/payments/order/{order_id}/", timeout=3)
                if r_pay.status_code == 200:
                    pay_data = r_pay.json()
                    if isinstance(pay_data, list) and pay_data:
                        payment = pay_data[0]
            except Exception:
                pass
    except Exception:
        pass
    return render(request, "store_order_detail.html", {
        "order": order,
        "customer": customer,
        "shipment": shipment,
        "payment": payment,
    })


def store_cancel_order(request, order_id):
    customer = _get_store_customer(request)
    if not customer:
        return redirect("store_login")
    
    if request.method == "POST":
        try:
            # Gửi yêu cầu DELETE sang order-service để hủy đơn và hoàn kho
            r = requests.delete(f"{ORDER_SERVICE_URL}/orders/{order_id}/", timeout=5)
            if r.status_code == 200:
                messages.success(request, "Đã hủy đơn hàng thành công và hệ thống đang hoàn lại sách vào kho.")
            else:
                resp = r.json()
                error_msg = resp.get("error", "Không thể hủy đơn hàng lúc này.")
                messages.error(request, f"Lỗi: {error_msg}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối khi hủy đơn: {e}")
            
    return redirect("store_order_detail", order_id=order_id)

def store_payment_simulate(request, order_id):
    customer = _get_store_customer(request)
    if not customer:
        return redirect("store_login")
    
    try:
        # 1. Get transaction info from pay-service
        r_pay_list = requests.get(f"{PAY_SERVICE_URL}/payments/order/{order_id}/", timeout=3)
        if r_pay_list.status_code == 200:
            payments = r_pay_list.json()
            if payments:
                pay = payments[0]
                # 2. CALL THE NEW SECURE WEBHOOK (Production approach)
                requests.post(f"{PAY_SERVICE_URL}/payments/confirm-payment/", 
                               json={
                                   "order_id": order_id,
                                   "transaction_id": pay["transaction_id"],
                                   "secure_token": "SECRET_PAYMENT_TOKEN" # In real life, this is a calculated signature
                               }, timeout=3)
                
                messages.success(request, "Thanh toán thành công! Hệ thống đang xử lý vận chuyển.")
            else:
                messages.error(request, "Không tìm thấy thông tin thanh toán cho đơn hàng này.")
        else:
            messages.error(request, "Lỗi kết nối tới dịch vụ thanh toán.")
    except Exception as e:
        messages.error(request, f"Lỗi xử lý thanh toán: {e}")
    
    return redirect("store_order_detail", order_id=order_id)


def store_confirm_receipt(request, order_id):
    customer = _get_store_customer(request)
    if not customer:
        return redirect("store_login")
    
    try:
        # 1. Update order status to delivered
        requests.patch(f"{ORDER_SERVICE_URL}/orders/{order_id}/", 
                       json={"status": "delivered"}, timeout=5)
        
        # 2. Update shipment status to delivered
        r_ship = requests.get(f"{SHIP_SERVICE_URL}/shipments/order/{order_id}/", timeout=3)
        if r_ship.status_code == 200:
            shipment = r_ship.json()
            ship_id = shipment["id"]
            requests.patch(f"{SHIP_SERVICE_URL}/shipments/{ship_id}/", 
                           json={"status": "delivered"}, timeout=3)
        
        messages.success(request, "Xác nhận nhận hàng thành công. Bạn có thể để lại đánh giá cho sản phẩm.")
    except Exception as e:
        messages.error(request, f"Lỗi xác nhận: {e}")
    
    return redirect("store_order_detail", order_id=order_id)


# ── ADMIN ORDER VIEWS ────────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_order_list(request):
    orders = []
    customers_map = {}
    try:
        r = requests.get(f"{ORDER_SERVICE_URL}/orders/", timeout=5)
        orders = r.json()
        if not isinstance(orders, list):
            orders = []
    except Exception:
        pass
    try:
        customers = requests.get(f"{CUSTOMER_SERVICE_URL}/customers/", timeout=3).json()
        if isinstance(customers, list):
            customers_map = {c["id"]: c for c in customers}
    except Exception:
        pass
    for order in orders:
        cust = customers_map.get(order.get("customer_id"), {})
        order["customer_name"] = cust.get("name", f"KH #{order.get('customer_id')}")
        order["customer_email"] = cust.get("email", "")
    return render(request, "orders.html", {"orders": orders})


@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_order_detail(request, order_id):
    if request.method == "POST":
        new_status = request.POST.get("status")
        if new_status:
            try:
                requests.patch(f"{ORDER_SERVICE_URL}/orders/{order_id}/",
                               json={"status": new_status}, timeout=5)
                messages.success(request, f"Đã cập nhật trạng thái đơn hàng #{order_id}.")
            except Exception as e:
                messages.error(request, f"Lỗi: {e}")
        return redirect("admin_order_detail", order_id=order_id)

    order = None
    try:
        r = requests.get(f"{ORDER_SERVICE_URL}/orders/{order_id}/", timeout=5)
        if r.status_code == 200:
            order = r.json()
            # Customer info
            try:
                customers = requests.get(f"{CUSTOMER_SERVICE_URL}/customers/", timeout=3).json()
                cust = next((c for c in customers if c["id"] == order.get("customer_id")), {})
                order["customer_name"] = cust.get("name", f"KH #{order.get('customer_id')}")
                order["customer_email"] = cust.get("email", "")
            except Exception:
                order["customer_name"] = f"KH #{order.get('customer_id')}"
                order["customer_email"] = ""
            # Enrich items
            books_map = {}
            clothes_map = {}
            try:
                books = requests.get(f"{BOOK_SERVICE_URL}/books/", timeout=3).json()
                if isinstance(books, list):
                    books_map = {b["id"]: b for b in books}
            except Exception:
                pass
            try:
                clothes = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/", timeout=3).json()
                if isinstance(clothes, list):
                    clothes_map = {c["id"]: c for c in clothes}
            except Exception:
                pass
            for item in order.get("items", []):
                actual_id = item.get("book_id")
                if isinstance(actual_id, int) and actual_id > 1000000:
                    clothe_id = actual_id - 1000000
                    clothe = clothes_map.get(clothe_id, {})
                    item["book_title"] = clothe.get("name", f"Quần áo #{clothe_id}")
                    item["book_author"] = clothe.get("material", "Thời trang")
                    item["image_url"] = f"https://loremflickr.com/100/140/fashion,clothes?lock={clothe_id}"
                else:
                    book = books_map.get(actual_id, {})
                    item["book_title"] = book.get("title", f"Sách #{actual_id}")
                    item["book_author"] = book.get("author", "")
                    item["image_url"] = f"https://loremflickr.com/100/140/book?lock={actual_id}"
    except Exception:
        pass
    return render(request, "order_detail.html", {"order": order})


# ── ADMIN STAFF VIEWS ─────────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_staff_list(request):
    error = None
    staff = []
    if request.method == "POST":
        data = {
            "name": request.POST.get("name"),
            "email": request.POST.get("email"),
            "phone": request.POST.get("phone", ""),
            "role": request.POST.get("role", "sales"),
        }
        try:
            r = requests.post(f"{STAFF_SERVICE_URL}/staff/", json=data, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Thêm nhân viên thành công!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Không kết nối được staff-service: {e}")
        return redirect("admin_staff_list")
    try:
        r = requests.get(f"{STAFF_SERVICE_URL}/staff/", timeout=3)
        staff = r.json()
        if not isinstance(staff, list):
            staff = []
    except Exception as e:
        error = str(e)
    return render(request, "staff.html", {"staff": staff, "error": error})


# ── ADMIN MANAGER VIEWS ──────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_manager_list(request):
    error = None
    managers = []
    if request.method == "POST":
        data = {
            "name": request.POST.get("name"),
            "email": request.POST.get("email"),
            "phone": request.POST.get("phone", ""),
            "department": request.POST.get("department", "general"),
        }
        try:
            r = requests.post(f"{MANAGER_SERVICE_URL}/managers/", json=data, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Thêm quản lý thành công!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Không kết nối được manager-service: {e}")
        return redirect("admin_manager_list")
    try:
        r = requests.get(f"{MANAGER_SERVICE_URL}/managers/", timeout=3)
        managers = r.json()
        if not isinstance(managers, list):
            managers = []
    except Exception as e:
        error = str(e)
    return render(request, "managers.html", {"managers": managers, "error": error})


# ── ADMIN CATALOG VIEWS ──────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_catalog_list(request):
    error = None
    categories = []
    if request.method == "POST":
        data = {
            "name": request.POST.get("name"),
            "description": request.POST.get("description", ""),
        }
        try:
            r = requests.post(f"{PRODUCT_CATEGORY_SERVICE_URL}/categories/", json=data, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Thêm danh mục thành công!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Không kết nối được product-service: {e}")
        return redirect("admin_catalog_list")
    try:
        r = requests.get(f"{PRODUCT_CATEGORY_SERVICE_URL}/categories/", timeout=3)
        categories = r.json()
        if not isinstance(categories, list):
            categories = []
    except Exception as e:
        error = str(e)
    return render(request, "catalog.html", {"categories": categories, "error": error})


# ── ADMIN PAYMENT VIEWS ──────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_payment_list(request):
    payments = []
    try:
        r = requests.get(f"{PAY_SERVICE_URL}/payments/", timeout=5)
        payments = r.json()
        if not isinstance(payments, list):
            payments = []
    except Exception:
        pass
    return render(request, "payments.html", {"payments": payments})


# ── ADMIN SHIPMENT VIEWS ─────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_shipment_list(request):
    shipments = []
    try:
        r = requests.get(f"{SHIP_SERVICE_URL}/shipments/", timeout=5)
        shipments = r.json()
        if not isinstance(shipments, list):
            shipments = []
    except Exception:
        pass
    return render(request, "shipments.html", {"shipments": shipments})


# ── ADMIN REVIEW VIEWS ───────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_review_list(request):
    reviews = []
    try:
        r = requests.get(f"{COMMENT_RATE_SERVICE_URL}/reviews/", timeout=5)
        reviews = r.json()
        if not isinstance(reviews, list):
            reviews = []
    except Exception:
        pass
    return render(request, "reviews.html", {"reviews": reviews})


# ── STORE REVIEW VIEWS ───────────────────────────────────────

def store_add_review(request, book_id):
    customer = _get_store_customer(request)
    if not customer:
        messages.error(request, "Vui lòng đăng nhập để đánh giá.")
        return redirect("store_login")
    
    # Check if customer has bought this book and order is delivered
    has_purchased = False
    try:
        r_orders = requests.get(f"{ORDER_SERVICE_URL}/orders/customer/{customer['id']}/", timeout=5)
        if r_orders.status_code == 200:
            orders = r_orders.json()
            for order in orders:
                if order.get("status") == "delivered":
                    for item in order.get("items", []):
                        if item.get("book_id") == book_id:
                            has_purchased = True
                            break
                if has_purchased: break
    except Exception:
        pass
    
    if not has_purchased:
        messages.error(request, "Bạn chỉ có thể đánh giá sách sau khi đã nhận hàng thành công.")
        return redirect("store_book_detail", book_id=book_id)

    if request.method == "POST":
        data = {
            "customer_id": customer["id"],
            "book_id": book_id,
            "rating": int(request.POST.get("rating", 5)),
            "comment": request.POST.get("comment", ""),
        }
        try:
            r = requests.post(f"{COMMENT_RATE_SERVICE_URL}/reviews/", json=data, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Đánh giá thành công!")
            else:
                resp = r.json()
                if "unique" in str(resp).lower() or "unique_together" in str(resp).lower():
                    messages.error(request, "Bạn đã đánh giá sách này rồi.")
                else:
                    messages.error(request, f"Lỗi: {resp}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối: {e}")
    return redirect("store_book_detail", book_id=book_id)


def api_secure_echo(request):
    return JsonResponse({
        "status": "ok",
        "message": "Gateway token validation passed",
    })


# ── CLOTHE VIEWS ─────────────────────────────────────────────

@user_passes_test(is_staff_check, login_url='/admin/login/')
def admin_clothe_list(request):
    error = None
    clothes = []
    if request.method == "POST":
        data = {
            "name": request.POST.get("name"),
            "material": request.POST.get("material"),
            "price": request.POST.get("price"),
            "stock": request.POST.get("stock"),
        }
        try:
            r = requests.post(f"{CLOTHE_SERVICE_URL}/clothes/", json=data, timeout=3)
            if r.status_code in (200, 201):
                messages.success(request, "Thêm quần áo thành công!")
            else:
                messages.error(request, f"Lỗi: {r.text}")
        except Exception as e:
            messages.error(request, f"Lỗi kết nối: {e}")
        return redirect("admin_clothe_list")
        
    try:
        r = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/", timeout=3)
        clothes = r.json()
    except Exception as e:
        error = str(e)
    return render(request, "clothes.html", {"clothes": clothes, "error": error})

def store_clothes(request):
    clothes = []
    try:
        r = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/", timeout=3)
        clothes = r.json()
    except Exception:
        pass
    
    customer = _get_store_customer(request)
    return render(request, "store_clothes.html", {"clothes": clothes, "customer": customer})

def store_clothe_detail(request, clothe_id):
    clothe = None
    try:
        r = requests.get(f"{CLOTHE_SERVICE_URL}/clothes/{clothe_id}/", timeout=3)
        if r.status_code == 200:
            clothe = r.json()
    except Exception:
        pass
    return render(request, "store_clothe_detail.html", {"clothe": clothe, "customer": _get_store_customer(request)})


def store_item_detail(request, item_type, item_id):
    """Generic detail page for any item type used by AI recommendation cards."""
    customer = _get_store_customer(request)
    item_type = str(item_type or "").lower()
    item = None
    error = ""

    try:
        item = _fetch_single_item_detail(item_type, item_id)
        if not item:
            error = "Không tìm thấy thông tin chi tiết sản phẩm."
    except Exception as e:
        error = str(e)

    if item_type == "book" and item:
        item_name = item.get("title") or "Book"
        item_price = item.get("price") or 0
        item_stock = _safe_int(item.get("stock"), 0)
        item_category = item.get("category") or "book"
    elif item_type == "clothe" and item:
        item_name = item.get("name") or "Clothe"
        item_price = item.get("price") or 0
        item_stock = _safe_int(item.get("stock"), 0)
        item_category = item.get("category") or "fashion"
    elif item:
        item_name = item.get("name") or item.get("title") or "Product"
        item_price = item.get("price") or 0
        item_stock = _safe_int(item.get("stock"), 0)
        item_category = item.get("category") or item_type
    else:
        item_name = ""
        item_price = 0
        item_stock = 0
        item_category = ""

    metadata = (item or {}).get("metadata") or {}
    item_description = metadata.get("description") or (item or {}).get("description") or "Mô tả sản phẩm đang được cập nhật."
    item_image_url = metadata.get("image_url") or _build_recommendation_image(item_type, item_id, item_name)

    payload = {
        "id": item_id,
        "type": item_type,
        "name": item_name,
        "price": item_price,
        "stock": item_stock,
        "in_stock": item_stock > 0,
        "category": item_category,
        "description": item_description,
        "raw": item or {},
        "image_url": item_image_url,
    }

    return render(
        request,
        "store_item_detail.html",
        {
            "customer": customer,
            "item": payload,
            "error": error,
        },
    )


# ── AI SERVICES PROXY VIEWS ──────────────────────────────────

def ai_behavior_analysis_proxy(request):
    """Proxy requests to Behavior Analysis Service (8014)"""
    customer = _get_store_customer(request)
    if not customer:
        return JsonResponse({"error": "Authentication required"}, status=401)
    
    if request.method == "GET":
        try:
            # Forward GET request for customer analysis
            r = requests.get(
                f"{BEHAVIOR_ANALYSIS_SERVICE_URL}/api/behavior/customer/{_safe_int(customer['id'])}/analysis/",
                headers={"Authorization": request.headers.get("Authorization", "")},
                timeout=5
            )
            return JsonResponse(r.json(), status=r.status_code)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Method not allowed"}, status=405)


def ai_behavior_track_proxy(request):
    """Proxy tracking events from storefront to behavior-analysis-service."""
    customer = _get_store_customer(request)
    if not customer:
        return JsonResponse({"error": "Authentication required"}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        import json

        body = {}
        if request.body:
            body = json.loads(request.body)

        payload = {
            "customer_id": _safe_int(customer.get("id"), 0),
            "event_type": body.get("event_type", "page_view"),
            "event_data": body.get("event_data", {}),
            "session_id": body.get("session_id") or request.session.session_key,
            "device": body.get("device", "desktop"),
            "category": body.get("category"),
            "product_id": body.get("product_id"),
        }

        r = requests.post(
            f"{BEHAVIOR_ANALYSIS_SERVICE_URL}/api/behavior/track/",
            json=payload,
            headers={"Authorization": request.headers.get("Authorization", "")},
            timeout=5,
        )
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def ai_recommendations_proxy(request):
    """Return personalized multi-item recommendations from real catalog + behavior data."""
    customer = _get_store_customer(request)
    if not customer:
        return JsonResponse({"error": "Authentication required"}, status=401)

    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    item_type = request.GET.get("type", "all").strip().lower()
    query = request.GET.get("q", "").strip()
    limit = _safe_int(request.GET.get("limit", 8), 8)

    behavior_data = _fetch_behavior_analysis(customer.get("id"))
    products = _fetch_ecommerce_products()
    recommendations = _build_ai_recommendations(
        products,
        behavior_data,
        limit=limit,
        item_type=item_type if item_type else "all",
        query=query,
    )

    return JsonResponse(
        {
            "success": True,
            "data": {
                "customer_id": customer.get("id"),
                "segment": behavior_data.get("segment", "Regular"),
                "predicted_categories": _normalize_behavior_categories(behavior_data),
                "total": len(recommendations),
                "recommendations": recommendations,
            },
        },
        status=200,
    )


def ai_chatbot_proxy(request):
    """Proxy requests to Consulting Chatbot Service (8015)"""
    customer = _get_store_customer(request)
    if not customer:
        return JsonResponse({"error": "Authentication required"}, status=401)
    
    if request.method == "POST":
        try:
            import json
            body = json.loads(request.body)
            # Add customer_id to the chat message
            body["customer_id"] = customer["id"]
            
            r = requests.post(
                f"{CHATBOT_SERVICE_URL}/api/chat/message/",
                json=body,
                headers={"Authorization": request.headers.get("Authorization", "")},
                timeout=10
            )
            return JsonResponse(r.json(), status=r.status_code)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Method not allowed"}, status=405)


def ai_chatbot_health(request):
    """Health check endpoint for Consulting Chatbot Service"""
    try:
        r = requests.get(f"{CHATBOT_SERVICE_URL}/api/chat/health/", timeout=3)
        if r.status_code == 200:
            return JsonResponse({"status": "healthy", "service": "consulting-chatbot"}, status=200)
        else:
            return JsonResponse({"status": "unhealthy"}, status=r.status_code)
    except Exception as e:
        return JsonResponse({"error": str(e), "service": "consulting-chatbot"}, status=503)


@user_passes_test(is_staff_check, login_url='/admin/login/')
def ai_chatbot_index_status(request):
    """Admin-only proxy: check chatbot KB index status."""
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        r = requests.get(f"{CHATBOT_SERVICE_URL}/api/chat/index/status/", timeout=6)
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@user_passes_test(is_staff_check, login_url='/admin/login/')
def ai_chatbot_rebuild_index(request):
    """Admin-only proxy: trigger chatbot KB index rebuild."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        import json

        body = {}
        if request.body:
            body = json.loads(request.body)

        payload = {
            "force": bool(body.get("force", True)),
            "include_product_sync": bool(body.get("include_product_sync", True)),
        }

        r = requests.post(
            f"{CHATBOT_SERVICE_URL}/api/chat/index/rebuild/",
            json=payload,
            timeout=30,
        )
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def _proxy_unified_ai_post(request, endpoint: str, timeout: int = 30):
    import json

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    body = {}
    if request.body:
        body = json.loads(request.body)

    try:
        r = requests.post(
            f"{UNIFIED_AI_SERVICE_URL}{endpoint}",
            json=body,
            headers={"Authorization": request.headers.get("Authorization", "")},
            timeout=timeout,
        )
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def ai_service_health_proxy(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        r = requests.get(f"{UNIFIED_AI_SERVICE_URL}/api/health/", timeout=5)
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def ai_service_generate_data_proxy(request):
    return _proxy_unified_ai_post(request, "/api/data/generate/", timeout=30)


def ai_service_train_models_proxy(request):
    return _proxy_unified_ai_post(request, "/api/models/train/", timeout=3600)


def ai_service_build_graph_proxy(request):
    return _proxy_unified_ai_post(request, "/api/graph/build/", timeout=300)


def ai_service_rag_query_proxy(request):
    return _proxy_unified_ai_post(request, "/api/rag/query/", timeout=120)


def ai_service_run_pipeline_proxy(request):
    return _proxy_unified_ai_post(request, "/api/pipeline/run/", timeout=7200)

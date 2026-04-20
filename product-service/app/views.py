from rest_framework.views import APIView
from rest_framework.response import Response
import os
import requests
import time
from urllib.parse import quote
from .models import Category, BookCatalog, ProductCatalog
from .serializers import CategorySerializer, BookCatalogSerializer, ProductCatalogSerializer
from .seed_data import CATEGORY_TREE, SAMPLE_PRODUCTS


BOOK_SERVICE_URL = os.environ.get('BOOK_SERVICE_URL', '').strip()
CLOTHE_SERVICE_URL = os.environ.get('CLOTHE_SERVICE_URL', '').strip()
REQUEST_TIMEOUT_SECONDS = 4
EXTERNAL_SYNC_COOLDOWN_SECONDS = int(os.environ.get('EXTERNAL_SYNC_COOLDOWN_SECONDS', '60'))
_LAST_EXTERNAL_SYNC_TS = 0.0


class HealthCheck(APIView):
    def get(self, request):
        return Response({'status': 'ok', 'service': 'product-service'})


def _build_seed_image(item_type, sku, name):
    label = str(item_type or "product").upper()
    safe_name = str(name or "Product")[:30].replace("&", "and")
    safe_sku = str(sku or "SKU")
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='720' height='560' viewBox='0 0 720 560'>"
        "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
        "<stop offset='0%' stop-color='#eef2ff'/><stop offset='100%' stop-color='#f8fafc'/>"
        "</linearGradient></defs>"
        "<rect width='720' height='560' fill='url(#g)'/>"
        "<rect x='40' y='40' width='640' height='480' rx='28' fill='white' stroke='#d7def3' stroke-width='3'/>"
        f"<text x='76' y='122' font-size='34' font-weight='700' font-family='Arial, sans-serif' fill='#3340a7'>{label}</text>"
        f"<text x='76' y='188' font-size='30' font-weight='700' font-family='Arial, sans-serif' fill='#111827'>{safe_name}</text>"
        f"<text x='76' y='236' font-size='24' font-family='Arial, sans-serif' fill='#6b7280'>{safe_sku}</text>"
        "<rect x='76' y='268' width='148' height='8' rx='4' fill='#4361ee' fill-opacity='0.35'/>"
        "</svg>"
    )
    return f"data:image/svg+xml,{quote(svg)}"


def _build_seed_description(item):
    name = item.get('name', 'Sản phẩm')
    item_type = str(item.get('item_type', 'general')).lower()
    category = item.get('category', 'General')
    return f"{name} thuộc ngành {item_type}, danh mục {category}. Sản phẩm chính hãng, phù hợp nhu cầu sử dụng hàng ngày và đã được kiểm duyệt trong kho P-Shop."


def _build_default_metadata_fields(item):
    item_type = str(item.get('item_type') or 'general').lower()
    name = str(item.get('name') or 'Product')

    default_brand = {
        'mobile': 'P-Mobile',
        'laptop': 'P-Laptop',
        'computer': 'P-Computer',
        'tablet': 'P-Tablet',
        'audio': 'P-Audio',
        'accessory': 'P-Accessory',
        'home': 'P-Home',
        'beauty': 'P-Beauty',
        'sports': 'P-Sports',
        'fashion': 'P-Fashion',
        'book': 'P-Author',
    }.get(item_type, 'P-Store')

    default_material = {
        'fashion': 'Cotton Blend',
        'accessory': 'ABS + Aluminum',
        'audio': 'ABS Composite',
        'home': 'Stainless Steel',
        'beauty': 'Dermatology-grade',
        'sports': 'Polyester Blend',
    }.get(item_type, 'Standard')

    return {
        'brand_or_author': item.get('brand_or_author') or default_brand,
        'material': item.get('material') or default_material,
        'description': item.get('description') or _build_seed_description(item),
        'image_url': item.get('image_url') or _build_seed_image(item.get('item_type'), item.get('sku'), name),
    }


def _to_legacy_book_payload(product):
    metadata = product.metadata or {}
    return {
        'id': product.id,
        'title': product.name,
        'author': metadata.get('brand_or_author', ''),
        'category': product.category.name,
        'price': float(product.price),
        'stock': product.stock,
    }


def _to_legacy_clothe_payload(product):
    metadata = product.metadata or {}
    return {
        'id': product.id,
        'name': product.name,
        'brand': metadata.get('brand_or_author', ''),
        'material': metadata.get('material', ''),
        'category': product.category.name,
        'price': float(product.price),
        'stock': product.stock,
    }


def _build_category_tree(root):
    children = []
    for child in root.children.all().order_by('name'):
        children.append(_build_category_tree(child))

    return {
        'id': root.id,
        'name': root.name,
        'description': root.description,
        'children': children,
    }


def _is_external_service_url(url):
    if not url:
        return False
    lowered = url.lower()
    return 'product-service' not in lowered


def _upsert_external_book_item(raw_book):
    try:
        external_id = int(raw_book.get('id'))
    except (TypeError, ValueError):
        return False

    title = str(raw_book.get('title') or '').strip()
    if not title:
        return False

    category_name = str(raw_book.get('category') or 'Programming').strip() or 'Programming'
    category, _ = Category.objects.get_or_create(
        name=category_name,
        defaults={'description': f'{category_name} category'},
    )

    try:
        price = float(raw_book.get('price', 0) or 0)
    except (TypeError, ValueError):
        price = 0

    try:
        stock = int(raw_book.get('stock', 0) or 0)
    except (TypeError, ValueError):
        stock = 0

    sku = f'EXT-BOOK-{external_id}'
    metadata = {
        'seeded': False,
        'source': 'book-service',
        'external_source': 'book-service',
        'external_id': external_id,
        'brand_or_author': str(raw_book.get('author') or 'P-Author'),
        'material': 'Paper',
        'description': f"{title} (đồng bộ từ book-service)",
        'image_url': _build_seed_image('book', sku, title),
        'currency': 'VND',
        'warranty_months': 0,
        'synced_from_external': True,
    }

    item, created = ProductCatalog.objects.get_or_create(
        sku=sku,
        defaults={
            'name': title,
            'item_type': 'book',
            'category': category,
            'price': price,
            'stock': stock,
            'metadata': metadata,
            'is_active': True,
        },
    )

    if created:
        return True

    changed = False
    if item.name != title:
        item.name = title
        changed = True
    if item.item_type != 'book':
        item.item_type = 'book'
        changed = True
    if item.category_id != category.id:
        item.category = category
        changed = True
    if float(item.price) != float(price):
        item.price = price
        changed = True
    if int(item.stock) != int(stock):
        item.stock = stock
        changed = True
    if not item.is_active:
        item.is_active = True
        changed = True
    if (item.metadata or {}) != metadata:
        item.metadata = metadata
        changed = True

    if changed:
        item.save()
    return changed


def _upsert_external_clothe_item(raw_clothe):
    try:
        external_id = int(raw_clothe.get('id'))
    except (TypeError, ValueError):
        return False

    name = str(raw_clothe.get('name') or '').strip()
    if not name:
        return False

    category_name = str(raw_clothe.get('category') or 'Fashion').strip() or 'Fashion'
    category, _ = Category.objects.get_or_create(
        name=category_name,
        defaults={'description': f'{category_name} category'},
    )

    try:
        price = float(raw_clothe.get('price', 0) or 0)
    except (TypeError, ValueError):
        price = 0

    try:
        stock = int(raw_clothe.get('stock', 0) or 0)
    except (TypeError, ValueError):
        stock = 0

    sku = f'EXT-CLOTHE-{external_id}'
    metadata = {
        'seeded': False,
        'source': 'clothe-service',
        'external_source': 'clothe-service',
        'external_id': external_id,
        'brand_or_author': str(raw_clothe.get('brand') or 'P-Fashion'),
        'material': str(raw_clothe.get('material') or 'Cotton Blend'),
        'description': f"{name} (đồng bộ từ clothe-service)",
        'image_url': _build_seed_image('fashion', sku, name),
        'currency': 'VND',
        'warranty_months': 1,
        'synced_from_external': True,
    }

    item, created = ProductCatalog.objects.get_or_create(
        sku=sku,
        defaults={
            'name': name,
            'item_type': 'fashion',
            'category': category,
            'price': price,
            'stock': stock,
            'metadata': metadata,
            'is_active': True,
        },
    )

    if created:
        return True

    changed = False
    if item.name != name:
        item.name = name
        changed = True
    if item.item_type != 'fashion':
        item.item_type = 'fashion'
        changed = True
    if item.category_id != category.id:
        item.category = category
        changed = True
    if float(item.price) != float(price):
        item.price = price
        changed = True
    if int(item.stock) != int(stock):
        item.stock = stock
        changed = True
    if not item.is_active:
        item.is_active = True
        changed = True
    if (item.metadata or {}) != metadata:
        item.metadata = metadata
        changed = True

    if changed:
        item.save()
    return changed


def _safe_external_get(url):
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200 and isinstance(response.json(), list):
            return response.json()
    except Exception:
        return []
    return []


def _sync_external_product_services():
    global _LAST_EXTERNAL_SYNC_TS

    now = time.monotonic()
    if now - _LAST_EXTERNAL_SYNC_TS < EXTERNAL_SYNC_COOLDOWN_SECONDS:
        return {
            'book_service_processed': 0,
            'book_service_upserted': 0,
            'clothe_service_processed': 0,
            'clothe_service_upserted': 0,
            'skipped': 'cooldown',
        }

    stats = {
        'book_service_processed': 0,
        'book_service_upserted': 0,
        'clothe_service_processed': 0,
        'clothe_service_upserted': 0,
    }

    if _is_external_service_url(BOOK_SERVICE_URL):
        books = _safe_external_get(f'{BOOK_SERVICE_URL}/books/')
        stats['book_service_processed'] = len(books)
        for raw_book in books:
            if _upsert_external_book_item(raw_book):
                stats['book_service_upserted'] += 1

    if _is_external_service_url(CLOTHE_SERVICE_URL):
        clothes = _safe_external_get(f'{CLOTHE_SERVICE_URL}/clothes/')
        stats['clothe_service_processed'] = len(clothes)
        for raw_clothe in clothes:
            if _upsert_external_clothe_item(raw_clothe):
                stats['clothe_service_upserted'] += 1

    _LAST_EXTERNAL_SYNC_TS = now
    return stats


def _ensure_minimum_catalog_items(minimum=10):
    # Seed is idempotent and also backfills metadata for existing records.
    _ensure_seed_data()
    sync_stats = _sync_external_product_services()
    if ProductCatalog.objects.count() < minimum:
        _ensure_seed_data()
    return {
        'total_products': ProductCatalog.objects.count(),
        'minimum_required': minimum,
        'sync_stats': sync_stats,
    }


def _ensure_seed_data():
    category_by_name = {}

    for root_name, branches in CATEGORY_TREE.items():
        root, _ = Category.objects.get_or_create(
            name=root_name,
            defaults={
                'description': 'Root category for all products',
                'parent': None,
            },
        )
        category_by_name[root_name] = root

        for parent_name, leaves in branches.items():
            parent, _ = Category.objects.get_or_create(
                name=parent_name,
                defaults={
                    'description': f'{parent_name} products',
                    'parent': root,
                },
            )
            if parent.parent_id != root.id:
                parent.parent = root
                parent.save(update_fields=['parent'])

            category_by_name[parent_name] = parent

            for leaf_name in leaves:
                leaf, _ = Category.objects.get_or_create(
                    name=leaf_name,
                    defaults={
                        'description': f'{leaf_name} category',
                        'parent': parent,
                    },
                )
                if leaf.parent_id != parent.id:
                    leaf.parent = parent
                    leaf.save(update_fields=['parent'])

                category_by_name[leaf_name] = leaf

    created_products = 0
    for item in SAMPLE_PRODUCTS:
        cat = category_by_name.get(item['category'])
        if not cat:
            continue

        metadata_defaults = _build_default_metadata_fields(item)
        desired_metadata = {
            'seeded': True,
            'source': 'product-service',
            'brand_or_author': metadata_defaults['brand_or_author'],
            'material': metadata_defaults['material'],
            'description': metadata_defaults['description'],
            'image_url': metadata_defaults['image_url'],
            'currency': 'VND',
            'warranty_months': 12 if item.get('item_type') != 'book' else 0,
        }

        product, created = ProductCatalog.objects.get_or_create(
            sku=item['sku'],
            defaults={
                'name': item['name'],
                'item_type': item['item_type'],
                'category': cat,
                'price': item['price'],
                'stock': item['stock'],
                'metadata': desired_metadata,
                'is_active': True,
            },
        )
        if created:
            created_products += 1
            continue

        # Keep seed endpoint idempotent while backfilling missing media/description fields.
        existing_metadata = product.metadata or {}
        merged_metadata = {**existing_metadata, **desired_metadata}
        changed = False

        if product.name != item['name']:
            product.name = item['name']
            changed = True
        if product.item_type != item['item_type']:
            product.item_type = item['item_type']
            changed = True
        if product.category_id != cat.id:
            product.category = cat
            changed = True
        if float(product.price) != float(item['price']):
            product.price = item['price']
            changed = True
        if int(product.stock) != int(item['stock']):
            product.stock = item['stock']
            changed = True
        if existing_metadata != merged_metadata:
            product.metadata = merged_metadata
            changed = True
        if not product.is_active:
            product.is_active = True
            changed = True

        if changed:
            product.save()

    return {
        'category_count': Category.objects.count(),
        'product_count': ProductCatalog.objects.count(),
        'created_products': created_products,
    }


class CategoryListCreate(APIView):
    def get(self, request):
        _ensure_minimum_catalog_items(minimum=10)

        root_only = request.query_params.get('root', 'false').lower() == 'true'
        as_tree = request.query_params.get('tree', 'false').lower() == 'true'

        queryset = Category.objects.all().select_related('parent')
        if root_only:
            queryset = queryset.filter(parent__isnull=True)

        if as_tree:
            roots = queryset.filter(parent__isnull=True).prefetch_related('children__children')
            data = [_build_category_tree(root) for root in roots.order_by('name')]
            return Response(data)

        return Response(CategorySerializer(queryset.order_by('name'), many=True).data)

    def post(self, request):
        payload = request.data.copy()
        parent_id = payload.get('parent_id')
        if parent_id and not payload.get('parent'):
            payload['parent'] = parent_id

        serializer = CategorySerializer(data=payload)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class CategoryDetail(APIView):
    def get(self, request, pk):
        try:
            cat = Category.objects.get(pk=pk)
            data = CategorySerializer(cat).data
            # Include book_ids in this category
            book_ids = list(BookCatalog.objects.filter(category=cat).values_list('book_id', flat=True))
            data['book_ids'] = book_ids
            data['children'] = CategorySerializer(cat.children.all().order_by('name'), many=True).data
            data['products'] = ProductCatalogSerializer(cat.products.all().order_by('name'), many=True).data
            return Response(data)
        except Category.DoesNotExist:
            return Response({"error": "Category not found"}, status=404)


class BookCatalogListCreate(APIView):
    def get(self, request):
        book_id = request.query_params.get('book_id')
        if book_id:
            items = BookCatalog.objects.filter(book_id=book_id).select_related('category')
        else:
            items = BookCatalog.objects.all().select_related('category')
        return Response(BookCatalogSerializer(items, many=True).data)

    def post(self, request):
        serializer = BookCatalogSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class ProductCatalogListCreate(APIView):
    def get(self, request):
        _ensure_minimum_catalog_items(minimum=10)

        item_type = request.query_params.get('item_type')
        category_name = request.query_params.get('category')
        in_stock_only = request.query_params.get('in_stock', 'false').lower() == 'true'

        items = ProductCatalog.objects.all().select_related('category')
        if item_type:
            items = items.filter(item_type=item_type)
        if category_name:
            items = items.filter(category__name__iexact=category_name)
        if in_stock_only:
            items = items.filter(stock__gt=0)

        return Response(
            {
                'total': items.count(),
                'products': ProductCatalogSerializer(items.order_by('name'), many=True).data,
            }
        )

    def post(self, request):
        serializer = ProductCatalogSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class ProductCatalogDetail(APIView):
    def get(self, request, pk):
        try:
            item = ProductCatalog.objects.select_related('category').get(pk=pk)
        except ProductCatalog.DoesNotExist:
            return Response({'error': 'Product not found'}, status=404)
        return Response(ProductCatalogSerializer(item).data)

    def patch(self, request, pk):
        try:
            item = ProductCatalog.objects.select_related('category').get(pk=pk)
        except ProductCatalog.DoesNotExist:
            return Response({'error': 'Product not found'}, status=404)

        serializer = ProductCatalogSerializer(item, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)


class ProductStockAdjust(APIView):
    def post(self, request, pk, action):
        try:
            item = ProductCatalog.objects.get(pk=pk)
        except ProductCatalog.DoesNotExist:
            return Response({'error': 'Product not found'}, status=404)

        quantity = request.data.get('quantity', 1)
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return Response({'error': 'quantity must be an integer'}, status=400)

        if quantity <= 0:
            return Response({'error': 'quantity must be > 0'}, status=400)

        if action == 'reduce':
            if item.stock < quantity:
                return Response({'error': 'Insufficient stock', 'stock': item.stock}, status=400)
            item.stock -= quantity
        else:
            item.stock += quantity

        item.save(update_fields=['stock', 'updated_at'])
        return Response({'id': item.id, 'stock': item.stock, 'action': action})


class LegacyBooksListCreate(APIView):
    def get(self, request):
        _ensure_minimum_catalog_items(minimum=10)
        items = ProductCatalog.objects.select_related('category').filter(item_type='book', is_active=True).order_by('name')
        return Response([_to_legacy_book_payload(item) for item in items])

    def post(self, request):
        data = request.data
        title = data.get('title')
        if not title:
            return Response({'error': 'title is required'}, status=400)

        category_name = data.get('category') or 'General'
        category, _ = Category.objects.get_or_create(
            name=category_name,
            defaults={'description': f'{category_name} category'},
        )

        sku = data.get('sku') or f"BOO-{ProductCatalog.objects.count() + 1:03d}"
        metadata = {
            'source': 'product-service',
            'brand_or_author': data.get('author', ''),
        }
        item = ProductCatalog.objects.create(
            sku=sku,
            name=title,
            item_type='book',
            category=category,
            price=data.get('price', 0),
            stock=data.get('stock', 0),
            metadata=metadata,
            is_active=True,
        )
        return Response(_to_legacy_book_payload(item), status=201)


class LegacyBookDetail(APIView):
    def get(self, request, pk):
        try:
            item = ProductCatalog.objects.select_related('category').get(pk=pk, item_type='book')
        except ProductCatalog.DoesNotExist:
            return Response({'error': 'Book not found'}, status=404)
        return Response(_to_legacy_book_payload(item))


class LegacyClothesListCreate(APIView):
    def get(self, request):
        _ensure_minimum_catalog_items(minimum=10)
        items = ProductCatalog.objects.select_related('category').filter(item_type='fashion', is_active=True).order_by('name')
        return Response([_to_legacy_clothe_payload(item) for item in items])

    def post(self, request):
        data = request.data
        name = data.get('name')
        if not name:
            return Response({'error': 'name is required'}, status=400)

        category_name = data.get('category') or 'Fashion'
        category, _ = Category.objects.get_or_create(
            name=category_name,
            defaults={'description': f'{category_name} category'},
        )

        sku = data.get('sku') or f"FAS-{ProductCatalog.objects.count() + 1:03d}"
        metadata = {
            'source': 'product-service',
            'brand_or_author': data.get('brand', ''),
            'material': data.get('material', ''),
        }
        item = ProductCatalog.objects.create(
            sku=sku,
            name=name,
            item_type='fashion',
            category=category,
            price=data.get('price', 0),
            stock=data.get('stock', 0),
            metadata=metadata,
            is_active=True,
        )
        return Response(_to_legacy_clothe_payload(item), status=201)


class LegacyClotheDetail(APIView):
    def get(self, request, pk):
        try:
            item = ProductCatalog.objects.select_related('category').get(pk=pk, item_type='fashion')
        except ProductCatalog.DoesNotExist:
            return Response({'error': 'Clothe not found'}, status=404)
        return Response(_to_legacy_clothe_payload(item))


class LegacyStockAdjust(APIView):
    def post(self, request, pk, action):
        try:
            item = ProductCatalog.objects.get(pk=pk)
        except ProductCatalog.DoesNotExist:
            return Response({'error': 'Product not found'}, status=404)

        quantity = request.data.get('quantity', 1)
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return Response({'error': 'quantity must be integer'}, status=400)

        if quantity <= 0:
            return Response({'error': 'quantity must be > 0'}, status=400)

        if action == 'reduce':
            if item.stock < quantity:
                return Response({'error': 'Insufficient stock', 'stock': item.stock}, status=400)
            item.stock -= quantity
        else:
            item.stock += quantity

        item.save(update_fields=['stock', 'updated_at'])
        return Response({'id': item.id, 'stock': item.stock})


class SeedCatalogData(APIView):
    def post(self, request):
        stats = _ensure_seed_data()
        integration_stats = _sync_external_product_services()
        return Response(
            {
                'message': 'Catalog seed completed',
                'stats': stats,
                'integration_stats': integration_stats,
            },
            status=200,
        )


class SimilarProducts(APIView):
    def get(self, request, pk):
        try:
            base = ProductCatalog.objects.select_related('category').get(pk=pk, is_active=True)
        except ProductCatalog.DoesNotExist:
            return Response({'error': 'Product not found'}, status=404)

        limit = request.query_params.get('limit', 8)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 8

        if limit <= 0:
            limit = 8

        same_category = ProductCatalog.objects.select_related('category').filter(
            is_active=True,
            category=base.category,
        ).exclude(pk=base.pk)

        same_type = ProductCatalog.objects.select_related('category').filter(
            is_active=True,
            item_type=base.item_type,
        ).exclude(pk=base.pk)

        # Prefer category affinity first, then type affinity.
        merged = list(same_category.order_by('-stock', 'name')[:limit])
        current_ids = {item.id for item in merged}
        if len(merged) < limit:
            for candidate in same_type.order_by('-stock', 'name'):
                if candidate.id in current_ids:
                    continue
                merged.append(candidate)
                current_ids.add(candidate.id)
                if len(merged) >= limit:
                    break

        return Response(
            {
                'product_id': base.id,
                'item_type': base.item_type,
                'category': base.category.name,
                'total': len(merged),
                'products': ProductCatalogSerializer(merged, many=True).data,
            }
        )


class SyncExternalProducts(APIView):
    def post(self, request):
        ensure_stats = _ensure_minimum_catalog_items(minimum=10)
        return Response(
            {
                'message': 'External product sync completed',
                'stats': ensure_stats,
            },
            status=200,
        )

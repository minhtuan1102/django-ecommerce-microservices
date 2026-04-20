import os
import django
import random

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'product_service.settings')
django.setup()

from app.models import ProductCatalog, Category

def seed_products():
    # Create categories
    categories_data = [
        "Sách văn học", "Sách kinh tế", "Sách kỹ năng",
        "Thời trang nam", "Thời trang nữ", "Phụ kiện"
    ]
    
    cats = {}
    for name in categories_data:
        cat, _ = Category.objects.get_or_create(name=name)
        cats[name] = cat

    products = [
        # Books
        {
            "sku": "book-1", "name": "Đắc Nhân Tâm", "item_type": "book", "category": cats["Sách kỹ năng"],
            "price": 86000, "stock": 100,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1544947950-fa07a98d237f", "description": "Cuốn sách kỹ năng sống nổi tiếng nhất thế giới."}
        },
        {
            "sku": "book-2", "name": "Nhà Giả Kim", "item_type": "book", "category": cats["Sách văn học"],
            "price": 79000, "stock": 50,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1589829085413-56de8ae18c73", "description": "Hành trình tìm kiếm vận mệnh của chàng chăn cừu Santiago."}
        },
        {
            "sku": "book-3", "name": "Cha Giàu Cha Nghèo", "item_type": "book", "category": cats["Sách kinh tế"],
            "price": 125000, "stock": 30,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1592492159418-39f319320569", "description": "Bài học về tài chính cá nhân từ Robert Kiyosaki."}
        },
        {
            "sku": "book-4", "name": "Lược Sử Thời Gian", "item_type": "book", "category": cats["Sách văn học"],
            "price": 150000, "stock": 15,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1532012197267-da84d127e765", "description": "Kiệt tác của Stephen Hawking về vũ trụ."}
        },
        {
            "sku": "book-5", "name": "Suối Nguồn", "item_type": "book", "category": cats["Sách văn học"],
            "price": 245000, "stock": 10,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1543003919-a995d01a5d92", "description": "Tác phẩm kinh điển của Ayn Rand."}
        },
        {
            "sku": "book-6", "name": "Tư Duy Nhanh Và Chậm", "item_type": "book", "category": cats["Sách kỹ năng"],
            "price": 185000, "stock": 25,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1512820790803-83ca734da794", "description": "Khám phá hai hệ thống tư duy của con người."}
        },
        # Fashion
        {
            "sku": "fashion-1", "name": "Áo Polo Nam Classic", "item_type": "fashion", "category": cats["Thời trang nam"],
            "price": 250000, "stock": 120,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1586363104862-3a5e2ab60d99", "description": "Áo polo chất liệu cotton co giãn 4 chiều."}
        },
        {
            "sku": "fashion-2", "name": "Váy Hoa Nhí Vintage", "item_type": "fashion", "category": cats["Thời trang nữ"],
            "price": 380000, "stock": 45,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1572804013307-a9a11117bb41", "description": "Váy hoa nhẹ nhàng cho mùa hè năng động."}
        },
        {
            "sku": "fashion-3", "name": "Quần Jean Slim Fit", "item_type": "fashion", "category": cats["Thời trang nam"],
            "price": 450000, "stock": 60,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d", "description": "Quần jean form dáng chuẩn mực, bền màu."}
        },
        {
            "sku": "fashion-4", "name": "Túi Xách Da Cao Cấp", "item_type": "fashion", "category": cats["Phụ kiện"],
            "price": 1200000, "stock": 12,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1584917033904-493bb3c3cc0a", "description": "Túi xách da thật sang trọng cho quý cô công sở."}
        },
        {
            "sku": "fashion-5", "name": "Giày Sneaker Trắng", "item_type": "fashion", "category": cats["Phụ kiện"],
            "price": 550000, "stock": 80,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1549298916-b41d501d3772", "description": "Giày sneaker phong cách trẻ trung, dễ phối đồ."}
        },
        {
            "sku": "fashion-6", "name": "Áo Khoác Blazer", "item_type": "fashion", "category": cats["Thời trang nữ"],
            "price": 650000, "stock": 20,
            "metadata": {"image_url": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea", "description": "Blazer phong cách Hàn Quốc thời thượng."}
        }
    ]

    for p_data in products:
        p, created = ProductCatalog.objects.update_or_create(
            sku=p_data["sku"],
            defaults={
                "name": p_data["name"],
                "item_type": p_data["item_type"],
                "category": p_data["category"],
                "price": p_data["price"],
                "stock": p_data["stock"],
                "metadata": p_data["metadata"]
            }
        )
        if created:
            print(f"Created product: {p.name}")
        else:
            print(f"Updated product: {p.name}")

if __name__ == "__main__":
    seed_products()
    print("Seeding complete!")

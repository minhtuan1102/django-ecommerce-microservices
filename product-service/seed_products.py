import os
import django
import random

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'product_service.settings')
django.setup()

from app.models import ProductCatalog, Category

def seed_products():
    categories_data = [
        "Sách văn học", "Sách kinh tế", "Sách kỹ năng", "Sách thiếu nhi",
        "Thời trang nam", "Thời trang nữ", "Phụ kiện",
        "Điện thoại", "Laptop", "Đồ gia dụng", "Thể thao", "Làm đẹp", "Thực phẩm"
    ]
    
    cats = {}
    for name in categories_data:
        cat, _ = Category.objects.get_or_create(name=name)
        cats[name] = cat

    # Generate 120 products programmatically
    products = []
    
    # 1. Books
    book_titles = ["Hạt giống tâm hồn", "Những kẻ xuất chúng", "Sapiens: Lược sử loài người", "Tội ác và hình phạt", "Hoàng tử bé", "Dạy con làm giàu", "Không gia đình", "Chiến tranh và hòa bình", "Hai số phận", "Cuốn theo chiều gió"]
    for i in range(25):
        cat_name = random.choice(["Sách văn học", "Sách kinh tế", "Sách kỹ năng", "Sách thiếu nhi"])
        products.append({
            "sku": f"book-{100+i}", "name": f"{random.choice(book_titles)} - Tập {i+1}", "item_type": "book", "category": cats[cat_name],
            "price": random.randint(50, 300) * 1000, "stock": random.randint(10, 200),
            "metadata": {"image_url": f"https://picsum.photos/seed/book{i}/400/400", "description": "Sách hay nên đọc."}
        })
    
    # 2. Fashion
    fashion_items = ["Áo thun", "Áo sơ mi", "Quần jean", "Quần âu", "Váy dạ hội", "Áo khoác mùa đông", "Mũ lưỡi trai", "Giày thể thao", "Kính râm", "Túi xách", "Đồng hồ"]
    for i in range(30):
        cat_name = random.choice(["Thời trang nam", "Thời trang nữ", "Phụ kiện"])
        products.append({
            "sku": f"fash-{100+i}", "name": f"{random.choice(fashion_items)} {['Cao cấp', 'Vintage', 'Thể thao', 'Hàn Quốc'][i%4]}", "item_type": "fashion", "category": cats[cat_name],
            "price": random.randint(100, 1500) * 1000, "stock": random.randint(5, 100),
            "metadata": {"image_url": f"https://picsum.photos/seed/fash{i}/400/400", "description": "Chất liệu thoáng mát, bền đẹp."}
        })

    # 3. Electronics
    tech_items = ["Điện thoại iPhone", "Điện thoại Samsung", "Laptop Dell", "Laptop Asus", "Tai nghe Bluetooth", "Sạc dự phòng", "Chuột không dây", "Bàn phím cơ"]
    for i in range(25):
        cat_name = random.choice(["Điện thoại", "Laptop", "Phụ kiện"])
        products.append({
            "sku": f"tech-{100+i}", "name": f"{random.choice(tech_items)} Pro {i+1}", "item_type": "electronics", "category": cats[cat_name],
            "price": random.randint(500, 30000) * 1000, "stock": random.randint(5, 50),
            "metadata": {"image_url": f"https://picsum.photos/seed/tech{i}/400/400", "description": "Hàng chính hãng, bảo hành 12 tháng."}
        })

    # 4. Others
    other_items = ["Máy xay sinh tố", "Nồi chiên không dầu", "Tạ đơn", "Thảm yoga", "Kem dưỡng da", "Son môi", "Nước hoa", "Bánh quy", "Trà xanh"]
    for i in range(25):
        cat_name = random.choice(["Đồ gia dụng", "Thể thao", "Làm đẹp", "Thực phẩm"])
        products.append({
            "sku": f"oth-{100+i}", "name": f"{random.choice(other_items)} {['Chính hãng', 'Nhập khẩu', 'Organic'][i%3]}", "item_type": "other", "category": cats[cat_name],
            "price": random.randint(50, 5000) * 1000, "stock": random.randint(10, 150),
            "metadata": {"image_url": f"https://picsum.photos/seed/oth{i}/400/400", "description": "Sản phẩm chất lượng cao."}
        })

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

if __name__ == "__main__":
    seed_products()
    print("Seeding complete! 105 products generated.")

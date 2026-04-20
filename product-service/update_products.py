import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'product_service.settings')
django.setup()

from app.models import ProductCatalog

updates = {
    'fashion-1': {
        'image': 'https://images.unsplash.com/photo-1515347619152-3a3399ab7e31',
        'desc': 'Áo thun thời trang, chất liệu cotton thoáng mát, thấm hút mồ hôi tốt. Kiểu dáng năng động, phù hợp mặc đi làm, đi chơi.'
    },
    'fashion-2': {
        'image': 'https://images.unsplash.com/photo-1539008835657-9e8e9680c956',
        'desc': 'Đầm váy dạo phố thiết kế thanh lịch, màu sắc tươi sáng. Chất liệu vải mềm mịn, tôn dáng người mặc.'
    },
    'fashion-3': {
        'image': 'https://images.unsplash.com/photo-1542272604-787c3835535d',
        'desc': 'Quần jean jean nam/nữ, form dáng chuẩn mực, trẻ trung. Độ bền cao, dễ dàng phối đồ.'
    },
    'book-1': {
        'image': 'https://images.unsplash.com/photo-1544947950-fa07a98d237f',
        'desc': 'Cuốn sách nổi bật nhất năm mang lại kiến thức bổ ích. Tác phẩm truyền cảm hứng với văn phong lôi cuốn.'
    },
    'book-2': {
        'image': 'https://images.unsplash.com/photo-1589829085413-56de8ae18c73',
        'desc': 'Tuyển tập những câu chuyện cuộc sống đầy ý nghĩa. Giúp bạn tìm thấy sự bình yên và động lực mỗi ngày.'
    },
    'book-3': {
        'image': 'https://images.unsplash.com/photo-1532012197267-da84d127e765',
        'desc': 'Sách phân tích sắc sảo. Lựa chọn tuyệt vời cho mọi độc giả.'
    }
}

for product in ProductCatalog.objects.all():
    update_data = updates.get(product.sku)
    if update_data:
        product.metadata['image_url'] = update_data['image']
        product.metadata['description'] = update_data['desc']
    else:
        # Fallback updates for others
        product.metadata['image_url'] = f'https://picsum.photos/seed/{product.id + 100}/600/800'
        product.metadata['description'] = f'Sản phẩm tuyệt vời mang tên {product.name}. Chất lượng đảm bảo từ hệ thống P-Shop.'
    
    product.save()

print("Products updated successfully!")

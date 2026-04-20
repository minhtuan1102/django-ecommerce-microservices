"""
Script để populate Knowledge Base với dữ liệu thực từ Book Service
và các thông tin chính sách của cửa hàng
"""
import os
import sys
import json
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_service.settings')

import django
django.setup()

from app.rag.retriever import KnowledgeRetriever


# Book Service URL (change if running outside Docker)
BOOK_SERVICE_URL = os.environ.get('BOOK_SERVICE_URL', 'http://localhost:8002')


def fetch_books_from_service():
    """Fetch books từ Book Service"""
    try:
        response = requests.get(f"{BOOK_SERVICE_URL}/books/", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Could not fetch from Book Service: {e}")
    return []


def get_sample_books():
    """Sample books nếu không kết nối được Book Service"""
    return [
        {"id": 1, "title": "Đắc Nhân Tâm", "author": "Dale Carnegie", "price": 89000, 
         "description": "Cuốn sách kinh điển về nghệ thuật giao tiếp và thu phục lòng người."},
        {"id": 2, "title": "Nhà Giả Kim", "author": "Paulo Coelho", "price": 79000,
         "description": "Câu chuyện về chàng chăn cừu Santiago và hành trình theo đuổi giấc mơ."},
        {"id": 3, "title": "Atomic Habits", "author": "James Clear", "price": 169000,
         "description": "Thay đổi tí hon, hiệu quả bất ngờ. Cách xây dựng thói quen tốt."},
        {"id": 4, "title": "Sapiens: Lược Sử Loài Người", "author": "Yuval Noah Harari", "price": 189000,
         "description": "Lịch sử phát triển của loài người từ thời tiền sử đến hiện đại."},
        {"id": 5, "title": "Hoàng Tử Bé", "author": "Antoine de Saint-Exupéry", "price": 65000,
         "description": "Truyện ngụ ngôn nổi tiếng về tình bạn, tình yêu và cuộc sống."},
        {"id": 6, "title": "Người Giàu Có Nhất Thành Babylon", "author": "George S. Clason", "price": 69000,
         "description": "Bí mật làm giàu qua câu chuyện về thành Babylon cổ đại."},
        {"id": 7, "title": "Clean Code", "author": "Robert C. Martin", "price": 350000,
         "description": "Hướng dẫn viết code sạch và dễ bảo trì cho lập trình viên."},
        {"id": 8, "title": "Tư Duy Nhanh Và Chậm", "author": "Daniel Kahneman", "price": 199000,
         "description": "Hai hệ thống tư duy ảnh hưởng đến quyết định của con người."},
        {"id": 9, "title": "7 Thói Quen Hiệu Quả", "author": "Stephen R. Covey", "price": 145000,
         "description": "7 thói quen của người thành đạt để phát triển bản thân."},
        {"id": 10, "title": "Cách Nghĩ Để Thành Công", "author": "Napoleon Hill", "price": 95000,
         "description": "Bí quyết tư duy tích cực để đạt được mục tiêu trong cuộc sống."},
        {"id": 11, "title": "Python Crash Course", "author": "Eric Matthes", "price": 420000,
         "description": "Học Python từ cơ bản đến nâng cao với các dự án thực tế."},
        {"id": 12, "title": "Khởi Nghiệp Tinh Gọn", "author": "Eric Ries", "price": 135000,
         "description": "Phương pháp khởi nghiệp hiệu quả với chi phí tối thiểu."},
        {"id": 13, "title": "Đời Ngắn Đừng Ngủ Dài", "author": "Robin Sharma", "price": 78000,
         "description": "Cách sống trọn vẹn từng giây phút để không hối tiếc."},
        {"id": 14, "title": "Sức Mạnh Của Thói Quen", "author": "Charles Duhigg", "price": 125000,
         "description": "Hiểu về thói quen và cách thay đổi để thành công."},
        {"id": 15, "title": "Chiến Lược Đại Dương Xanh", "author": "W. Chan Kim", "price": 185000,
         "description": "Tạo không gian thị trường mới không có đối thủ cạnh tranh."},
    ]


def get_policies():
    """Danh sách chính sách của cửa hàng"""
    return [
        {
            "id": "policy_return",
            "title": "Chính sách đổi trả",
            "content": """Chính sách đổi trả hàng của BookStore:
            
• Thời gian đổi trả: 7 ngày kể từ ngày nhận hàng
• Điều kiện: Sản phẩm còn nguyên seal, chưa qua sử dụng
• Sách bị lỗi in ấn: Đổi mới miễn phí
• Sách đã mở seal: Không áp dụng đổi trả (trừ lỗi sản xuất)
• Phí đổi trả: Miễn phí nếu lỗi từ shop, khách chịu phí nếu đổi ý

Để đổi trả, vui lòng liên hệ hotline 1900-1234 hoặc email support@bookstore.vn""",
            "category": "policies"
        },
        {
            "id": "policy_shipping",
            "title": "Chính sách vận chuyển",
            "content": """Chính sách giao hàng của BookStore:

• Miễn phí ship cho đơn từ 300.000đ
• Phí ship nội thành: 15.000đ - 25.000đ
• Phí ship ngoại thành: 25.000đ - 40.000đ
• Thời gian giao hàng:
  - Nội thành TP.HCM, Hà Nội: 1-2 ngày làm việc
  - Các tỉnh thành khác: 3-5 ngày làm việc
  - Vùng sâu vùng xa: 5-7 ngày làm việc
• Đối tác vận chuyển: GHN, GHTK, J&T Express
• Có thể theo dõi đơn hàng qua mã vận đơn""",
            "category": "policies"
        },
        {
            "id": "policy_payment",
            "title": "Phương thức thanh toán",
            "content": """Các phương thức thanh toán tại BookStore:

1. COD (Ship COD): Thanh toán khi nhận hàng
2. Chuyển khoản ngân hàng:
   - Vietcombank: 1234567890 - BOOKSTORE CO LTD
   - Techcombank: 0987654321 - BOOKSTORE CO LTD
3. Ví điện tử: MoMo, ZaloPay, VNPay
4. Thẻ tín dụng/ghi nợ: Visa, Mastercard, JCB
5. Trả góp 0% qua thẻ tín dụng (đơn từ 3 triệu)

Lưu ý: Giữ hóa đơn để đổi trả khi cần thiết.""",
            "category": "policies"
        },
        {
            "id": "policy_warranty",
            "title": "Chính sách bảo hành",
            "content": """Chính sách bảo hành sách tại BookStore:

• Sách lỗi in ấn: Đổi mới 100% miễn phí
• Sách bị hư hại khi vận chuyển: Đổi mới miễn phí
• Sách thiếu trang: Đổi mới miễn phí
• Thời gian bảo hành: 30 ngày kể từ ngày mua

Không bảo hành các trường hợp:
• Sách bị rách, bẩn do người dùng
• Sách đã ghi chú, highlight
• Sách bị ướt, mốc do bảo quản không đúng cách""",
            "category": "policies"
        }
    ]


def get_faqs():
    """Danh sách câu hỏi thường gặp"""
    return [
        {
            "id": "faq_order_tracking",
            "question": "Làm sao để theo dõi đơn hàng?",
            "answer": """Để theo dõi đơn hàng tại BookStore:

1. Đăng nhập tài khoản tại website
2. Vào mục "Đơn hàng của tôi"
3. Chọn đơn hàng cần theo dõi
4. Xem trạng thái và mã vận đơn

Hoặc bạn có thể:
• Gọi hotline: 1900-1234
• Email: support@bookstore.vn
• Chat với AI Chatbot này để kiểm tra nhanh""",
            "category": "faqs"
        },
        {
            "id": "faq_cancel_order",
            "question": "Làm sao để hủy đơn hàng?",
            "answer": """Để hủy đơn hàng:

1. Nếu đơn chưa được xử lý:
   - Đăng nhập → Đơn hàng của tôi → Chọn đơn → Hủy đơn
   
2. Nếu đơn đã được giao cho vận chuyển:
   - Liên hệ hotline 1900-1234 để được hỗ trợ
   - Có thể từ chối nhận hàng khi shipper giao
   
Lưu ý: Đơn đã thanh toán online sẽ được hoàn tiền trong 3-5 ngày làm việc.""",
            "category": "faqs"
        },
        {
            "id": "faq_membership",
            "question": "Làm sao để trở thành thành viên?",
            "answer": """Để đăng ký thành viên BookStore:

1. Truy cập website và nhấn "Đăng ký"
2. Điền thông tin: họ tên, email, số điện thoại
3. Xác nhận qua email

Quyền lợi thành viên:
• Tích điểm mỗi đơn hàng (1.000đ = 1 điểm)
• Đổi điểm lấy voucher giảm giá
• Ưu đãi sinh nhật
• Flash sale dành riêng cho thành viên
• Miễn phí ship cho thành viên VIP""",
            "category": "faqs"
        },
        {
            "id": "faq_discount",
            "question": "Làm sao để nhận mã giảm giá?",
            "answer": """Cách nhận mã giảm giá tại BookStore:

1. Đăng ký thành viên mới: Giảm 10% đơn đầu tiên
2. Theo dõi fanpage Facebook, Instagram
3. Đăng ký nhận newsletter qua email
4. Tham gia các event, minigame
5. Sinh nhật thành viên: Voucher 20%
6. Flash sale hàng tuần

Lưu ý: Mỗi đơn chỉ áp dụng 1 mã giảm giá.""",
            "category": "faqs"
        },
        {
            "id": "faq_bulk_order",
            "question": "Mua số lượng lớn có được giảm giá?",
            "answer": """Chính sách mua số lượng lớn:

• Từ 10 cuốn cùng loại: Giảm 5%
• Từ 50 cuốn cùng loại: Giảm 10%
• Từ 100 cuốn: Giảm 15% + Miễn phí vận chuyển

Đối với đơn hàng doanh nghiệp, trường học:
• Liên hệ trực tiếp: sales@bookstore.vn
• Báo giá trong 24h
• Hỗ trợ xuất hóa đơn VAT""",
            "category": "faqs"
        }
    ]


def populate_knowledge_base():
    """Populate knowledge base với tất cả dữ liệu"""
    print("="*60)
    print("POPULATE KNOWLEDGE BASE FOR AI CHATBOT")
    print("="*60)
    
    retriever = KnowledgeRetriever()
    
    # 1. Fetch và add books
    print("\n[1/3] Adding books to knowledge base...")
    books = fetch_books_from_service()
    if not books:
        print("  Using sample books (could not connect to Book Service)")
        books = get_sample_books()
    
    book_documents = []
    book_metadatas = []
    book_ids = []
    
    for book in books:
        # Create rich document content
        doc_content = f"""Sách: {book.get('title', 'N/A')}
Tác giả: {book.get('author', 'N/A')}
Giá: {book.get('price', 0):,}đ
Mô tả: {book.get('description', 'Sách hay, đáng đọc.')}
Còn hàng: {'Có' if book.get('stock', 0) > 0 else 'Hết hàng'}"""
        
        book_documents.append(doc_content)
        book_metadatas.append({
            'category': 'products',
            'type': 'book',
            'id': book.get('id'),
            'title': book.get('title', ''),
            'author': book.get('author', ''),
            'price': book.get('price', 0)
        })
        book_ids.append(f"book_{book.get('id')}")
    
    if book_documents:
        retriever.add_documents(book_documents, book_metadatas, book_ids)
        print(f"  ✓ Added {len(book_documents)} books")
    
    # 2. Add policies
    print("\n[2/3] Adding policies to knowledge base...")
    policies = get_policies()
    
    policy_documents = []
    policy_metadatas = []
    policy_ids = []
    
    for policy in policies:
        policy_documents.append(f"{policy['title']}\n\n{policy['content']}")
        policy_metadatas.append({
            'category': 'policies',
            'type': policy['id'],
            'title': policy['title']
        })
        policy_ids.append(policy['id'])
    
    if policy_documents:
        retriever.add_documents(policy_documents, policy_metadatas, policy_ids)
        print(f"  ✓ Added {len(policy_documents)} policies")
    
    # 3. Add FAQs
    print("\n[3/3] Adding FAQs to knowledge base...")
    faqs = get_faqs()
    
    faq_documents = []
    faq_metadatas = []
    faq_ids = []
    
    for faq in faqs:
        faq_documents.append(f"Câu hỏi: {faq['question']}\n\nTrả lời: {faq['answer']}")
        faq_metadatas.append({
            'category': 'faqs',
            'type': faq['id'],
            'question': faq['question']
        })
        faq_ids.append(faq['id'])
    
    if faq_documents:
        retriever.add_documents(faq_documents, faq_metadatas, faq_ids)
        print(f"  ✓ Added {len(faq_documents)} FAQs")
    
    # Print stats
    print("\n" + "="*60)
    stats = retriever.get_collection_stats()
    print("KNOWLEDGE BASE STATS:")
    print(f"  Collection: {stats.get('collection_name')}")
    print(f"  Total documents: {stats.get('document_count')}")
    print(f"  Storage path: {stats.get('chroma_path')}")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    populate_knowledge_base()

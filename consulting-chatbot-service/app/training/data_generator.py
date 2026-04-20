"""
Data Generator cho Training
Tạo synthetic training data từ templates và knowledge base
"""
import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ConversationSample:
    """Single conversation sample"""
    query: str
    response: str
    intent: str
    context: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class ChatbotDataGenerator:
    """
    Generate training data cho chatbot từ templates
    
    Strategies:
    1. Template-based generation với variations
    2. Augmentation (synonym replacement, paraphrase)
    3. Knowledge base extraction
    """
    
    # Intent templates với nhiều variations
    TEMPLATES = {
        "greeting": {
            "queries": [
                "Xin chào",
                "Chào bạn",
                "Hello",
                "Hi",
                "Chào shop",
                "Alo",
                "Chào buổi sáng",
                "Chào buổi chiều",
                "Chào buổi tối",
                "Hey",
                "Xin chào shop",
                "Chào anh/chị",
                "Mình chào bạn"
            ],
            "responses": [
                "Xin chào! 👋 Tôi là trợ lý AI của BookStore. Tôi có thể giúp bạn tìm sách, kiểm tra đơn hàng, hoặc giải đáp thắc mắc. Bạn cần hỗ trợ gì?",
                "Chào bạn! 😊 Rất vui được hỗ trợ bạn hôm nay. Bạn đang tìm kiếm sách gì, hay cần hỗ trợ về đơn hàng?",
                "Hello! Tôi là AI assistant của BookStore. Tôi sẵn sàng giúp bạn tìm sách phù hợp hoặc giải đáp mọi thắc mắc. Bạn cần gì?",
                "Xin chào quý khách! Cảm ơn bạn đã ghé thăm BookStore. Tôi có thể hỗ trợ gì cho bạn?",
                "Chào bạn! 📚 BookStore luôn sẵn sàng phục vụ. Bạn muốn tìm sách hay cần hỗ trợ đơn hàng?"
            ]
        },
        
        "product_query": {
            "patterns": [
                # Tìm kiếm chung
                ("Tôi muốn tìm sách về {topic}", "Dạ, tôi tìm thấy một số sách về {topic} cho bạn:\n\n{product_list}\n\nBạn quan tâm đến cuốn nào?"),
                ("Có sách {topic} không?", "Có ạ! Chúng tôi có nhiều sách {topic}:\n\n{product_list}\n\nBạn muốn xem chi tiết cuốn nào?"),
                ("Tìm sách {topic}", "Đây là các sách {topic} được đánh giá cao:\n\n{product_list}"),
                ("Cho tôi xem sách {topic}", "Dạ vâng, đây là sách {topic} bên shop:\n\n{product_list}"),
                
                # Hỏi giá
                ("Giá sách {book_name} bao nhiêu?", "Sách {book_name} có giá {price}đ. Bạn muốn đặt mua không ạ?"),
                ("Cuốn {book_name} giá bao nhiêu?", "{book_name} hiện đang có giá {price}đ tại BookStore."),
                ("{book_name} bao nhiêu tiền?", "Giá của {book_name} là {price}đ. Bạn có thể xem chi tiết tại: {link}"),
                
                # Hỏi về tác giả
                ("Có sách của tác giả {author} không?", "Có ạ! Các sách của {author} tại shop:\n\n{product_list}"),
                ("Sách {author} viết", "Đây là các tác phẩm của {author}:\n\n{product_list}"),
                ("Tìm sách tác giả {author}", "Các sách của tác giả {author}:\n\n{product_list}"),
                
                # Hỏi chi tiết
                ("Cho tôi biết thêm về sách {book_name}", "📚 **{book_name}**\n\n{description}\n\n💰 Giá: {price}đ\n📖 Thể loại: {category}\n✍️ Tác giả: {author}"),
                ("Nội dung sách {book_name} là gì?", "**{book_name}** - {description}\n\nBạn có muốn đặt mua không?"),
                
                # Best seller
                ("Sách bán chạy nhất là gì?", "📈 Top sách bán chạy:\n\n{product_list}\n\nBạn quan tâm đến cuốn nào?"),
                ("Cho tôi xem sách best seller", "Đây là những cuốn sách được yêu thích nhất:\n\n{product_list}"),
                ("Sách nào đang hot?", "🔥 Sách HOT hiện tại:\n\n{product_list}")
            ],
            "topics": [
                "kinh doanh", "self-help", "kỹ năng sống", "marketing", 
                "tâm lý học", "tiểu thuyết", "khoa học", "lịch sử",
                "văn học", "truyện ngắn", "triết học", "kinh tế",
                "lập trình", "công nghệ", "ngoại ngữ", "thiếu nhi",
                "sức khỏe", "nấu ăn", "du lịch", "nghệ thuật"
            ],
            "books": [
                {"name": "Đắc Nhân Tâm", "author": "Dale Carnegie", "price": "89000", "category": "self-help"},
                {"name": "Nhà Giả Kim", "author": "Paulo Coelho", "price": "79000", "category": "tiểu thuyết"},
                {"name": "Sapiens", "author": "Yuval Noah Harari", "price": "189000", "category": "lịch sử"},
                {"name": "Atomic Habits", "author": "James Clear", "price": "169000", "category": "self-help"},
                {"name": "Clean Code", "author": "Robert C. Martin", "price": "350000", "category": "lập trình"},
                {"name": "Tư Duy Nhanh Và Chậm", "author": "Daniel Kahneman", "price": "199000", "category": "tâm lý học"},
                {"name": "Người Giàu Có Nhất Thành Babylon", "author": "George S. Clason", "price": "69000", "category": "tài chính"},
                {"name": "Hoàng Tử Bé", "author": "Antoine de Saint-Exupéry", "price": "55000", "category": "văn học"}
            ]
        },
        
        "policy_query": {
            "qa_pairs": [
                # Đổi trả
                ("Chính sách đổi trả như thế nào?", "📋 **Chính sách đổi trả BookStore:**\n\n✅ Đổi trả trong vòng **7 ngày** kể từ khi nhận hàng\n✅ Sản phẩm còn nguyên seal, chưa qua sử dụng\n✅ Có hóa đơn mua hàng\n\n📞 Hotline: 1900-xxxx\n📧 Email: support@bookstore.vn"),
                ("Tôi muốn đổi sách", "Bạn có thể đổi sách trong vòng 7 ngày nếu:\n- Sách còn nguyên seal\n- Có hóa đơn mua hàng\n\nVui lòng liên hệ hotline 1900-xxxx để được hỗ trợ!"),
                ("Trả hàng được không?", "Được ạ! BookStore hỗ trợ trả hàng trong 7 ngày. Sách cần còn nguyên, chưa sử dụng. Bạn cần hỗ trợ thêm không?"),
                
                # Vận chuyển
                ("Phí ship bao nhiêu?", "🚚 **Phí vận chuyển:**\n\n- Nội thành: 15,000đ - 25,000đ\n- Ngoại thành: 25,000đ - 35,000đ\n- Miễn phí ship cho đơn từ 300,000đ\n\nThời gian: 2-5 ngày làm việc"),
                ("Giao hàng mất bao lâu?", "⏰ Thời gian giao hàng:\n- Nội thành HCM/HN: 1-2 ngày\n- Các tỉnh khác: 3-5 ngày\n\nBạn có thể theo dõi đơn hàng qua link trong email xác nhận."),
                ("Có giao hàng COD không?", "Có ạ! BookStore hỗ trợ thanh toán COD (nhận hàng trả tiền) cho tất cả đơn hàng."),
                
                # Thanh toán
                ("Có những hình thức thanh toán nào?", "💳 **Phương thức thanh toán:**\n\n1. COD (thanh toán khi nhận hàng)\n2. Chuyển khoản ngân hàng\n3. Ví MoMo, ZaloPay, VNPay\n4. Thẻ tín dụng/ghi nợ\n\nBạn chọn hình thức nào?"),
                ("Thanh toán bằng Momo được không?", "Được ạ! BookStore hỗ trợ thanh toán qua MoMo. Bạn chọn MoMo khi checkout nhé!"),
                
                # Khuyến mãi
                ("Có mã giảm giá không?", "🎁 Hiện tại BookStore có các ưu đãi:\n\n- Giảm 10% cho đơn đầu tiên: **WELCOME10**\n- Miễn phí ship đơn từ 300K\n- Tích điểm đổi voucher\n\nBạn muốn áp dụng mã nào?"),
                ("Khuyến mãi gì đang có?", "🔥 **Ưu đãi tháng này:**\n\n- Flash sale: Giảm đến 50%\n- Combo sách: Mua 2 giảm 15%\n- Member: Tích điểm x2\n\nXem chi tiết tại: bookstore.vn/khuyen-mai"),
                
                # Thành viên
                ("Làm sao để đăng ký thành viên?", "Đăng ký thành viên BookStore rất đơn giản:\n\n1. Truy cập bookstore.vn\n2. Click 'Đăng ký'\n3. Nhập email và tạo mật khẩu\n\n🎁 Ưu đãi member: Tích điểm, voucher sinh nhật, sale riêng!"),
                ("Thành viên có ưu đãi gì?", "👑 **Ưu đãi thành viên:**\n\n- Tích điểm 1% giá trị đơn hàng\n- Voucher sinh nhật 50K\n- Ưu tiên mua sách limited\n- Early access sale\n- Freeship từ 200K")
            ]
        },
        
        "order_support": {
            "patterns": [
                # Kiểm tra đơn hàng
                ("Kiểm tra đơn hàng {order_id}", "📦 **Đơn hàng #{order_id}:**\n\nTrạng thái: {status}\nDự kiến giao: {delivery_date}\n\nBạn có thể theo dõi chi tiết tại: bookstore.vn/don-hang/{order_id}"),
                ("Đơn hàng của tôi đến đâu rồi?", "Vui lòng cho tôi mã đơn hàng để kiểm tra nhé! Mã đơn có dạng: BK-XXXXXX"),
                ("Tôi muốn xem đơn hàng", "Bạn vui lòng cung cấp mã đơn hàng hoặc số điện thoại đặt hàng để tôi kiểm tra nhé!"),
                
                # Hủy đơn
                ("Tôi muốn hủy đơn hàng", "Bạn có thể hủy đơn nếu đơn chưa được giao cho đơn vị vận chuyển. Vui lòng cho tôi mã đơn hàng để kiểm tra!"),
                ("Hủy đơn {order_id}", "Tôi đã ghi nhận yêu cầu hủy đơn #{order_id}. Bạn sẽ nhận được xác nhận qua email trong 24h. Cần hỗ trợ thêm không?"),
                ("Làm sao hủy đơn?", "Để hủy đơn, bạn có thể:\n1. Vào Tài khoản > Đơn hàng > Hủy\n2. Hoặc liên hệ hotline 1900-xxxx\n\nLưu ý: Chỉ hủy được khi đơn chưa giao shipper."),
                
                # Thay đổi địa chỉ
                ("Đổi địa chỉ giao hàng", "Bạn muốn thay đổi địa chỉ giao hàng? Vui lòng cung cấp:\n1. Mã đơn hàng\n2. Địa chỉ mới\n\nTôi sẽ kiểm tra và cập nhật cho bạn!"),
                ("Tôi nhập sai địa chỉ", "Đừng lo! Nếu đơn chưa được giao shipper, tôi có thể giúp bạn cập nhật. Cho tôi mã đơn và địa chỉ mới nhé!"),
                
                # Vấn đề giao hàng
                ("Đơn hàng giao chậm", "Tôi xin lỗi về sự bất tiện này! Cho tôi mã đơn hàng để kiểm tra tình trạng và liên hệ đơn vị vận chuyển nhé."),
                ("Chưa nhận được hàng", "Xin lỗi bạn! Vui lòng cung cấp mã đơn hàng, tôi sẽ kiểm tra ngay và hỗ trợ bạn.")
            ],
            "order_statuses": [
                "Đang xử lý",
                "Đã xác nhận",
                "Đang đóng gói",
                "Đã giao shipper",
                "Đang giao hàng",
                "Đã giao thành công"
            ]
        },
        
        "recommendation": {
            "patterns": [
                # Gợi ý chung
                ("Gợi ý sách cho tôi", "📚 Để gợi ý sách phù hợp, bạn có thể cho tôi biết:\n\n1. Bạn thích thể loại gì? (self-help, tiểu thuyết, kinh doanh...)\n2. Sách gần đây bạn đọc và thích?\n3. Mục đích đọc? (học hỏi, giải trí, phát triển bản thân...)"),
                ("Tôi nên đọc sách gì?", "Tùy thuộc vào mục đích, tôi gợi ý:\n\n📈 **Phát triển bản thân:** Atomic Habits, Đắc Nhân Tâm\n📖 **Tiểu thuyết hay:** Nhà Giả Kim, 1984\n💼 **Kinh doanh:** Zero to One, Good to Great\n\nBạn quan tâm lĩnh vực nào?"),
                
                # Theo sở thích
                ("Tôi thích {genre}", "Tuyệt vời! Với sách {genre}, tôi gợi ý:\n\n{recommendations}\n\nBạn muốn xem chi tiết cuốn nào?"),
                ("Gợi ý sách {genre}", "Đây là top sách {genre} được đánh giá cao:\n\n{recommendations}"),
                
                # Theo mục đích
                ("Sách để học {skill}", "Để học {skill}, bạn nên đọc:\n\n{recommendations}\n\nCuốn nào phù hợp với level của bạn?"),
                ("Sách cho người mới bắt đầu {topic}", "Cho người mới với {topic}, tôi recommend:\n\n{recommendations}\n\nBắt đầu từ cuốn đầu tiên nhé!"),
                
                # Tương tự
                ("Sách giống {book_name}", "Nếu bạn thích {book_name}, bạn sẽ thích:\n\n{recommendations}\n\nCùng tác giả hoặc thể loại tương tự!"),
                ("Đọc gì sau {book_name}?", "Sau {book_name}, bạn có thể đọc:\n\n{recommendations}")
            ],
            "genres": [
                "self-help", "kinh doanh", "tiểu thuyết", "khoa học",
                "tâm lý", "lịch sử", "triết học", "lập trình"
            ]
        },
        
        "general_chat": {
            "qa_pairs": [
                # Cảm ơn
                ("Cảm ơn", "Không có gì ạ! 😊 Rất vui được hỗ trợ bạn. Nếu cần gì thêm, cứ hỏi nhé!"),
                ("Cảm ơn bạn", "Dạ không có chi! Chúc bạn có trải nghiệm mua sắm tuyệt vời! 📚"),
                ("Thanks", "You're welcome! 😊 Có gì cần hỗ trợ thêm không ạ?"),
                
                # Tạm biệt
                ("Tạm biệt", "Tạm biệt bạn! 👋 Hẹn gặp lại. Chúc bạn ngày tốt lành!"),
                ("Bye", "Bye bye! Cảm ơn bạn đã ghé thăm BookStore! 📚"),
                
                # Hài hước
                ("Bạn là ai?", "Tôi là AI assistant của BookStore! 🤖 Tôi được train để giúp bạn tìm sách, kiểm tra đơn hàng, và giải đáp thắc mắc. Rất vui được gặp bạn!"),
                ("Bạn có thông minh không?", "Tôi cố gắng hết sức để hỗ trợ bạn tốt nhất! 😄 Dù là AI, tôi luôn học hỏi để trả lời chính xác hơn. Có gì tôi giúp được không?"),
                
                # Không hiểu
                ("...", "Tôi không chắc hiểu ý bạn. Bạn có thể nói rõ hơn được không? Tôi có thể giúp:\n- Tìm sách\n- Kiểm tra đơn hàng\n- Giải đáp chính sách"),
                ("???", "Có vẻ như tôi chưa hiểu câu hỏi của bạn. Bạn cần hỗ trợ về sách, đơn hàng, hay chính sách?")
            ]
        }
    }
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path
        self.samples: List[ConversationSample] = []
        
    def generate_greeting_samples(self, n_per_pair: int = 3) -> List[ConversationSample]:
        """Generate greeting samples với variations"""
        samples = []
        greeting_data = self.TEMPLATES["greeting"]
        
        for query in greeting_data["queries"]:
            for _ in range(n_per_pair):
                response = random.choice(greeting_data["responses"])
                samples.append(ConversationSample(
                    query=query,
                    response=response,
                    intent="greeting"
                ))
        
        return samples
    
    def generate_product_samples(self, n_samples: int = 200) -> List[ConversationSample]:
        """Generate product query samples"""
        samples = []
        product_data = self.TEMPLATES["product_query"]
        
        for _ in range(n_samples):
            pattern, response_template = random.choice(product_data["patterns"])
            
            # Fill in placeholders
            topic = random.choice(product_data["topics"])
            book = random.choice(product_data["books"])
            
            query = pattern.format(
                topic=topic,
                book_name=book["name"],
                author=book["author"]
            )
            
            # Create product list
            selected_books = random.sample(product_data["books"], min(3, len(product_data["books"])))
            product_list = "\n".join([
                f"📖 **{b['name']}** - {b['author']} - {b['price']}đ"
                for b in selected_books
            ])
            
            response = response_template.format(
                topic=topic,
                book_name=book["name"],
                author=book["author"],
                price=book["price"],
                category=book["category"],
                description=f"Cuốn sách {book['category']} hấp dẫn của tác giả {book['author']}",
                product_list=product_list,
                link=f"bookstore.vn/sach/{book['name'].lower().replace(' ', '-')}"
            )
            
            samples.append(ConversationSample(
                query=query,
                response=response,
                intent="product_query",
                metadata={"topic": topic, "book": book["name"]}
            ))
        
        return samples
    
    def generate_policy_samples(self, n_per_pair: int = 3) -> List[ConversationSample]:
        """Generate policy query samples"""
        samples = []
        policy_data = self.TEMPLATES["policy_query"]
        
        for query, response in policy_data["qa_pairs"]:
            for _ in range(n_per_pair):
                # Add slight variations
                query_var = self._add_variation(query)
                samples.append(ConversationSample(
                    query=query_var,
                    response=response,
                    intent="policy_query"
                ))
        
        return samples
    
    def generate_order_samples(self, n_samples: int = 100) -> List[ConversationSample]:
        """Generate order support samples"""
        samples = []
        order_data = self.TEMPLATES["order_support"]
        
        for _ in range(n_samples):
            pattern, response_template = random.choice(order_data["patterns"])
            
            # Generate random order ID
            order_id = f"BK-{random.randint(100000, 999999)}"
            status = random.choice(order_data["order_statuses"])
            
            query = pattern.format(order_id=order_id)
            response = response_template.format(
                order_id=order_id,
                status=status,
                delivery_date="2-3 ngày tới"
            )
            
            samples.append(ConversationSample(
                query=query,
                response=response,
                intent="order_support",
                metadata={"order_id": order_id}
            ))
        
        return samples
    
    def generate_recommendation_samples(self, n_samples: int = 100) -> List[ConversationSample]:
        """Generate recommendation samples"""
        samples = []
        rec_data = self.TEMPLATES["recommendation"]
        product_data = self.TEMPLATES["product_query"]
        
        for _ in range(n_samples):
            pattern, response_template = random.choice(rec_data["patterns"])
            
            genre = random.choice(rec_data["genres"])
            book = random.choice(product_data["books"])
            
            # Create recommendations list
            recs = random.sample(product_data["books"], min(3, len(product_data["books"])))
            recommendations = "\n".join([
                f"📖 **{b['name']}** - {b['author']}"
                for b in recs
            ])
            
            query = pattern.format(
                genre=genre,
                skill=genre,
                topic=genre,
                book_name=book["name"]
            )
            
            response = response_template.format(
                genre=genre,
                skill=genre,
                topic=genre,
                book_name=book["name"],
                recommendations=recommendations
            )
            
            samples.append(ConversationSample(
                query=query,
                response=response,
                intent="recommendation",
                metadata={"genre": genre}
            ))
        
        return samples
    
    def generate_general_samples(self, n_per_pair: int = 3) -> List[ConversationSample]:
        """Generate general chat samples"""
        samples = []
        general_data = self.TEMPLATES["general_chat"]
        
        for query, response in general_data["qa_pairs"]:
            for _ in range(n_per_pair):
                query_var = self._add_variation(query)
                samples.append(ConversationSample(
                    query=query_var,
                    response=response,
                    intent="general_chat"
                ))
        
        return samples
    
    def _add_variation(self, text: str) -> str:
        """Add slight variations to text"""
        variations = [
            lambda t: t,  # Original
            lambda t: t + " ạ",  # Add politeness
            lambda t: t + " nhé",
            lambda t: t + " vậy",
            lambda t: t.lower(),
            lambda t: t + "?",
            lambda t: "Cho tôi hỏi " + t.lower(),
            lambda t: "Mình muốn hỏi " + t.lower(),
        ]
        return random.choice(variations)(text)
    
    def generate_all(
        self,
        greeting_n: int = 3,
        product_n: int = 200,
        policy_n: int = 3,
        order_n: int = 100,
        recommendation_n: int = 100,
        general_n: int = 3
    ) -> List[ConversationSample]:
        """Generate all samples"""
        all_samples = []
        
        all_samples.extend(self.generate_greeting_samples(greeting_n))
        all_samples.extend(self.generate_product_samples(product_n))
        all_samples.extend(self.generate_policy_samples(policy_n))
        all_samples.extend(self.generate_order_samples(order_n))
        all_samples.extend(self.generate_recommendation_samples(recommendation_n))
        all_samples.extend(self.generate_general_samples(general_n))
        
        # Shuffle
        random.shuffle(all_samples)
        
        self.samples = all_samples
        return all_samples
    
    def save(self, path: str):
        """Save samples to JSON"""
        data = [
            {
                "query": s.query,
                "response": s.response,
                "intent": s.intent,
                "context": s.context,
                "metadata": s.metadata
            }
            for s in self.samples
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(data)} samples to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ChatbotDataGenerator':
        """Load samples from JSON"""
        generator = cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        generator.samples = [
            ConversationSample(
                query=d["query"],
                response=d["response"],
                intent=d["intent"],
                context=d.get("context"),
                metadata=d.get("metadata", {})
            )
            for d in data
        ]
        
        print(f"Loaded {len(generator.samples)} samples from {path}")
        return generator
    
    def get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution of intents"""
        dist = {}
        for s in self.samples:
            dist[s.intent] = dist.get(s.intent, 0) + 1
        return dist


def generate_training_data(
    output_path: str,
    knowledge_base_path: Optional[str] = None
) -> ChatbotDataGenerator:
    """
    Convenient function to generate and save training data
    """
    generator = ChatbotDataGenerator(knowledge_base_path)
    generator.generate_all()
    generator.save(output_path)
    
    print("\nIntent distribution:")
    for intent, count in generator.get_intent_distribution().items():
        print(f"  {intent}: {count}")
    
    return generator

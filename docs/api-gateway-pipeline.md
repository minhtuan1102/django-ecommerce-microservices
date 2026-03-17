# Vai trò của API Gateway và Quy trình Nghiệp vụ trong Microservices

Tài liệu này giải thích cách API Gateway hoạt động trong hệ thống và luồng xử lý chi tiết (pipeline) từ khi người dùng tạo giỏ hàng cho đến khi hoàn tất đơn hàng.

## 1. Vai trò của API Gateway trong Hệ thống
Trong kiến trúc microservices này, API Gateway (nằm tại thư mục `api-gateway/`) đóng vai trò là **điểm truy cập duy nhất (Single Entry Point)** cho mọi client.

### Thể hiện của REST API:
- **Định tuyến (Routing):** Gateway nhận các request RESTful (GET, POST, PUT, DELETE) và chuyển tiếp đến các service tương ứng (book-service, cart-service, order-service...).
- **Xác thực JWT (JWT Authentication):** Middleware `JWTValidationMiddleware` kiểm tra token trong header `Authorization: Bearer <token>`.
- **Kiểm soát quyền truy cập (RBAC):** Gateway kiểm tra vai trò người dùng (admin, customer, staff) từ claims của JWT để quyết định quyền truy cập vào các endpoint bảo mật.
- **Giới hạn lưu lượng (Rate Limiting):** Ngăn chặn tấn công Brute-force hoặc quá tải hệ thống bằng cách giới hạn số lượng request từ một IP trong một khoảng thời gian.
- **Tập hợp dữ liệu (Aggregation):** Thay vì client phải gọi 5 service khác nhau, Gateway có thể đóng vai trò trung gian để điều phối.

---

## 2. Pipeline Quy trình nghiệp vụ: Từ Giỏ hàng đến Đặt hàng

Dưới đây là luồng đi của dữ liệu qua các service:

### Bước 1: Tạo giỏ hàng (Create Cart)
- **Client:** Gửi POST request kèm `book_id` và `quantity`.
- **API Gateway:** Xác thực JWT, kiểm tra quyền `customer`. Chuyển tiếp request đến `cart-service`.
- **Cart Service:** Lưu thông tin vào database (SQLite), liên kết với `user_id` từ JWT.

### Bước 2: Checkout (Kiểm tra đơn hàng)
- **Client:** Gửi request yêu cầu checkout.
- **API Gateway:** Chuyển tiếp đến `cart-service`.
- **Cart Service:** 
    - Tổng hợp danh sách sách trong giỏ.
    - Gọi `book-service` để kiểm tra giá và tồn kho hiện tại.
    - Trả về thông tin tạm tính (Subtotal).

### Bước 3: Chọn Shipping (Ship Service)
- **Client:** Chọn phương thức vận chuyển.
- **API Gateway:** Chuyển tiếp đến `ship-service`.
- **Ship Service:** Tính toán phí vận chuyển dựa trên địa chỉ người dùng và khối lượng hàng hóa. Trả về `ship_id` và phí ship.

### Bước 4: Chọn Thanh toán (Pay Service)
- **Client:** Cung cấp thông tin thanh toán (thẻ hoặc ví điện tử).
- **API Gateway:** Chuyển tiếp đến `pay-service`.
- **Pay Service:** Xử lý giao dịch qua cổng thanh toán (giả lập). Nếu thành công, trả về `transaction_id`.

### Bước 5: Đặt hàng (Order Service)
- **Client:** Gửi request cuối cùng để xác nhận đặt hàng.
- **API Gateway:** Chuyển tiếp đến `order-service`.
- **Order Service (Nhạc trưởng):**
    1. Tạo bản ghi đơn hàng mới (Status: Pending).
    2. Gọi `cart-service` để lấy dữ liệu cuối cùng.
    3. Gọi `book-service` để trừ số lượng tồn kho (Inventory update).
    4. Cập nhật trạng thái thanh toán và vận chuyển.
    5. Gửi thông báo (nếu có) và yêu cầu `cart-service` xóa giỏ hàng sau khi đặt thành công.

---

## 3. Minh họa Luồng Dữ liệu (Sequence)

```
Client -> API Gateway (Auth & Rate Limit) 
          |
          +--> Cart Service (Lưu giỏ hàng)
          |
          +--> Ship Service (Tính phí vận chuyển)
          |
          +--> Pay Service (Xử lý thanh toán)
          |
          +--> Order Service (Chốt đơn & Trừ tồn kho)
```

Tài liệu này giúp hiểu rõ cách các service nhỏ lẻ phối hợp với nhau dưới sự điều phối của Gateway để tạo ra một quy trình thương mại điện tử hoàn chỉnh.

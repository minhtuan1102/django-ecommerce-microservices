# 📚 Bookstore Microservice Application

> Hệ thống quản lý nhà sách trực tuyến theo kiến trúc **Microservice**, xây dựng với **Django REST Framework**, triển khai bằng **Docker Compose**.

---

## 📑 Mục lục

1. [Tổng quan hệ thống](#-tổng-quan-hệ-thống)
2. [Kiến trúc hệ thống](#️-kiến-trúc-hệ-thống)
3. [Danh sách Services](#-danh-sách-services)
4. [Công nghệ sử dụng](#️-công-nghệ-sử-dụng)
5. [Cài đặt & Triển khai](#-cài-đặt--triển-khai)
6. [Biến môi trường](#️-biến-môi-trường)
7. [Chi tiết từng Service](#-chi-tiết-từng-service)
8. [Thiết kế Database](#️-thiết-kế-database)
9. [API Endpoints đầy đủ](#-api-endpoints-đầy-đủ)
10. [Luồng hoạt động chính](#-luồng-hoạt-động-chính)
11. [Giao diện người dùng](#️-giao-diện-người-dùng)
12. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
13. [Lệnh hữu ích](#-lệnh-hữu-ích)
14. [Troubleshooting](#-troubleshooting)
15. [Thông tin dự án](#-thông-tin-dự-án)

---

## 🔍 Tổng quan hệ thống

**Bookstore Microservice** là ứng dụng thương mại điện tử chuyên về sách, được phân chia thành **12 microservice** độc lập, mỗi service chịu trách nhiệm một nghiệp vụ riêng biệt (Single Responsibility Principle). Các service giao tiếp với nhau thông qua **HTTP REST API** nội bộ, không có message broker.

**Tính năng chính:**
- 🛒 Storefront hoàn chỉnh: duyệt sách, giỏ hàng, đặt hàng, theo dõi giao hàng
- 🧑‍💼 Admin panel: quản lý toàn bộ hệ thống (sách, khách hàng, nhân viên, đơn hàng...)
- ⭐ Hệ thống đánh giá & bình luận sách
- 🤖 AI Recommender: gợi ý sách cá nhân hóa theo lịch sử mua
- 💳 Quản lý thanh toán (COD, Banking, MoMo, VNPay)
- 🚚 Theo dõi vận chuyển thời gian thực (GHN, GHTK, Viettel Post, J&T)
- 📂 Danh mục & phân loại sách
- 👨‍💼 Quản lý nhân viên & quản lý cấp trên

---

## 🏗️ Kiến trúc hệ thống

```
                    ┌──────────────────────────────────┐
                    │          CLIENTS                  │
                    │  Browser / Mobile / API Client    │
                    └──────────────┬───────────────────┘
                                   │ HTTP
                    ┌──────────────▼───────────────────┐
                    │          API GATEWAY              │
                    │         Port :8000                │
                    │  Django + Bootstrap 5 Templates   │
                    │  ┌─────────────┐ ┌─────────────┐ │
                    │  │  Admin UI   │ │  Storefront │ │
                    │  │ /admin-*    │ │  /store/*   │ │
                    │  └─────────────┘ └─────────────┘ │
                    └──┬───┬───┬───┬───┬───┬───┬───┬──┘
                       │   │   │   │   │   │   │   │
           ┌───────────┘   │   │   │   │   │   │   └──────────────┐
           │           ┌───┘   │   │   │   └───┐                  │
           │           │       │   │   │       │                  │
    ┌──────▼──┐ ┌──────▼──┐ ┌──▼───▼──┐ ┌─────▼───┐ ┌───────────▼──┐
    │Customer │ │  Book   │ │  Cart   │ │  Order  │ │    Staff     │
    │Service  │ │Service  │ │Service  │ │Service  │ │   Service    │
    │  :8001  │ │  :8002  │ │  :8003  │ │  :8004  │ │   :8005     │
    │ MySQL   │ │ MySQL   │ │Postgres │ │Postgres │ │  Postgres   │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘ └──────────────┘

    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐
    │ Manager │ │ Catalog │ │   Pay   │ │  Ship   │ │Comment-Rate │
    │Service  │ │Service  │ │Service  │ │Service  │ │   Service   │
    │  :8006  │ │  :8007  │ │  :8008  │ │  :8009  │ │   :8010    │
    │Postgres │ │Postgres │ │Postgres │ │Postgres │ │  Postgres  │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘ └──────────────┘

                        ┌──────────────────┐
                        │  Recommender AI  │
                        │     Service      │
                        │     :8011        │
                        │  (Stateless,     │
                        │   No own DB)     │
                        └──────────────────┘

    ┌────────────────────┐     ┌─────────────────────────┐
    │    MySQL 8.0       │     │     PostgreSQL 15        │
    │  Host port :3307   │     │    Host port :5433       │
    │  ┌──────────────┐  │     │  ┌───────────────────┐   │
    │  │bookstore_mysql│ │     │  │ bookstore_postgres │  │
    │  │  (book-svc)  │  │     │  │   (cart/gateway)  │  │
    │  ├──────────────┤  │     │  ├───────────────────┤   │
    │  │ customer_db  │  │     │  │     order_db      │   │
    │  │(customer-svc)│  │     │  │     staff_db      │   │
    │  └──────────────┘  │     │  │    manager_db     │   │
    └────────────────────┘     │  │    product_db     │   │
                               │  │     pay_db        │   │
    ┌────────────────────┐     │  │     ship_db       │   │
    │   phpMyAdmin       │     │  │    comment_db     │   │
    │   Host port :8080  │     │  │  recommender_db   │   │
    └────────────────────┘     │  └───────────────────┘   │
                               └─────────────────────────┘
    ┌────────────────────┐
    │     pgAdmin        │
    │   Host port :8081  │
    └────────────────────┘
```

### Nguyên tắc thiết kế

| Nguyên tắc | Mô tả |
|-----------|-------|
| **Database per Service** | Mỗi service có database riêng biệt, không dùng chung schema |
| **API Gateway Pattern** | Mọi request từ client đều đi qua API Gateway |
| **Stateless Services** | Các service không lưu trạng thái, dễ scale |
| **Internal HTTP** | Service giao tiếp qua tên Docker container (Docker DNS nội bộ) |
| **Health Checks** | MySQL và PostgreSQL có health check trước khi service khởi động |

---

## 📋 Danh sách Services

| # | Service | Internal Host | Port | Database | Công nghệ | Mô tả |
|---|---------|--------------|------|----------|-----------|-------|
| 1 | **customer-service** | `customer-service:8001` | 8001 | MySQL `customer_db` | Django + PyMySQL | Quản lý thông tin khách hàng |
| 2 | **book-service** | `book-service:8002` | 8002 | MySQL `bookstore_mysql` | Django + PyMySQL | Quản lý sách, tồn kho |
| 3 | **cart-service** | `cart-service:8003` | 8003 | PostgreSQL `bookstore_postgres` | Django + psycopg2 | Giỏ hàng của khách |
| 4 | **order-service** | `order-service:8004` | 8004 | PostgreSQL `order_db` | Django + psycopg2 | Quản lý đơn hàng |
| 5 | **staff-service** | `staff-service:8005` | 8005 | PostgreSQL `staff_db` | Django + psycopg2 | Quản lý nhân viên |
| 6 | **manager-service** | `manager-service:8006` | 8006 | PostgreSQL `manager_db` | Django + psycopg2 | Quản lý cấp quản lý |
| 7 | **product-service** | `product-service:8007` | 8007 | PostgreSQL `product_db` | Django + psycopg2 | Danh mục & thể loại sách |
| 8 | **pay-service** | `pay-service:8008` | 8008 | PostgreSQL `pay_db` | Django + psycopg2 | Xử lý thanh toán |
| 9 | **ship-service** | `ship-service:8009` | 8009 | PostgreSQL `ship_db` | Django + psycopg2 | Vận chuyển & tracking |
| 10 | **comment-rate-service** | `comment-rate-service:8010` | 8010 | PostgreSQL `comment_db` | Django + psycopg2 | Đánh giá & bình luận |
| 11 | **recommender-ai-service** | `recommender-ai-service:8011` | 8011 | Stateless | Django + requests | Gợi ý sách AI |
| 12 | **api-gateway** | `api-gateway:8000` | 8000 | PostgreSQL `bookstore_postgres` | Django + Templates | Cổng API & giao diện |

### Công cụ hỗ trợ

| Công cụ | Port Host | Mô tả | Đăng nhập |
|---------|-----------|-------|-----------|
| **phpMyAdmin** | 8080 | Quản lý MySQL qua web | root / rootpassword |
| **pgAdmin** | 8081 | Quản lý PostgreSQL qua web | admin@admin.com / admin |

---

## 🛠️ Công nghệ sử dụng

### Backend
| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|---------|
| Python | 3.10 | Ngôn ngữ lập trình chính |
| Django | 5.x | Web framework |
| Django REST Framework | 3.x | Xây dựng REST API |
| PyMySQL | latest | Driver kết nối MySQL |
| cryptography | latest | Phụ trợ cho PyMySQL (auth plugin) |
| psycopg2-binary | latest | Driver kết nối PostgreSQL |
| requests | latest | HTTP client (gọi nội bộ giữa services) |
| gunicorn | latest | WSGI production server |

### Frontend (API Gateway)
| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|---------|
| Bootstrap | 5.3.3 | CSS framework responsive |
| Bootstrap Icons | 1.11.x | Icon library |
| Django Templates | - | Server-side rendering (SSR) |

### Infrastructure
| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|---------|
| Docker | latest | Container runtime |
| Docker Compose | v2 | Container orchestration |
| MySQL | 8.0 | Relational DB cho customer & book |
| PostgreSQL | 15 | Relational DB cho các service còn lại |
| phpMyAdmin | latest | GUI quản lý MySQL |
| pgAdmin | latest | GUI quản lý PostgreSQL |

---

## 🚀 Cài đặt & Triển khai

### Yêu cầu hệ thống

- **Docker Desktop** ≥ 4.x (đã cài đặt và đang chạy)
- **RAM** tối thiểu 4GB (khuyến nghị 8GB cho 16 container)
- **Disk** tối thiểu 5GB trống
- **OS**: Windows 10/11, macOS, Linux

### Triển khai lần đầu (Fresh Install)

```bash
# 1. Clone repository
git clone <repository-url>
cd bookstore-microservice

# 2. Build tất cả images và khởi động containers
docker-compose up --build -d

# 3. Chờ khoảng 1-2 phút để databases sẵn sàng
# Kiểm tra trạng thái (tất cả phải "Up")
docker-compose ps

# 4. Hệ thống sẵn sàng!
```

> ⚠️ **Lưu ý**: PostgreSQL init script (`init-order-db.sh`) chỉ chạy **một lần** khi volume được tạo lần đầu. Tất cả databases sẽ được tạo tự động khi fresh install.

### Truy cập hệ thống

| URL | Mô tả |
|-----|-------|
| http://localhost:8000/ | Trang chủ Admin Dashboard |
| http://localhost:8000/store/ | Storefront – Cửa hàng khách hàng |
| http://localhost:8000/admin/ | Django Admin (superuser) |
| http://localhost:8080 | phpMyAdmin – Quản lý MySQL |
| http://localhost:8081 | pgAdmin – Quản lý PostgreSQL |
| http://localhost:8001/ | Customer Service API trực tiếp |
| http://localhost:8002/ | Book Service API trực tiếp |
| http://localhost:8005/ | Staff Service API trực tiếp |
| http://localhost:8011/ | Recommender AI API trực tiếp |

### Khởi động lại sau khi tắt máy

```bash
# Volume vẫn còn, chỉ cần start lại (không cần --build)
docker-compose up -d

# Kiểm tra tất cả đã Up
docker-compose ps
```

---

## ⚙️ Biến môi trường

Tất cả biến môi trường được cấu hình trong `docker-compose.yml`.

### Database Credentials

| Biến | Giá trị | Dùng cho |
|------|---------|---------|
| `MYSQL_ROOT_PASSWORD` | `rootpassword` | MySQL root |
| `MYSQL_USER` | `bookstore_user` | MySQL app user |
| `MYSQL_PASSWORD` | `bookstore_pass` | MySQL app password |
| `POSTGRES_USER` | `bookstore_user` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `bookstore_pass` | PostgreSQL password |
| `POSTGRES_DB` | `bookstore_postgres` | Default PostgreSQL DB |

### Service Environment Variables

```yaml
# Ví dụ cấu hình cho staff-service trong docker-compose.yml
environment:
  - DB_HOST=postgres-db       # Hostname của PostgreSQL container
  - DB_PORT=5432              # Port nội bộ (trong Docker network)
  - DB_NAME=staff_db          # Tên database riêng của service
  - DB_USER=bookstore_user
  - DB_PASSWORD=bookstore_pass
```

### Django Settings Pattern (trong settings.py của mỗi service)

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}
```

---

## 📊 Chi tiết từng Service

### 1. 👤 Customer Service (Port 8001)

**Mục đích**: Quản lý thông tin khách hàng. Khi tạo khách hàng mới, tự động gọi cart-service để khởi tạo giỏ hàng trống.

**Model: Customer**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `name` | CharField(100) | NOT NULL | Tên khách hàng |
| `email` | EmailField | UNIQUE | Email (dùng để đăng nhập) |
| `created_at` | DateTimeField | auto_now_add | Thời gian tạo tài khoản |

**Database**: MySQL – `customer_db`

**Endpoints**:
| Method | URL | Mô tả | Request Body |
|--------|-----|-------|-------------|
| `GET` | `/customers/` | Danh sách tất cả khách hàng | - |
| `POST` | `/customers/` | Tạo khách hàng mới + tự tạo giỏ hàng | `{name, email}` |
| `GET` | `/customers/<id>/` | Chi tiết 1 khách hàng | - |
| `PATCH` | `/customers/<id>/` | Cập nhật thông tin | `{name?, email?}` |
| `DELETE` | `/customers/<id>/` | Xóa khách hàng | - |

---

### 2. 📖 Book Service (Port 8002)

**Mục đích**: Quản lý catalog sách và tồn kho. Cung cấp endpoint đặc biệt để trừ tồn kho khi đặt hàng.

**Model: Book**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `title` | CharField(200) | NOT NULL | Tên sách |
| `author` | CharField(200) | NOT NULL | Tác giả |
| `price` | DecimalField(10,2) | NOT NULL | Giá bán (VND) |
| `stock` | IntegerField | default=0 | Số lượng tồn kho |
| `description` | TextField | blank=True | Mô tả sách |
| `cover_image` | URLField | blank=True | URL ảnh bìa |
| `created_at` | DateTimeField | auto_now_add | Thời gian thêm sách |

**Database**: MySQL – `bookstore_mysql`

**Endpoints**:
| Method | URL | Mô tả | Request Body |
|--------|-----|-------|-------------|
| `GET` | `/books/` | Danh sách tất cả sách | - |
| `POST` | `/books/` | Thêm sách mới | `{title, author, price, stock, ...}` |
| `GET` | `/books/<id>/` | Chi tiết sách | - |
| `PUT` | `/books/<id>/` | Cập nhật toàn bộ | Full body |
| `PATCH` | `/books/<id>/` | Cập nhật một phần | Partial body |
| `DELETE` | `/books/<id>/` | Xóa sách | - |
| `POST` | `/books/<id>/reduce-stock/` | Trừ tồn kho | `{quantity: N}` |

---

### 3. 🛒 Cart Service (Port 8003)

**Mục đích**: Quản lý giỏ hàng. Mỗi khách hàng có 1 giỏ hàng, chứa nhiều `CartItem`.

**Models:**

*Cart*
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `customer_id` | IntegerField | UNIQUE | ID khách hàng (ref customer-service) |
| `created_at` | DateTimeField | auto_now_add | Thời gian tạo |

*CartItem*
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `cart` | ForeignKey(Cart) | CASCADE | Giỏ hàng chứa item này |
| `book_id` | IntegerField | NOT NULL | ID sách (ref book-service) |
| `quantity` | IntegerField | default=1 | Số lượng |
| `unique_together` | (cart, book_id) | - | Mỗi sách chỉ 1 CartItem/giỏ |

**Database**: PostgreSQL – `bookstore_postgres`

**Endpoints**:
| Method | URL | Mô tả |
|--------|-----|-------|
| `POST` | `/carts/` | Tạo giỏ hàng mới |
| `GET` | `/carts/<customer_id>/` | Xem giỏ hàng của khách |
| `DELETE` | `/carts/<customer_id>/clear/` | Xóa toàn bộ giỏ hàng |
| `POST` | `/cart-items/` | Thêm/tăng sản phẩm vào giỏ |
| `PATCH` | `/cart-items/<cart_id>/<book_id>/` | Cập nhật số lượng |
| `DELETE` | `/cart-items/<cart_id>/<book_id>/` | Xóa 1 sản phẩm |

---

### 4. 📦 Order Service (Port 8004)

**Mục đích**: Quản lý toàn bộ vòng đời đơn hàng từ lúc tạo đến khi giao thành công.

**Model: Order**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `customer_id` | IntegerField | NOT NULL | ID khách hàng |
| `total_price` | DecimalField(10,2) | NOT NULL | Tổng tiền hàng |
| `shipping_fee` | DecimalField(10,2) | default=0 | Phí vận chuyển |
| `payment_method` | CharField | NOT NULL | Phương thức thanh toán |
| `shipping_address` | TextField | NOT NULL | Địa chỉ giao hàng |
| `status` | CharField | default=pending | Trạng thái đơn hàng |
| `created_at` | DateTimeField | auto_now_add | Thời gian đặt hàng |
| `updated_at` | DateTimeField | auto_now | Lần cập nhật cuối |

*Order Status Flow*: `pending` → `paid` → `shipping` → `delivered` / `cancelled`

**Model: OrderItem**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `order` | ForeignKey(Order) | CASCADE | Đơn hàng chứa item |
| `book_id` | IntegerField | NOT NULL | ID sách |
| `quantity` | IntegerField | NOT NULL | Số lượng đặt |
| `price` | DecimalField(10,2) | NOT NULL | Giá tại thời điểm đặt |

**Database**: PostgreSQL – `order_db`

**Endpoints**:
| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/orders/` | Tất cả đơn hàng |
| `POST` | `/orders/` | Tạo đơn hàng mới |
| `GET` | `/orders/<id>/` | Chi tiết đơn hàng (kèm OrderItems) |
| `PATCH` | `/orders/<id>/` | Cập nhật trạng thái |
| `DELETE` | `/orders/<id>/` | Xóa đơn hàng |
| `GET` | `/orders/customer/<customer_id>/` | Đơn hàng theo khách |

---

### 5. 👨‍💼 Staff Service (Port 8005)

**Mục đích**: Quản lý nhân viên bán hàng, nhân viên kho và nhân viên hỗ trợ.

**Model: Staff**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `name` | CharField(100) | NOT NULL | Tên nhân viên |
| `email` | EmailField | UNIQUE | Email liên hệ |
| `phone` | CharField(20) | blank=True | Số điện thoại |
| `role` | CharField | choices | Chức vụ |
| `is_active` | BooleanField | default=True | Đang làm việc hay không |
| `created_at` | DateTimeField | auto_now_add | Ngày tuyển dụng |

*Roles*: `sales` (Bán hàng) | `warehouse` (Kho) | `support` (Hỗ trợ khách hàng)

**Database**: PostgreSQL – `staff_db`

**Endpoints**:
| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/staff/` | Danh sách nhân viên |
| `POST` | `/staff/` | Thêm nhân viên mới |
| `GET` | `/staff/<id>/` | Chi tiết nhân viên |
| `PATCH` | `/staff/<id>/` | Cập nhật thông tin |
| `DELETE` | `/staff/<id>/` | Xóa nhân viên |

---

### 6. 🏢 Manager Service (Port 8006)

**Mục đích**: Quản lý cấp quản lý, phân theo phòng ban.

**Model: Manager**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `name` | CharField(100) | NOT NULL | Tên quản lý |
| `email` | EmailField | UNIQUE | Email liên hệ |
| `phone` | CharField(20) | blank=True | Số điện thoại |
| `department` | CharField | choices | Phòng ban phụ trách |
| `is_active` | BooleanField | default=True | Đang làm việc không |
| `created_at` | DateTimeField | auto_now_add | Ngày bổ nhiệm |

*Departments*: `general` (Ban giám đốc) | `sales` (Kinh doanh) | `warehouse` (Kho vận)

**Database**: PostgreSQL – `manager_db`

**Endpoints**:
| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/managers/` | Danh sách quản lý |
| `POST` | `/managers/` | Thêm quản lý mới |
| `GET` | `/managers/<id>/` | Chi tiết quản lý |
| `PATCH` | `/managers/<id>/` | Cập nhật thông tin |
| `DELETE` | `/managers/<id>/` | Xóa quản lý |

---

### 7. 📂 Catalog Service (Port 8007)

**Mục đích**: Quản lý danh mục (thể loại) sách. Một sách có thể thuộc nhiều danh mục.

**Model: Category**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `name` | CharField(100) | UNIQUE | Tên danh mục (VD: Văn học, Kỹ năng...) |
| `description` | TextField | blank=True | Mô tả danh mục |
| `created_at` | DateTimeField | auto_now_add | Ngày tạo |

**Model: BookCatalog** (Bảng trung gian Many-to-Many)
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `book_id` | IntegerField | NOT NULL | ID sách (từ book-service) |
| `category` | ForeignKey(Category) | CASCADE | Danh mục |
| `unique_together` | (book_id, category) | - | Mỗi sách/danh mục chỉ 1 lần |

**Database**: PostgreSQL – `product_db`

**Endpoints**:
| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/categories/` | Danh sách danh mục |
| `POST` | `/categories/` | Tạo danh mục mới |
| `GET` | `/categories/<id>/` | Chi tiết danh mục (kèm book_ids) |
| `PATCH` | `/categories/<id>/` | Cập nhật danh mục |
| `DELETE` | `/categories/<id>/` | Xóa danh mục |
| `GET` | `/book-catalogs/` | Tất cả mapping sách-danh mục |
| `POST` | `/book-catalogs/` | Gắn sách vào danh mục |
| `DELETE` | `/book-catalogs/<id>/` | Gỡ sách khỏi danh mục |

---

### 8. 💳 Pay Service (Port 8008)

**Mục đích**: Ghi nhận và theo dõi trạng thái thanh toán. Hỗ trợ nhiều phương thức phổ biến tại Việt Nam. Tự động sinh mã giao dịch (UUID).

**Model: Payment**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `order_id` | IntegerField | NOT NULL | ID đơn hàng (từ order-service) |
| `customer_id` | IntegerField | NOT NULL | ID khách hàng |
| `amount` | DecimalField(12,2) | NOT NULL | Số tiền thanh toán |
| `method` | CharField | choices | Phương thức thanh toán |
| `status` | CharField | default=pending | Trạng thái thanh toán |
| `transaction_id` | UUIDField | auto=uuid4 | Mã giao dịch tự sinh |
| `created_at` | DateTimeField | auto_now_add | Thời gian tạo |
| `updated_at` | DateTimeField | auto_now | Lần cập nhật cuối |

*Methods*: `cod` (Tiền mặt khi nhận) | `banking` (Chuyển khoản) | `momo` (Ví MoMo) | `vnpay` (VNPay)

*Status Flow*: `pending` → `completed` / `failed` / `refunded`

**Database**: PostgreSQL – `pay_db`

**Endpoints**:
| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/payments/` | Danh sách thanh toán |
| `POST` | `/payments/` | Tạo bản ghi thanh toán |
| `GET` | `/payments/<id>/` | Chi tiết thanh toán |
| `PATCH` | `/payments/<id>/` | Cập nhật trạng thái |
| `DELETE` | `/payments/<id>/` | Xóa bản ghi |
| `GET` | `/payments/order/<order_id>/` | Thanh toán theo đơn hàng |

---

### 9. 🚚 Ship Service (Port 8009)

**Mục đích**: Tạo và theo dõi đơn vận chuyển. Hỗ trợ các đơn vị vận chuyển phổ biến tại Việt Nam, tự động sinh mã tracking (UUID).

**Model: Shipment**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `order_id` | IntegerField | NOT NULL | ID đơn hàng |
| `tracking_code` | UUIDField | auto=uuid4 | Mã tracking tự sinh |
| `carrier` | CharField | choices | Đơn vị vận chuyển |
| `status` | CharField | default=preparing | Trạng thái vận chuyển |
| `shipping_address` | TextField | NOT NULL | Địa chỉ giao hàng |
| `estimated_days` | IntegerField | default=3 | Số ngày giao dự kiến |
| `created_at` | DateTimeField | auto_now_add | Ngày tạo đơn vận chuyển |
| `updated_at` | DateTimeField | auto_now | Lần cập nhật cuối |

*Carriers*: `ghn` (Giao Hàng Nhanh) | `ghtk` (Giao Hàng Tiết Kiệm) | `viettel_post` (Viettel Post) | `jt_express` (J&T Express)

*Status Flow*: `preparing` → `shipped` → `in_transit` → `delivered` / `cancelled`

**Database**: PostgreSQL – `ship_db`

**Endpoints**:
| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/shipments/` | Danh sách vận chuyển |
| `POST` | `/shipments/` | Tạo đơn vận chuyển |
| `GET` | `/shipments/<id>/` | Chi tiết đơn vận chuyển |
| `PATCH` | `/shipments/<id>/` | Cập nhật trạng thái |
| `DELETE` | `/shipments/<id>/` | Xóa đơn vận chuyển |
| `GET` | `/shipments/order/<order_id>/` | Vận chuyển theo đơn hàng |

---

### 10. ⭐ Comment-Rate Service (Port 8010)

**Mục đích**: Cho phép khách hàng đánh giá sao (1-5) và viết bình luận. Mỗi khách chỉ được đánh giá một lần cho mỗi cuốn sách. Endpoint `/reviews/book/<id>/` trả về điểm trung bình kèm danh sách đánh giá.

**Model: Review**
| Field | Type | Constraint | Mô tả |
|-------|------|-----------|-------|
| `id` | AutoField | PK | ID tự tăng |
| `customer_id` | IntegerField | NOT NULL | ID khách hàng |
| `book_id` | IntegerField | NOT NULL | ID sách |
| `rating` | IntegerField | 1–5 | Số sao đánh giá |
| `comment` | TextField | blank=True | Nội dung bình luận |
| `created_at` | DateTimeField | auto_now_add | Thời gian đánh giá |
| `unique_together` | (customer_id, book_id) | - | Mỗi cặp khách-sách chỉ 1 đánh giá |

**Database**: PostgreSQL – `comment_db`

**Endpoints**:
| Method | URL | Mô tả | Response đặc biệt |
|--------|-----|-------|-------------------|
| `GET` | `/reviews/` | Tất cả đánh giá | - |
| `POST` | `/reviews/` | Tạo đánh giá mới | - |
| `GET` | `/reviews/book/<book_id>/` | Đánh giá theo sách | Kèm `average_rating`, `total_reviews` |
| `DELETE` | `/reviews/<id>/` | Xóa đánh giá | - |

**Response mẫu** – `GET /reviews/book/1/`:
```json
{
  "book_id": 1,
  "average_rating": 4.3,
  "total_reviews": 15,
  "reviews": [
    {
      "id": 1,
      "customer_id": 3,
      "rating": 5,
      "comment": "Sách hay lắm, rất bổ ích!",
      "created_at": "2026-03-10T10:30:00Z"
    }
  ]
}
```

---

### 11. 🤖 Recommender AI Service (Port 8011)

**Mục đích**: Gợi ý sách cá nhân hóa dựa trên lịch sử mua hàng và đánh giá cộng đồng. Service này **stateless** — không có database riêng, tổng hợp dữ liệu từ các service khác theo thời gian thực.

**Thuật toán gợi ý (Hybrid Recommendation):**
```
1. Gọi order-service   → lấy lịch sử đơn hàng của customer
2. Trích xuất          → danh sách book_id đã mua
3. Gọi book-service    → lấy toàn bộ catalog sách
4. Lọc bỏ             → sách khách đã mua
5. Gọi comment-rate   → lấy đánh giá cho từng sách
6. Tính điểm          → score = avg_rating × log(1 + total_reviews)
7. Sắp xếp            → giảm dần theo score
8. Trả về             → top 5 sách gợi ý
```

**Gợi ý sách phổ biến** (endpoint `/popular/`, dùng khi chưa đăng nhập):
```
1. Gọi book-service    → lấy toàn bộ sách
2. Gọi comment-rate   → lấy rating từng sách
3. Xếp hạng           → theo điểm × lượt đánh giá
4. Trả về             → top 5 phổ biến nhất
```

**Database**: Stateless (không có DB riêng)

**Endpoints**:
| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/recommendations/<customer_id>/` | Top 5 gợi ý cá nhân hóa |
| `GET` | `/popular/` | Top 5 sách phổ biến nhất |

**Response mẫu** – `GET /recommendations/1/`:
```json
{
  "customer_id": 1,
  "recommendations": [
    {
      "id": 5,
      "title": "Đắc Nhân Tâm",
      "author": "Dale Carnegie",
      "price": "89000.00",
      "stock": 50,
      "average_rating": 4.8,
      "total_reviews": 120
    }
  ]
}
```

---

### 12. 🌐 API Gateway (Port 8000)

**Mục đích**: Cổng duy nhất cho mọi request từ client. Không expose trực tiếp các service nội bộ. Đảm nhận cả vai trò **reverse proxy** và **rendering giao diện** (SSR với Django Templates).

**Service URLs cấu hình sẵn trong `views.py`:**
```python
CUSTOMER_SERVICE_URL     = "http://customer-service:8001"
BOOK_SERVICE_URL         = "http://book-service:8002"
CART_SERVICE_URL         = "http://cart-service:8003"
ORDER_SERVICE_URL        = "http://order-service:8004"
STAFF_SERVICE_URL        = "http://staff-service:8005"
MANAGER_SERVICE_URL      = "http://manager-service:8006"
PRODUCT_SERVICE_URL      = "http://product-service:8007"
PAY_SERVICE_URL          = "http://pay-service:8008"
SHIP_SERVICE_URL         = "http://ship-service:8009"
COMMENT_RATE_SERVICE_URL = "http://comment-rate-service:8010"
RECOMMENDER_SERVICE_URL  = "http://recommender-ai-service:8011"
```

**Authentication**: Session-based (`customer_id`, `customer_name` lưu trong Django session → PostgreSQL)

**Database**: PostgreSQL – `bookstore_postgres` (lưu session)

**Admin Panel Routes**:
| Route | View function | Mô tả |
|-------|--------------|-------|
| `/` | `home` | Dashboard 7 thống kê |
| `/books/` | `admin_book_list` | Quản lý sách |
| `/customers/` | `admin_customer_list` | Quản lý khách hàng |
| `/orders/` | `admin_order_list` | Quản lý đơn hàng |
| `/staff/` | `admin_staff_list` | Quản lý nhân viên |
| `/managers/` | `admin_manager_list` | Quản lý cấp quản lý |
| `/catalog/` | `admin_catalog_list` | Quản lý danh mục |
| `/payments/` | `admin_payment_list` | Quản lý thanh toán |
| `/shipments/` | `admin_shipment_list` | Quản lý vận chuyển |
| `/reviews/` | `admin_review_list` | Quản lý đánh giá |

**Storefront Routes**:
| Route | View function | Mô tả |
|-------|--------------|-------|
| `/store/` | `store_home` | Trang chủ (kèm gợi ý AI) |
| `/store/books/` | `store_book_list` | Danh sách tất cả sách |
| `/store/book/<id>/` | `store_book_detail` | Chi tiết sách + đánh giá |
| `/store/cart/` | `store_cart` | Xem giỏ hàng |
| `/store/checkout/` | `store_checkout` | Thanh toán |
| `/store/orders/` | `store_order_list` | Lịch sử đơn hàng |
| `/store/order/<id>/` | `store_order_detail` | Chi tiết đơn + tracking |
| `/store/review/<book_id>/` | `store_add_review` | Gửi đánh giá sách |
| `/store/login/` | `store_login` | Đăng nhập |
| `/store/register/` | `store_register` | Đăng ký |

---

## 🗄️ Thiết kế Database

### Sơ đồ quan hệ

```
[customer_db – MySQL]
  Customer(id, name, email, created_at)
       │
       │ customer_id (logical FK)
       ▼
[bookstore_postgres – PostgreSQL]
  Cart(id, customer_id, created_at)
       │
       └── CartItem(id, cart_id, book_id, quantity)
                                  │
                                  │ book_id (logical FK)
                                  ▼
[bookstore_mysql – MySQL]
  Book(id, title, author, price, stock, description, cover_image)

[order_db – PostgreSQL]
  Order(id, customer_id, total_price, shipping_fee, payment_method,
        shipping_address, status, created_at)
       │
       └── OrderItem(id, order_id, book_id, quantity, price)
       │
       ├── order_id ──► Payment(order_id, customer_id, amount, method,
       │                        status, transaction_id)  [pay_db]
       │
       └── order_id ──► Shipment(order_id, tracking_code, carrier,
                                 status, shipping_address, estimated_days)  [ship_db]

[comment_db – PostgreSQL]
  Review(id, customer_id, book_id, rating, comment, created_at)
  UNIQUE: (customer_id, book_id)

[product_db – PostgreSQL]
  Category(id, name, description, created_at)
       │
       └── BookCatalog(id, book_id, category_id)
           UNIQUE: (book_id, category_id)

[staff_db]    Staff(id, name, email, phone, role, is_active, created_at)
[manager_db]  Manager(id, name, email, phone, department, is_active, created_at)
[recommender_db]  (trống – service stateless)
```

### MySQL (Host: `localhost:3307` | Docker internal: `mysql-db:3306`)

| Database | Service | Tables | Ghi chú |
|----------|---------|--------|--------|
| `bookstore_mysql` | book-service | `app_book` | Catalog sách |
| `customer_db` | customer-service | `app_customer` | Thông tin khách hàng |

### PostgreSQL (Host: `localhost:5433` | Docker internal: `postgres-db:5432`)

| Database | Service | Tables | Ghi chú |
|----------|---------|--------|--------|
| `bookstore_postgres` | cart-service, api-gateway | `app_cart`, `app_cartitem`, `django_session` | Shared |
| `order_db` | order-service | `app_order`, `app_orderitem` | Đơn hàng |
| `staff_db` | staff-service | `app_staff` | Nhân viên |
| `manager_db` | manager-service | `app_manager` | Quản lý |
| `product_db` | product-service | `app_category`, `app_bookcatalog` | Danh mục |
| `pay_db` | pay-service | `app_payment` | Thanh toán |
| `ship_db` | ship-service | `app_shipment` | Vận chuyển |
| `comment_db` | comment-rate-service | `app_review` | Đánh giá |
| `recommender_db` | recommender-ai-service | (trống) | Stateless |

---

## 🔗 API Endpoints đầy đủ

```
Customer Service (:8001)
  GET    /customers/
  POST   /customers/                     body: {name, email}
  GET    /customers/{id}/
  PATCH  /customers/{id}/               body: {name?, email?}
  DELETE /customers/{id}/

Book Service (:8002)
  GET    /books/
  POST   /books/                         body: {title, author, price, stock, ...}
  GET    /books/{id}/
  PUT    /books/{id}/                    body: full
  PATCH  /books/{id}/                    body: partial
  DELETE /books/{id}/
  POST   /books/{id}/reduce-stock/       body: {quantity}

Cart Service (:8003)
  POST   /carts/                         body: {customer_id}
  GET    /carts/{customer_id}/
  DELETE /carts/{customer_id}/clear/
  POST   /cart-items/                    body: {cart_id, book_id, quantity}
  PATCH  /cart-items/{cart_id}/{book_id}/  body: {quantity}
  DELETE /cart-items/{cart_id}/{book_id}/

Order Service (:8004)
  GET    /orders/
  POST   /orders/                        body: {customer_id, items[], shipping_address, ...}
  GET    /orders/{id}/
  PATCH  /orders/{id}/                   body: {status}
  DELETE /orders/{id}/
  GET    /orders/customer/{customer_id}/

Staff Service (:8005)
  GET    /staff/
  POST   /staff/                         body: {name, email, phone, role}
  GET    /staff/{id}/
  PATCH  /staff/{id}/                    body: partial
  DELETE /staff/{id}/

Manager Service (:8006)
  GET    /managers/
  POST   /managers/                      body: {name, email, phone, department}
  GET    /managers/{id}/
  PATCH  /managers/{id}/                 body: partial
  DELETE /managers/{id}/

Catalog Service (:8007)
  GET    /categories/
  POST   /categories/                    body: {name, description}
  GET    /categories/{id}/
  PATCH  /categories/{id}/               body: partial
  DELETE /categories/{id}/
  GET    /book-catalogs/
  POST   /book-catalogs/                 body: {book_id, category}
  DELETE /book-catalogs/{id}/

Pay Service (:8008)
  GET    /payments/
  POST   /payments/                      body: {order_id, customer_id, amount, method}
  GET    /payments/{id}/
  PATCH  /payments/{id}/                 body: {status}
  DELETE /payments/{id}/
  GET    /payments/order/{order_id}/

Ship Service (:8009)
  GET    /shipments/
  POST   /shipments/                     body: {order_id, carrier, shipping_address, estimated_days}
  GET    /shipments/{id}/
  PATCH  /shipments/{id}/                body: {status}
  DELETE /shipments/{id}/
  GET    /shipments/order/{order_id}/

Comment-Rate Service (:8010)
  GET    /reviews/
  POST   /reviews/                       body: {customer_id, book_id, rating, comment}
  GET    /reviews/book/{book_id}/        → {average_rating, total_reviews, reviews[]}
  DELETE /reviews/{id}/

Recommender AI Service (:8011)
  GET    /recommendations/{customer_id}/ → {customer_id, recommendations[top5]}
  GET    /popular/                        → {books[top5]}
```

---

## 🔄 Luồng hoạt động chính

### Luồng 1: Đăng ký & Đăng nhập

```
Khách hàng nhập tên + email
         │
         ▼
  API Gateway POST /store/register/
         │
         ├─► Customer Service POST /customers/
         │       ├─ Tạo Customer record (MySQL)
         │       └─ Gọi Cart Service POST /carts/ → tạo giỏ hàng trống
         │
         ▼
  API Gateway lưu session: {customer_id, customer_name}
         │
         ▼
  Redirect → /store/ (trang chủ với gợi ý AI)
```

### Luồng 2: Mua hàng (End-to-End Checkout)

```
Khách xem sách → Nhấn "Thêm vào giỏ"
         │
         ├─► Cart Service POST /cart-items/  (cart_id, book_id, qty)
         │
         ▼
Khách vào giỏ hàng → Nhấn "Đặt hàng"
         │
         ▼
  API Gateway /store/checkout/
         │
         ├─ 1. Lấy giỏ hàng     → Cart Service GET /carts/{customer_id}/
         ├─ 2. Kiểm tra tồn kho → Book Service GET /books/{id}/  (từng sách)
         ├─ 3. Trừ tồn kho      → Book Service POST /books/{id}/reduce-stock/
         ├─ 4. Tạo đơn hàng     → Order Service POST /orders/
         ├─ 5. Tạo thanh toán   → Pay Service POST /payments/
         ├─ 6. Tạo vận chuyển   → Ship Service POST /shipments/
         └─ 7. Xóa giỏ hàng     → Cart Service DELETE /carts/{id}/clear/
         │
         ▼
  Redirect → Trang xác nhận đặt hàng thành công
```

### Luồng 3: Gợi ý sách AI

```
Khách đăng nhập → Truy cập /store/
         │
         ▼
  API Gateway gọi Recommender AI
  GET /recommendations/{customer_id}/
         │
         ▼
  Recommender AI (stateless):
    ├─ GET /orders/customer/{id}/     → sách đã mua (order-service)
    ├─ GET /books/                    → toàn bộ catalog (book-service)
    ├─ GET /reviews/book/{id}/        → rating từng sách (comment-rate-service)
    ├─ Lọc bỏ sách đã mua
    ├─ Tính score = avg_rating × log(1 + total_reviews)
    └─ Trả về TOP 5 cao điểm nhất
         │
         ▼
  Hiển thị "Gợi ý cho bạn" trên trang chủ storefront
  (fallback: /popular/ nếu chưa có lịch sử mua)
```

### Luồng 4: Đánh giá sách

```
Khách vào trang chi tiết sách → Điền form đánh giá
         │
         ▼
  API Gateway POST /store/review/{book_id}/
         │
         ├─► Comment-Rate Service POST /reviews/
         │       {customer_id, book_id, rating, comment}
         │       → Kiểm tra unique_together, lưu vào DB
         │
         ▼
  Trang sách reload → GET /reviews/book/{book_id}/
  Hiển thị rating trung bình mới + đánh giá của khách
```

---

## 🖥️ Giao diện người dùng

### Admin Panel

| Trang | URL | Tính năng chính |
|-------|-----|----------------|
| Dashboard | `/` | 7 thẻ thống kê realtime: sách, khách hàng, đơn hàng, nhân viên, quản lý, thanh toán, vận chuyển |
| Quản lý sách | `/books/` | Grid hiển thị tồn kho, form thêm/sửa/xóa, cảnh báo hết hàng |
| Quản lý khách | `/customers/` | Danh sách, tìm kiếm, xem chi tiết, thêm khách mới |
| Quản lý đơn hàng | `/orders/` | Filter theo trạng thái, xem chi tiết items, cập nhật trạng thái đơn |
| Quản lý nhân viên | `/staff/` | CRUD đầy đủ, badge role (sales/warehouse/support), toggle active |
| Quản lý cấp quản lý | `/managers/` | CRUD đầy đủ, badge phòng ban, toggle active |
| Danh mục sách | `/catalog/` | Tạo danh mục, gắn sách vào danh mục, xem sách theo danh mục |
| Thanh toán | `/payments/` | Xem trạng thái, cập nhật completed/failed/refunded, lọc theo method |
| Vận chuyển | `/shipments/` | Theo dõi tracking code, cập nhật trạng thái từng bước giao hàng |
| Đánh giá | `/reviews/` | Xem tất cả review + rating sao, xóa review vi phạm |

### Storefront (Giao diện khách hàng)

| Trang | URL | Tính năng chính |
|-------|-----|----------------|
| Trang chủ | `/store/` | Banner quảng cáo, section "Gợi ý cho bạn" (AI), sách mới nhất |
| Danh sách sách | `/store/books/` | Grid Bootstrap, hiển thị giá + tồn kho, nút thêm giỏ hàng nhanh |
| Chi tiết sách | `/store/book/<id>/` | Ảnh bìa, mô tả đầy đủ, điểm trung bình sao, danh sách đánh giá, form viết review (cần đăng nhập) |
| Giỏ hàng | `/store/cart/` | Danh sách sản phẩm, số lượng, tổng tiền, xóa từng món, nút checkout |
| Thanh toán | `/store/checkout/` | Form điền địa chỉ, chọn phương thức (COD/Banking/MoMo/VNPay), tóm tắt đơn hàng |
| Lịch sử đơn | `/store/orders/` | Tất cả đơn hàng với badge trạng thái màu sắc |
| Chi tiết đơn | `/store/order/<id>/` | Danh sách sản phẩm + card tracking vận chuyển (carrier, status, tracking code) + card thanh toán |

---

## 📁 Cấu trúc thư mục

```
bookstore-microservice/
├── docker-compose.yml                  # Orchestration 16 containers
├── README.md                           # Tài liệu này
│
├── init-scripts/                       # Database initialization scripts
│   ├── init-customer-db.sql            # MySQL: tạo customer_db database
│   └── init-order-db.sh                # PostgreSQL: tạo 9 databases
│
├── api-gateway/                        # Service 12: Cổng API & giao diện
│   ├── Dockerfile
│   ├── manage.py
│   ├── requirements.txt                # Django, DRF, psycopg2, requests
│   ├── api_gateway/
│   │   ├── settings.py                 # DB config, INSTALLED_APPS, TEMPLATES
│   │   └── urls.py                     # Include app.urls
│   └── app/
│       ├── views.py                    # 35+ view functions (admin + storefront)
│       ├── urls.py                     # 30+ URL patterns
│       └── templates/
│           ├── base.html               # Layout chung + Bootstrap 5 sidebar
│           ├── home.html               # Admin dashboard (7 stat cards)
│           ├── books.html              # Admin: quản lý sách
│           ├── customers.html          # Admin: quản lý khách hàng
│           ├── orders.html             # Admin: quản lý đơn hàng
│           ├── staff.html              # Admin: quản lý nhân viên
│           ├── managers.html           # Admin: quản lý cấp trên
│           ├── catalog.html            # Admin: danh mục sách
│           ├── payments.html           # Admin: thanh toán
│           ├── shipments.html          # Admin: vận chuyển
│           ├── reviews.html            # Admin: đánh giá
│           ├── store_home.html         # Storefront: trang chủ + AI gợi ý
│           ├── store_books.html        # Storefront: danh sách sách
│           ├── store_book_detail.html  # Storefront: chi tiết + form review
│           ├── store_cart.html         # Storefront: giỏ hàng
│           ├── store_checkout.html     # Storefront: form thanh toán
│           ├── store_orders.html       # Storefront: lịch sử đơn hàng
│           ├── store_order_detail.html # Storefront: chi tiết + tracking card
│           ├── store_login.html        # Storefront: đăng nhập
│           └── store_register.html     # Storefront: đăng ký
│
├── customer-service/                   # Service 1: Khách hàng (MySQL)
├── book-service/                       # Service 2: Sách (MySQL)
├── cart-service/                       # Service 3: Giỏ hàng (PostgreSQL)
├── order-service/                      # Service 4: Đơn hàng (PostgreSQL)
├── staff-service/                      # Service 5: Nhân viên (PostgreSQL)
├── manager-service/                    # Service 6: Quản lý (PostgreSQL)
├── product-service/                    # Service 7: Danh mục (PostgreSQL)
├── pay-service/                        # Service 8: Thanh toán (PostgreSQL)
├── ship-service/                       # Service 9: Vận chuyển (PostgreSQL)
├── comment-rate-service/               # Service 10: Đánh giá (PostgreSQL)
└── recommender-ai-service/             # Service 11: Gợi ý AI (Stateless)
```

### Cấu trúc chuẩn mỗi service

```
{service-name}/
├── Dockerfile                   # FROM python:3.10-slim
├── manage.py                    # Django CLI
├── requirements.txt             # Django, DRF, DB driver
│
├── {service_name}/              # Django project settings
│   ├── settings.py              # DB config từ env vars
│   ├── urls.py                  # Include app.urls
│   ├── wsgi.py
│   └── asgi.py
│
└── app/                         # Django app chính
    ├── models.py                # Data models + migrations
    ├── serializers.py           # DRF ModelSerializer
    ├── views.py                 # APIView / ModelViewSet
    ├── urls.py                  # URL routing
    ├── admin.py                 # Django admin registration
    └── migrations/              # Database migration files
```

### Dockerfile chuẩn (áp dụng cho tất cả services)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE {PORT}
CMD ["python", "manage.py", "runserver", "0.0.0.0:{PORT}"]
```

---

## 🔧 Lệnh hữu ích

### Khởi động & Dừng

```bash
# Fresh install: build và khởi động tất cả
docker-compose up --build -d

# Khởi động lại khi đã build rồi (nhanh hơn)
docker-compose up -d

# Dừng tất cả containers (giữ nguyên data trong volumes)
docker-compose down

# ⚠️ NGUY HIỂM: Dừng và XÓA toàn bộ volumes (reset sạch database)
docker-compose down -v

# Rebuild và restart 1 service cụ thể
docker-compose up --build -d book-service

# Restart nhiều service cùng lúc
docker-compose up -d staff-service manager-service product-service
```

### Theo dõi Logs

```bash
# Logs realtime tất cả services
docker-compose logs -f

# Logs 1 service
docker-compose logs -f api-gateway

# Logs nhiều service cùng lúc
docker-compose logs -f book-service order-service

# 50 dòng log gần nhất không realtime
docker-compose logs --tail=50 recommender-ai-service

# Logs từ 5 phút gần đây
docker-compose logs -f --since="5m" staff-service
```

### Kiểm tra trạng thái

```bash
# Xem trạng thái tất cả containers trong project
docker-compose ps

# Xem tất cả containers kể cả đã dừng
docker ps -a

# Kiểm tra resource usage (CPU, RAM)
docker stats

# Inspect chi tiết 1 container
docker inspect bookstore-microservice-api-gateway-1

# Liệt kê tất cả Docker networks
docker network ls
```

### Database Operations

```bash
# === PostgreSQL ===

# Vào psql CLI
docker exec -it bookstore-microservice-postgres-db-1 psql -U bookstore_user -d bookstore_postgres

# Liệt kê tất cả databases
docker exec bookstore-microservice-postgres-db-1 psql -U bookstore_user -l

# Chạy query nhanh
docker exec bookstore-microservice-postgres-db-1 \
  psql -U bookstore_user -d order_db -c "SELECT * FROM app_order LIMIT 5;"

# === MySQL ===

# Vào MySQL CLI
docker exec -it bookstore-microservice-mysql-db-1 mysql -u bookstore_user -pbookstore_pass

# Liệt kê databases
docker exec bookstore-microservice-mysql-db-1 \
  mysql -u bookstore_user -pbookstore_pass -e "SHOW DATABASES;"

# === Django Migrations ===

# Chạy migration cho 1 service
docker-compose exec staff-service python manage.py migrate

# Tạo migration mới sau khi thay đổi models
docker-compose exec book-service python manage.py makemigrations
docker-compose exec book-service python manage.py migrate

# Tạo Django superuser
docker-compose exec api-gateway python manage.py createsuperuser
```

### Tạo thủ công databases PostgreSQL

> Chỉ cần khi volume đã tồn tại từ trước mà thiếu database (lỗi "database does not exist")

```bash
docker exec bookstore-microservice-postgres-db-1 psql -U bookstore_user -d bookstore_postgres -c "CREATE DATABASE staff_db OWNER bookstore_user"
docker exec bookstore-microservice-postgres-db-1 psql -U bookstore_user -d bookstore_postgres -c "CREATE DATABASE manager_db OWNER bookstore_user"
docker exec bookstore-microservice-postgres-db-1 psql -U bookstore_user -d bookstore_postgres -c "CREATE DATABASE product_db OWNER bookstore_user"
docker exec bookstore-microservice-postgres-db-1 psql -U bookstore_user -d bookstore_postgres -c "CREATE DATABASE pay_db OWNER bookstore_user"
docker exec bookstore-microservice-postgres-db-1 psql -U bookstore_user -d bookstore_postgres -c "CREATE DATABASE ship_db OWNER bookstore_user"
docker exec bookstore-microservice-postgres-db-1 psql -U bookstore_user -d bookstore_postgres -c "CREATE DATABASE comment_db OWNER bookstore_user"
docker exec bookstore-microservice-postgres-db-1 psql -U bookstore_user -d bookstore_postgres -c "CREATE DATABASE recommender_db OWNER bookstore_user"
```

> ⚠️ `CREATE DATABASE` không thể chạy trong transaction block → phải tạo từng database riêng biệt.

---

## 🔎 Troubleshooting

### ❌ Lỗi: `database "xxx_db" does not exist`

**Nguyên nhân**: PostgreSQL init script (`init-order-db.sh`) chỉ chạy khi tạo volume Docker lần đầu. Nếu volume đã tồn tại, script không chạy lại kể cả khi bạn đã cập nhật nó.

**Giải pháp**: Tạo thủ công database bằng lệnh ở mục trên.

---

### ❌ Service `Exited (1)` ngay sau khi start

```bash
# Xem log để tìm nguyên nhân cụ thể
docker-compose logs --tail=30 {tên-service}
```

| Lỗi trong log | Nguyên nhân | Giải pháp |
|---------------|------------|-----------|
| `database "xxx" does not exist` | DB chưa được tạo | Tạo DB thủ công (xem trên) |
| `could not connect to server` | DB container chưa ready | Chờ 30s rồi restart service |
| `ModuleNotFoundError` | Package chưa install | `docker-compose up --build -d {service}` |
| `Port already in use` | Port bị chiếm | Đổi port host trong `docker-compose.yml` |
| `django.db.migrations.exceptions` | Migration lỗi | `docker-compose exec {svc} python manage.py migrate` |

---

### ❌ Services không kết nối được với nhau

```bash
# Kiểm tra tất cả containers có cùng network không
docker network inspect bookstore-microservice_bookstore-network

# Kiểm tra container có trong network không
docker network inspect bookstore-microservice_bookstore-network \
  --format='{{range .Containers}}{{.Name}} {{end}}'
```

Nếu service thiếu trong network, restart nó:
```bash
docker-compose up -d {tên-service}
```

---

### ❌ Lỗi MySQL `Access denied`

```bash
docker exec -it bookstore-microservice-mysql-db-1 mysql -u root -prootpassword -e \
  "ALTER USER 'bookstore_user'@'%' IDENTIFIED BY 'bookstore_pass'; FLUSH PRIVILEGES;"
```

---

### ❌ pgAdmin không kết nối được PostgreSQL

Cấu hình server trong pgAdmin:
- **Host**: `postgres-db` (nếu dùng từ container khác) HOẶC `localhost` (từ host machine)
- **Port**: `5432` (Docker internal) / `5433` (từ host)
- **Username**: `bookstore_user`
- **Password**: `bookstore_pass`

---

### 🔄 Reset toàn bộ hệ thống

```bash
# ⚠️ CẢNH BÁO: Xóa toàn bộ dữ liệu và volumes
docker-compose down -v --remove-orphans
docker system prune -f
docker-compose up --build -d
```

---

## 📌 Thông tin dự án

| Thuộc tính | Giá trị |
|------------|---------|
| **Kiến trúc** | Microservice Architecture |
| **Số lượng services** | 12 microservices |
| **Tổng containers** | 16 (12 service + MySQL + PostgreSQL + phpMyAdmin + pgAdmin) |
| **Backend framework** | Django 5.x + Django REST Framework 3.x |
| **Frontend** | Bootstrap 5.3.3 + Bootstrap Icons (Server-side rendering) |
| **Databases** | MySQL 8.0 (2 DBs) + PostgreSQL 15 (9 DBs) |
| **Containerization** | Docker Compose v2 |
| **Service communication** | HTTP REST (Docker internal DNS) |
| **Authentication** | Django Session-based (customer_id stored in PostgreSQL) |
| **AI/ML** | Hybrid recommendation (Content-based + Popularity-based) |
| **Tổng API endpoints** | ~55 endpoints trên 11 services |
| **Ngôn ngữ** | Python 3.10 |


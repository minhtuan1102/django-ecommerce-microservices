

| Ch??ng |     |       |     | 2   |     |            |     |     |
| ------ | --- | ----- | --- | --- | --- | ---------- | --- | --- |
| Ph�t   |     | tri?n |     |     | H?  | E-Commerce |     |     |
Microservices
| 2.1   | X�c            |            | ??nh | y�u |              | c?u |                    |          |
| ----- | -------------- | ---------- | ---- | --- | ------------ | --- | ------------------ | -------- |
| 2.1.1 |                | Functional |      |     | Requirements |     |                    |          |
|       | (cid:136) Qu?n | l� s?n     | ph?m | (?a | domain:      |     | book, electronics, | fashion) |
(cid:136)
|     | Qu?n | l� ng??i |     | d�ng | (admin, | staff, | customer) |     |
| --- | ---- | -------- | --- | ---- | ------- | ------ | --------- | --- |
(cid:136)
|     | Gi? | h�ng | (cart) |     |     |     |     |     |
| --- | --- | ---- | ------ | --- | --- | --- | --- | --- |
(cid:136)
|     | ??t | h�ng | (order) |     |     |     |     |     |
| --- | --- | ---- | ------- | --- | --- | --- | --- | --- |
(cid:136)
|       | Thanh          | to�n           | (payment)  |       |      |              |     |     |
| ----- | -------------- | -------------- | ---------- | ----- | ---- | ------------ | --- | --- |
|       | (cid:136) Giao | h�ng           | (shipping) |       |      |              |     |     |
|       | (cid:136) T�m  | ki?m           | v� g?i     | � s?n | ph?m |              |     |     |
| 2.1.2 |                | Non-functional |            |       |      | Requirements |     |     |
(cid:136)
|     | Scalability: |     | scale | t?ng | service | ??c | l?p |     |
| --- | ------------ | --- | ----- | ---- | ------- | --- | --- | --- |
(cid:136)
|       | High                       | Availability: |      | h?             | th?ng   | lu�n | s?n s�ng |     |
| ----- | -------------------------- | ------------- | ---- | -------------- | ------- | ---- | -------- | --- |
|       | (cid:136) Security:        |               | JWT, | authentication |         |      |          |     |
|       | (cid:136) Maintainability: |               |      | d?             | b?o tr� |      |          |     |
| 2.2   | Ph�n                       |               | r�   | h?             | th?ng   |      | theo DDD |     |
| 2.2.1 |                            | Bounded       |      | Context        |         |      |          |     |
(cid:136)
|     | User              | Context | ?   | user-service |                 |     |     |     |
| --- | ----------------- | ------- | --- | ------------ | --------------- | --- | --- | --- |
|     | (cid:136) Product | Context |     | ?            | product-service |     |     |     |
|     | (cid:136) Cart    | Context | ?   | cart-service |                 |     |     |     |
11

(cid:136)
|     | Order | Context ? | order-service |     |     |     |     |
| --- | ----- | --------- | ------------- | --- | --- | --- | --- |
(cid:136)
|       | Payment            | Context | ? payment-service  |     |     |     |     |
| ----- | ------------------ | ------- | ------------------ | --- | --- | --- | --- |
|       | (cid:136) Shipping | Context | ? shipping-service |     |     |     |     |
| 2.2.2 | Nguy�n             | t?c     |                    |     |     |     |     |
(cid:136)
|     | M?i context | = 1 | database | ri�ng |     |     |     |
| --- | ----------- | --- | -------- | ----- | --- | --- | --- |
(cid:136)
|       | Giao ti?p | qua REST | API     |      |         |          |     |
| ----- | --------- | -------- | ------- | ---- | ------- | -------- | --- |
| 2.3   | Thi?t     | k?       | Product |      | Service | (Django) |     |
| 2.3.1 | Ph�n      | lo?i     | s?n     | ph?m |         |          |     |
(cid:136)
|     | Book: | gi�o tr�nh, | ti?u | thuy?t |     |     |     |
| --- | ----- | ----------- | ---- | ------ | --- | --- | --- |
(cid:136)
|     | Electronics: | mobile, | laptop, |     | t? l?nh, ?i?u | h�a |     |
| --- | ------------ | ------- | ------- | --- | ------------- | --- | --- |
(cid:136)
|       | Fashion:                | �o, qu?n,                          | gi�y |     |     |     |                   |
| ----- | ----------------------- | ---------------------------------- | ---- | --- | --- | --- | ----------------- |
| 2.3.2 | Model                   | t?ng                               | qu�t |     |     |     |                   |
| class | Category(models.Model): |                                    |      |     |     |     |                   |
|       | name                    | = models.CharField(max_length=100) |      |     |     |     |                   |
| class | Product(models.Model):  |                                    |      |     |     |     |                   |
|       | name                    | = models.CharField(max_length=255) |      |     |     |     |                   |
|       | price                   | = models.FloatField()              |      |     |     |     |                   |
|       | stock                   | = models.IntegerField()            |      |     |     |     |                   |
|       | category                | = models.ForeignKey(Category,      |      |     |     |     | on_delete=models. |
CASCADE)
| 2.3.3 | Chi | ti?t theo | domain |     |     |     |     |
| ----- | --- | --------- | ------ | --- | --- | --- | --- |
Book
| class | Book(models.Model): |                                 |     |     |     |     |                   |
| ----- | ------------------- | ------------------------------- | --- | --- | --- | --- | ----------------- |
|       | product             | = models.OneToOneField(Product, |     |     |     |     | on_delete=models. |
CASCADE)
|     | author    | = models.CharField(max_length=255) |     |     |     |     |     |
| --- | --------- | ---------------------------------- | --- | --- | --- | --- | --- |
|     | publisher | = models.CharField(max_length=255) |     |     |     |     |     |
|     | isbn      | = models.CharField(max_length=20)  |     |     |     |     |     |
Electronics
| class | Electronics(models.Model): |                                 |     |     |     |     |                   |
| ----- | -------------------------- | ------------------------------- | --- | --- | --- | --- | ----------------- |
|       | product                    | = models.OneToOneField(Product, |     |     |     |     | on_delete=models. |
CASCADE)
|     | brand    | = models.CharField(max_length=100) |     |     |     |     |     |
| --- | -------- | ---------------------------------- | --- | --- | --- | --- | --- |
|     | warranty | = models.IntegerField()            |     |     |     |     |     |
12

Fashion
| class Fashion(models.Model): |                                 |     |     |     |                   |
| ---------------------------- | ------------------------------- | --- | --- | --- | ----------------- |
| product                      | = models.OneToOneField(Product, |     |     |     | on_delete=models. |
CASCADE)
| size      | = models.CharField(max_length=10) |     |     |     |     |
| --------- | --------------------------------- | --- | --- | --- | --- |
| color     | = models.CharField(max_length=50) |     |     |     |     |
| 2.3.4 API |                                   |     |     |     |     |
GET /products/
POST /products/
GET /products/{id}
| 2.4 Thi?t        | k?         | User      | Service | (Django) |     |
| ---------------- | ---------- | --------- | ------- | -------- | --- |
| 2.4.1 Ph�n       | lo?i       | ng??i     | d�ng    |          |     |
| (cid:136) Admin: | to�n quy?n | h?        | th?ng   |          |     |
| (cid:136) Staff: | x? l� ??n  | h�ng, v?n | h�nh    |          |     |
(cid:136)
| Customer:                       | mua h�ng     |              |     |                     |     |
| ------------------------------- | ------------ | ------------ | --- | ------------------- | --- |
| 2.4.2 Model                     |              |              |     |                     |     |
| from django.contrib.auth.models |              |              |     | import AbstractUser |     |
| class User(AbstractUser):       |              |              |     |                     |     |
| ROLE_CHOICES                    |              | = (          |     |                     |     |
|                                 | (�admin�,    | �Admin�),    |     |                     |     |
|                                 | (�staff�,    | �Staff�),    |     |                     |     |
|                                 | (�customer�, | �Customer�), |     |                     |     |
)
| role             | = models.CharField(max_length=20, |        |     |     | choices=ROLE_CHOICES) |
| ---------------- | --------------------------------- | ------ | --- | --- | --------------------- |
| 2.4.3 Ph�n       | quy?n                             | (RBAC) |     |     |                       |
| (cid:136) Admin: | CRUD to�n                         | b?     |     |     |                       |
(cid:136)
| Staff: | x? l� order, | shipping |     |     |     |
| ------ | ------------ | -------- | --- | --- | --- |
(cid:136)
| Customer: | mua h�ng, | xem | s?n ph?m |     |     |
| --------- | --------- | --- | -------- | --- | --- |
| 2.4.4 API |           |     |          |     |     |
POST /auth/register
POST /auth/login
GET /users/
13

| 2.5 Thi?t   | k? Cart | Service |     |
| ----------- | ------- | ------- | --- |
| 2.5.1 Model |         |         |     |
class Cart(models.Model):
| user_id | = models.IntegerField() |     |     |
| ------- | ----------------------- | --- | --- |
class CartItem(models.Model):
| cart =      | models.ForeignKey(Cart, |     | on_delete=models.CASCADE) |
| ----------- | ----------------------- | --- | ------------------------- |
| product_id  | = models.IntegerField() |     |                           |
| quantity    | = models.IntegerField() |     |                           |
| 2.5.2 Logic |                         |     |                           |
(cid:136)
| Add product      | v�o cart |     |     |
| ---------------- | -------- | --- | --- |
| (cid:136) Update | s? l??ng |     |     |
| (cid:136) Remove | item     |     |     |
| 2.5.3 API        |          |     |     |
POST /cart/add
GET /cart/
| DELETE /cart/remove |          |         |     |
| ------------------- | -------- | ------- | --- |
| 2.6 Thi?t           | k? Order | Service |     |
| 2.6.1 Model         |          |         |     |
class Order(models.Model):
| user_id     | = models.IntegerField()           |     |     |
| ----------- | --------------------------------- | --- | --- |
| total_price | = models.FloatField()             |     |     |
| status      | = models.CharField(max_length=50) |     |     |
class OrderItem(models.Model):
| order          | = models.ForeignKey(Order, |     | on_delete=models.CASCADE) |
| -------------- | -------------------------- | --- | ------------------------- |
| product_id     | = models.IntegerField()    |     |                           |
| quantity       | = models.IntegerField()    |     |                           |
| 2.6.2 Workflow |                            |     |                           |
(cid:136)
| T?o order | t? cart |     |     |
| --------- | ------- | --- | --- |
(cid:136)
| G?i request | sang payment-service |     |     |
| ----------- | -------------------- | --- | --- |
(cid:136)
| Sau khi | thanh to�n ? | shipping |     |
| ------- | ------------ | -------- | --- |
14

| 2.7 Thi?t | k? Payment | Service |
| --------- | ---------- | ------- |
2.7.1 Model
class Payment(models.Model):
| order_id    | = models.IntegerField()         |     |
| ----------- | ------------------------------- | --- |
| amount =    | models.FloatField()             |     |
| status =    | models.CharField(max_length=50) |     |
| 2.7.2 Tr?ng | th�i                            |     |
(cid:136) Pending
(cid:136) Success
(cid:136) Failed
2.7.3 API
POST /payment/pay
GET /payment/status
| 2.8 Thi?t | k? Shipping | Service |
| --------- | ----------- | ------- |
2.8.1 Model
class Shipment(models.Model):
| order_id    | = models.IntegerField()         |     |
| ----------- | ------------------------------- | --- |
| address     | = models.TextField()            |     |
| status =    | models.CharField(max_length=50) |     |
| 2.8.2 Tr?ng | th�i                            |     |
(cid:136)
Processing
(cid:136) Shipping
(cid:136) Delivered
2.8.3 API
POST /shipping/create
GET /shipping/status
15

| 2.9    |                 | Lu?ng |                    | h? th?ng            |      | t?ng         | th?      |      |     |
| ------ | --------------- | ----- | ------------------ | ------------------- | ---- | ------------ | -------- | ---- | --- |
|        | 1. User         | ??ng  | nh?p               | (user-service)      |      |              |          |      |     |
|        | 2. Xem          | s?n   | ph?m               | (product-service)   |      |              |          |      |     |
|        | 3. Th�m         |       | v�o gi?            | h�ng (cart-service) |      |              |          |      |     |
|        | 4. T?o          | ??n   | h�ng               | (order-service)     |      |              |          |      |     |
|        | 5. Thanh        |       | to�n               | (payment-service)   |      |              |          |      |     |
|        | 6. Giao         | h�ng  | (shipping-service) |                     |      |              |          |      |     |
| 2.10   |                 | H??ng |                    | d?n                 | th?c | h�nh         |          |      |     |
| 2.10.1 |                 | M?c   |                    | ti�u                |      |              |          |      |     |
| Sinh   | vi�n            | c?n:  |                    |                     |      |              |          |      |     |
|        | (cid:136) Thi?t | k?    | Class              | Diagram             | b?ng | Visual       | Paradigm | (VP) |     |
|        | (cid:136) X�y   | d?ng  | database           | cho                 | t?ng | microservice |          |      |     |
(cid:136)
|        | Mapping |       | t?       | Class Diagram |          | ? Database |          |             |          |
| ------ | ------- | ----- | -------- | ------------- | -------- | ---------- | -------- | ----------- | -------- |
| 2.10.2 |         | H??ng |          | d?n           | v? Class |            | Diagram  | b?ng Visual | Paradigm |
| B??c   | 1:      | X�c   | ??nh     | l?p (Classes) |          |            |          |             |          |
| Sinh   | vi�n    | x�c   | ??nh     | c�c l?p       | ch�nh    | theo t?ng  | service: |             |          |
|        | Product |       | Service: |               |          |            |          |             |          |
(cid:136)
Product
(cid:136)
Category
(cid:136) Book
(cid:136) Electronics
(cid:136) Fashion
|     | User | Service: |     |     |     |     |     |     |     |
| --- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- |
(cid:136)
User
(cid:136)
Role
|     | Order | Service: |     |     |     |     |     |     |     |
| --- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- |
(cid:136)
Order
(cid:136) OrderItem
16

B??c 2: X�c ??nh thu?c t�nh
V� d? l?p Product:
(cid:136)
id: int
(cid:136)
name: string
(cid:136)
price: float
(cid:136)
stock: int
B??c 3: X�c ??nh quan h? (Relationships)
(cid:136)
Association: Product ? Category
(cid:136)
Inheritance: Book, Electronics, Fashion k? th?a Product
(cid:136)
Composition: Order ch?a OrderItem
K� hi?u UML
(cid:136)
1..* (one-to-many)
(cid:136)
1..1 (one-to-one)
Y�u c?u b�i n?p
(cid:136)
Export s? ?? t? VP (PNG/PDF)
(cid:136)
C� ??y ?? class + relationship
2.10.3 Mapping Class Diagram sang Database
Nguy�n t?c
(cid:136)
Class ? Table
(cid:136)
Attribute ? Column
(cid:136)
Relationship ? Foreign Key
V� d?
Product(id, name, price)
Category(id, name)
=> Product.category_id (FK)
2.10.4 Thi?t k? Database cho t?ng Service
Nguy�n t?c Microservices
(cid:136)
M?i service c� database ri�ng v?i data model t??ng ?ng v?i Bi?u ?? l?p
(cid:136)
Kh�ng share database gi?a c�c service
17

| 1.  | Product | Service     | Database |     | (PostgreSQL) |     |
| --- | ------- | ----------- | -------- | --- | ------------ | --- |
| L�  | do ch?n | PostgreSQL: |          |     |              |     |
(cid:136)
|     | H?  | tr? t?t | JSON |     |     |     |
| --- | --- | ------- | ---- | --- | --- | --- |
(cid:136)
|        | Ph�  | h?p d?       | li?u ph?c | t?p  |     |     |
| ------ | ---- | ------------ | --------- | ---- | --- | --- |
| CREATE |      | TABLE        | category  | (    |     |     |
|        | id   | SERIAL       | PRIMARY   | KEY, |     |     |
|        | name | VARCHAR(100) |           |      |     |     |
);
| CREATE |             | TABLE         | product | (          |     |              |
| ------ | ----------- | ------------- | ------- | ---------- | --- | ------------ |
|        | id          | SERIAL        | PRIMARY | KEY,       |     |              |
|        | name        | VARCHAR(255), |         |            |     |              |
|        | price       | FLOAT,        |         |            |     |              |
|        | stock       | INT,          |         |            |     |              |
|        | category_id |               | INT     | REFERENCES |     | category(id) |
);
| CREATE |            | TABLE         | book | (       |      |     |
| ------ | ---------- | ------------- | ---- | ------- | ---- | --- |
|        | product_id |               | INT  | PRIMARY | KEY, |     |
|        | author     | VARCHAR(255), |      |         |      |     |
|        | isbn       | VARCHAR(20)   |      |         |      |     |
);
| 2.  | User Service |        | Database | (MySQL) |     |     |
| --- | ------------ | ------ | -------- | ------- | --- | --- |
| L�  | do ch?n      | MySQL: |          |         |     |     |
(cid:136)
|        | Ph?           | bi?n               |               |     |         |      |
| ------ | ------------- | ------------------ | ------------- | --- | ------- | ---- |
|        | (cid:136) Ph� | h?p authentication |               |     |         |      |
| CREATE |               | TABLE              | user          | (   |         |      |
|        | id            | INT AUTO_INCREMENT |               |     | PRIMARY | KEY, |
|        | username      |                    | VARCHAR(100), |     |         |      |
|        | password      |                    | VARCHAR(255), |     |         |      |
|        | role          | VARCHAR(20)        |               |     |         |      |
);
| 3.     | Cart Service |             |      |      |     |     |
| ------ | ------------ | ----------- | ---- | ---- | --- | --- |
| CREATE |              | TABLE       | cart | (    |     |     |
|        | id           | INT PRIMARY |      | KEY, |     |     |
|        | user_id      |             | INT  |      |     |     |
);
| CREATE |     | TABLE       | cart_item |      | (   |     |
| ------ | --- | ----------- | --------- | ---- | --- | --- |
|        | id  | INT PRIMARY |           | KEY, |     |     |
18

cart_id INT,
product_id INT,
quantity INT
);
4. Order Service
CREATE TABLE orders (
id INT PRIMARY KEY,
user_id INT,
total_price FLOAT,
status VARCHAR(50)
);
CREATE TABLE order_item (
id INT PRIMARY KEY,
order_id INT,
product_id INT,
quantity INT
);
5. Payment Service
CREATE TABLE payment (
id INT PRIMARY KEY,
order_id INT,
amount FLOAT,
status VARCHAR(50)
);
6. Shipping Service
CREATE TABLE shipment (
id INT PRIMARY KEY,
order_id INT,
address TEXT,
status VARCHAR(50)
);
2.10.5 So s�nh MySQL vs PostgreSQL
Ti�u ch� MySQL PostgreSQL
Hi?u n?ng T?t T?t
JSON Trung b�nh M?nh
Quan h? ph?c t?p Trung b�nh T?t
19

| 2.10.6 |                    | B�i t?p       |     |         |               |     |
| ------ | ------------------ | ------------- | --- | ------- | ------------- | --- |
|        | (cid:136) V? Class | Diagram       | cho | to�n b? | h? th?ng b?ng | VP  |
|        | (cid:136) Mapping  | sang database |     | schema  |               |     |
(cid:136)
|        | Tri?n                | khai database | b?ng      | MySQL/PostgreSQL |     |     |
| ------ | -------------------- | ------------- | --------- | ---------------- | --- | --- |
| 2.10.7 |                      | Checklist     | ?�nh      | gi�              |     |     |
|        | (cid:136) C� s?      | ?? class ?�ng | UML       |                  |     |     |
|        | (cid:136) C� mapping | r�            | r�ng sang | database         |     |     |
|        | (cid:136) Database   | t�ch ri�ng    | t?ng      | service          |     |     |
(cid:136)
|      | C� s?            | d?ng c? MySQL      |          | v� PostgreSQL |            |      |
| ---- | ---------------- | ------------------ | -------- | ------------- | ---------- | ---- |
| 2.11 |                  | K?t lu?n           |          |               |            |      |
|      | (cid:136) Ki?n   | tr�c microservices |          | gi�p h?       | th?ng linh | ho?t |
|      | (cid:136) Django | ph� h?p            | x�y d?ng | nhanh         |            |      |
(cid:136)
|     | DDD | gi�p thi?t | k? ?�ng | ngay | t? ??u |     |
| --- | --- | ---------- | ------- | ---- | ------ | --- |
20

Ch??ng 3
AI Service cho t? v?n s?n ph?m
3.1 M?c ti�u
X�y d?ng h? th?ng AI g?i � s?n ph?m d?a tr�n:
(cid:136)
H�nh vi ng??i d�ng (click, search, add-to-cart)
(cid:136)
Quan h? s?n ph?m (similarity)
(cid:136)
Ng? c?nh truy v?n (chatbot)
Output:
(cid:136)
Danh s�ch s?n ph?m ?? xu?t
(cid:136)
Chatbot t? v?n
3.2 Ki?n tr�c AI Service
AI Service ???c thi?t k? nh? m?t microservice ??c l?p:
(cid:136)
Input: user behavior, query
(cid:136)
Processing:
� LSTM model
� Knowledge Graph
� RAG
(cid:136)
Output: recommendation / chatbot response
3.3 Thu th?p d? li?u
3.3.1 User Behavior Data
(cid:136)
user_id
(cid:136)
product_id
(cid:136)
action (view, click, add_to_cart)
(cid:136)
timestamp
21

| 3.3.2    |      | V� d?                 | dataset |               |                 |                |            |
| -------- | ---- | --------------------- | ------- | ------------- | --------------- | -------------- | ---------- |
| user_id, |      | product_id,           |         | action,       | time            |                |            |
| 1,       | 101, | view,                 | t1      |               |                 |                |            |
| 1,       | 102, | add_to_cart,          |         | t2            |                 |                |            |
| 3.4      |      | M� h�nh               |         | LSTM          | (Sequence       | Modeling)      |            |
| 3.4.1    |      | � t??ng               |         |               |                 |                |            |
| D?       | ?o�n | s?n ph?m              | ti?p    | theo d?a      | tr�n chu?i h�nh | vi.            |            |
| 3.4.2    |      | Model                 | chi     | ti?t          |                 |                |            |
| import   |      | torch                 |         |               |                 |                |            |
| import   |      | torch.nn              | as      | nn            |                 |                |            |
| class    |      | LSTMModel(nn.Module): |         |               |                 |                |            |
|          | def  | __init__(self,        |         | input_dim=10, |                 | hidden_dim=64, | output_dim |
=100):
super().__init__()
|     |     | self.lstm |     | = nn.LSTM(input_dim, |     | hidden_dim, | batch_first= |
| --- | --- | --------- | --- | -------------------- | --- | ----------- | ------------ |
True)
|           |        | self.fc                                | =            | nn.Linear(hidden_dim, |     | output_dim) |     |
| --------- | ------ | -------------------------------------- | ------------ | --------------------- | --- | ----------- | --- |
|           | def    | forward(self,                          |              | x):                   |     |             |     |
|           |        | out, _                                 | =            | self.lstm(x)          |     |             |     |
|           |        | out =                                  | out[:,       | -1,                   | :]  |             |     |
|           |        | return                                 | self.fc(out) |                       |     |             |     |
| 3.4.3     |        | Training                               |              |                       |     |             |     |
| criterion |        | = nn.CrossEntropyLoss()                |              |                       |     |             |     |
| optimizer |        | = torch.optim.Adam(model.parameters()) |              |                       |     |             |     |
| for       | epoch  | in range(epochs):                      |              |                       |     |             |     |
|           | output | = model(x)                             |              |                       |     |             |     |
|           | loss   | = criterion(output,                    |              |                       | y)  |             |     |
loss.backward()
optimizer.step()
| 3.5   |     | Knowledge |     | Graph | v?i Neo4j |     |     |
| ----- | --- | --------- | --- | ----- | --------- | --- | --- |
| 3.5.1 |     | M� h�nh   | ??  | th?   |           |     |     |
(cid:136)
|     | Node: | User, | Product |     |     |     |     |
| --- | ----- | ----- | ------- | --- | --- | --- | --- |
(cid:136) Edge:
22

� BUY
� VIEW
� SIMILAR
3.5.2 V� d? Cypher
CREATE (u:User {id:1})
CREATE (p:Product {id:101})
CREATE (u)-[:BUY]->(p)
3.5.3 Truy v?n g?i �
MATCH (u:User {id:1})-[:BUY]->(p)-[:SIMILAR]->(rec)
RETURN rec
3.6 RAG (Retrieval-Augmented Generation)
3.6.1 Pipeline
(cid:136)
Retrieve:
� T�m s?n ph?m li�n quan t? DB / vector DB
(cid:136)
Generate:
� Sinh c�u tr? l?i b?ng LLM
3.6.2 Vector Database
(cid:136)
FAISS / ChromaDB
(cid:136)
Embedding t? m� t? s?n ph?m
3.6.3 V� d?
query = "laptop gaming"
results = vector_db.search(query)
response = LLM.generate(results)
3.7 K?t h?p Hybrid Model
(cid:136)
LSTM: d? ?o�n h�nh vi
(cid:136)
Graph: quan h? s?n ph?m
(cid:136)
RAG: hi?u ng? ngh?a
Final Recommendation:
final_score = w1 * lstm + w2 * graph + w3 * rag
23

| 3.8 Hai |                   | d?ng | AI  | Service |      |
| ------- | ----------------- | ---- | --- | ------- | ---- |
| 3.8.1   | 1. Recommendation |      |     |         | List |
Use cases
| (cid:136) Khi | search      |     |     |     |     |
| ------------- | ----------- | --- | --- | --- | --- |
| (cid:136) Khi | add-to-cart |     |     |     |     |
API
GET /recommend?user_id=1
Output
| [101, 102, |            | 205] |     |     |     |
| ---------- | ---------- | ---- | --- | --- | --- |
| 3.8.2      | 2. Chatbot |      | t?  | v?n |     |
Input
| "t�i c?n laptop |     | gi� r?" |     |     |     |
| --------------- | --- | ------- | --- | --- | --- |
Pipeline
| (cid:136) NLP      | hi?u | intent   |     |     |     |
| ------------------ | ---- | -------- | --- | --- | --- |
| (cid:136) Retrieve |      | s?n ph?m |     |     |     |
(cid:136)
| Generate |     | response |     |     |     |
| -------- | --- | -------- | --- | --- | --- |
API
POST /chatbot
Output
| "B?n c� th? | tham | kh?o  | Laptop | XYZ        | gi� 10 tri?u..." |
| ----------- | ---- | ----- | ------ | ---------- | ---------------- |
| 3.9 Tri?n   |      | khai  |        | AI Service |                  |
| 3.9.1       | Tech | stack |        |            |                  |
(cid:136)
| tensorflow/PyTorch |     |     | (LSTM) |     |     |
| ------------------ | --- | --- | ------ | --- | --- |
(cid:136)
| Neo4j             | (Graph) |           |     |     |     |
| ----------------- | ------- | --------- | --- | --- | --- |
| (cid:136) FAISS   | (Vector |           | DB) |     |     |
| (cid:136) FastAPI |         | (service) |     |     |     |
24

3.9.2 Ki?n tr�c
(cid:136)
AI service ??c l?p
(cid:136)
Giao ti?p v?i c�c service kh�c qua API
3.10 B�i t?p
(cid:136)
X�y d?ng model LSTM ??n gi?n
(cid:136)
T?o graph trong Neo4j
(cid:136)
Implement API recommendation
(cid:136)
X�y d?ng chatbot c? b?n
3.11 Checklist ?�nh gi�
(cid:136)
C� pipeline AI r� r�ng
(cid:136)
C� model (LSTM)
(cid:136)
C� Graph v� RAG
(cid:136)
C� API ho?t ??ng
3.12 K?t lu?n
(cid:136)
AI gi�p c� nh�n h�a tr?i nghi?m
(cid:136)
K?t h?p nhi?u m� h�nh cho hi?u qu? cao
(cid:136)
Ph� h?p h? e-commerce hi?n ??i
25

| Ch??ng |     |      | 4    |      |       |     |      |       |
| ------ | --- | ---- | ---- | ---- | ----- | --- | ---- | ----- |
| X�y    |     | d?ng |      | h?   | th?ng |     | ho�n | ch?nh |
| 4.1    |     | Ki?n | tr�c | t?ng |       | th? |      |       |
| 4.1.1  |     | M�   | h�nh | h?   | th?ng |     |      |       |
H? th?ng ???c x�y d?ng theo ki?n tr�c microservices, m?i service l� m?t Django project
??c l?p.
(cid:136)
|     | API | Gateway | (Nginx) |     |     |     |     |     |
| --- | --- | ------- | ------- | --- | --- | --- | --- | --- |
(cid:136)
|     | user-service              |     | (Django) |          |     |     |     |     |
| --- | ------------------------- | --- | -------- | -------- | --- | --- | --- | --- |
|     | (cid:136) product-service |     |          | (Django) |     |     |     |     |
|     | (cid:136) cart-service    |     | (Django) |          |     |     |     |     |
|     | (cid:136) order-service   |     | (Django) |          |     |     |     |     |
(cid:136)
|     | payment-service |     |     | (Django) |     |     |     |     |
| --- | --------------- | --- | --- | -------- | --- | --- | --- | --- |
(cid:136)
|     | shipping-service |     |     | (Django) |     |     |     |     |
| --- | ---------------- | --- | --- | -------- | --- | --- | --- | --- |
(cid:136)
|       | ai-service    |         | (FastAPI/Python) |     |       |     |     |     |
| ----- | ------------- | ------- | ---------------- | --- | ----- | --- | --- | --- |
| 4.1.2 |               | Nguy�n  |                  | t?c |       |     |     |     |
|       | (cid:136) M?i | service | c� database      |     | ri�ng |     |     |     |
(cid:136)
|     | Giao | ti?p | qua REST |     | API |     |     |     |
| --- | ---- | ---- | -------- | --- | --- | --- | --- | --- |
(cid:136)
|       | Kh�ng | truy     | c?p | DB           | c?a service | kh�c |     |     |
| ----- | ----- | -------- | --- | ------------ | ----------- | ---- | --- | --- |
| 4.2   |       | System   |     | Architecture |             |      |     |     |
| 4.2.1 |       | Overview |     |              |             |      |     |     |
The proposed system, named ecom-final, is designed as a fully distributed microservice-
based e-commerce platform. The architecture follows modern enterprise design principles,
| ensuring |     | scalability, | maintainability, |     |     | and fault | isolation. |     |
| -------- | --- | ------------ | ---------------- | --- | --- | --------- | ---------- | --- |
26

Each core business domain is implemented as an independent Django REST microser-
vice, while an API Gateway is employed to manage request routing, authentication, and
system-wide policies.
4.2.2 Microservice Architecture
The system consists of the following core services:
(cid:136)
User Service: Handles authentication, authorization, and user management.
(cid:136)
Product Service: Manages product catalog, categories, and inventory.
(cid:136)
Order Service: Processes customer orders and order lifecycle.
(cid:136)
Payment Service: Handles payment transactions and billing.
(cid:136)
Notification Service: Sends asynchronous notifications (email, SMS).
Each service is independently deployable and maintains its own database, following
the principle of database-per-service.
4.2.3 API Gateway
An API Gateway layer is introduced as the single entry point for all client requests. The
gateway is responsible for:
(cid:136)
Routing incoming requests to appropriate microservices
(cid:136)
Handling authentication using JSON Web Tokens (JWT)
(cid:136)
Enforcing rate limiting and security policies
(cid:136)
Logging and monitoring API usage
In this system, the API Gateway is implemented using NGINX as a reverse proxy.
4.2.4 Service Communication
The system adopts a hybrid communication strategy:
(cid:136)
Synchronous communication: RESTful APIs over HTTP for real-time opera-
tions (e.g., order validation).
(cid:136)
Asynchronous communication: Message queues (e.g., Redis, RabbitMQ) for
event-driven workflows.
For example, when an order is created, an event is published and consumed by the
payment and notification services.
4.2.5 Containerization and Deployment
All services are containerized using Docker to ensure consistency across environments.
The system is orchestrated using Docker Compose for development and can be extended
to Kubernetes for production deployment.
27

| 4.2.6 |     | System | Structure |     |     |     |     |     |     |     |
| ----- | --- | ------ | --------- | --- | --- | --- | --- | --- | --- | --- |
ecom-final/
|-- gateway/
| |   | |--              | nginx.conf |     |     |     | //TH?    |      | HI?N QUAN | TR?NG    |     |
| --- | ---------------- | ---------- | --- | --- | --- | -------- | ---- | --------- | -------- | --- |
| |-- | user-service/    |            |     |     |     | //staff, |      | admin,    | customer |     |
| |-- | product-service/ |            |     |     |     | //10     | nh�m | lo?i      | s?n ph?m |     |
|-- cart-service/
|-- order-service/
|-- payment-service/
|-- ai-service/
|-- infrastructure/
| |     | |--      | docker-compose.yml |            |              |              |               |     |                |     |        |
| ----- | -------- | ------------------ | ---------- | ------------ | ------------ | ------------- | --- | -------------- | --- | ------ |
|       |          | H�nh               | 4.1:       | Microservice | architecture |               | of  | the ecom-final |     | system |
| 4.2.7 |          | Design             | Principles |              |              |               |     |                |     |        |
| The   | proposed | architecture       |            | adheres      | to           | the following |     | principles:    |     |        |
(cid:136) Loose Coupling: Services interact only through APIs or messaging systems.
(cid:136) High Cohesion: Each service encapsulates a single business domain.
|     | (cid:136) Scalability: |     | Services | can | be scaled | independently. |     |     |     |     |
| --- | ---------------------- | --- | -------- | --- | --------- | -------------- | --- | --- | --- | --- |
(cid:136)
|          | Fault | Isolation: |                | Failure | in one | service | does not | affect | others. |     |
| -------- | ----- | ---------- | -------------- | ------- | ------ | ------- | -------- | ------ | ------- | --- |
| 4.2.8    |       | Security   | Considerations |         |        |         |          |        |         |     |
| Security | is    | enforced   | through:       |         |        |         |          |        |         |     |
(cid:136)
|     | JWT-based |     | authentication |     |     |     |     |     |     |     |
| --- | --------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
(cid:136)
|     | API | Gateway | validation |     |     |     |     |     |     |     |
| --- | --- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
(cid:136)
|       | Role-based |            | access | control | (RBAC) |     |     |     |     |     |
| ----- | ---------- | ---------- | ------ | ------- | ------ | --- | --- | --- | --- | --- |
| 4.2.9 |            | Discussion |        |         |        |     |     |     |     |     |
Compared to monolithic architectures, the proposed microservice design significantly im-
proves system flexibility and scalability. However, it introduces additional complexity in
deployment and service coordination, which is mitigated through containerization and
| standardized |     | communication |     | protocols. |     |     |     |     |     |     |
| ------------ | --- | ------------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
28

4.3 API Gateway (Nginx)
4.3.1 Vai tr�
(cid:136)
Entry point cho to�n h? th?ng
(cid:136)
Routing request ??n ?�ng service
(cid:136)
X? l� authentication
4.3.2 C?u h�nh m?u
location /users/ {
proxy_pass http://user-service:8000;
}
location /products/ {
proxy_pass http://product-service:8001;
}
4.4 Authentication (JWT)
4.4.1 C�i ??t
pip install djangorestframework-simplejwt
4.4.2 C?u h�nh
from rest_framework_simplejwt.views import TokenObtainPairView
4.4.3 Lu?ng
(cid:136)
User login ? nh?n token
(cid:136)
G?i token trong header
(cid:136)
C�c service verify token
4.5 Giao ti?p gi?a c�c Service
4.5.1 REST API call
import requests
response = requests.get(
"http://product-service:8001/products/"
)
29

| 4.5.2 | Best | Practice |     |     |
| ----- | ---- | -------- | --- | --- |
(cid:136)
Timeout
(cid:136)
Retry
(cid:136)
| Circuit | breaker    | (advanced) |          |     |
| ------- | ---------- | ---------- | -------- | --- |
| 4.6     | Docker     | h�a        | h? th?ng |     |
| 4.6.1   | Dockerfile | (Django)   |          |     |
FROM python:3.10
| WORKDIR        | /app               |                     |              |                 |
| -------------- | ------------------ | ------------------- | ------------ | --------------- |
| COPY .         | .                  |                     |              |                 |
| RUN pip        | install            | -r requirements.txt |              |                 |
| CMD ["python", |                    | "manage.py",        | "runserver", | "0.0.0.0:8000"] |
| 4.6.2          | docker-compose.yml |                     |              |                 |
| version:       | �3�                |                     |              |                 |
services:
user-service:
| build: | ./user-service |     |     |     |
| ------ | -------------- | --- | --- | --- |
ports:
- "8000:8000"
product-service:
| build: | ./product-service |     |     |     |
| ------ | ----------------- | --- | --- | --- |
ports:
- "8001:8001"
| 4.7      | Lu?ng                 | h? th?ng           | (End-to-End) |     |
| -------- | --------------------- | ------------------ | ------------ | --- |
| 4.7.1    | Use case:             | Mua                | h�ng         |     |
| 1. User  | login                 | (user-service)     |              |     |
| 2. Xem   | s?n ph?m              | (product-service)  |              |     |
| 3. Add   | to cart               | (cart-service)     |              |     |
| 4. T?o   | order (order-service) |                    |              |     |
| 5. Thanh | to�n                  | (payment-service)  |              |     |
| 6. Giao  | h�ng                  | (shipping-service) |              |     |
30

| 4.7.2 | Sequence |     | logic |     |     |
| ----- | -------- | --- | ----- | --- | --- |
(cid:136)
| order-service |     | g?i | payment-service |     |     |
| ------------- | --- | --- | --------------- | --- | --- |
(cid:136)
| payment     |            | success | ? g?i      | shipping-service |            |
| ----------- | ---------- | ------- | ---------- | ---------------- | ---------- |
| 4.8         | Tri?n      | khai    | Kubernetes |                  | (Optional) |
| 4.8.1       | Deployment |         |            |                  |            |
| apiVersion: |            | apps/v1 |            |                  |            |
| kind:       | Deployment |         |            |                  |            |
metadata:
| name: | user-service |     |     |     |     |
| ----- | ------------ | --- | --- | --- | --- |
| 4.8.2 | Service      |     |     |     |     |
| kind: | Service      |     |     |     |     |
spec:
| type: | ClusterIP |     |               |     |     |
| ----- | --------- | --- | ------------- | --- | --- |
| 4.9   | Logging   |     | v� Monitoring |     |     |
(cid:136)
| Logging: |     | ELK stack |     |     |     |
| -------- | --- | --------- | --- | --- | --- |
(cid:136)
| Monitoring:        |      | Prometheus |     | + Grafana |     |
| ------------------ | ---- | ---------- | --- | --------- | --- |
| 4.10               | ?�nh | gi�        | h?  | th?ng     |     |
| 4.10.1             | Hi?u | n?ng       |     |           |     |
| (cid:136) Response |      | time       |     |           |     |
(cid:136)
Throughput
| 4.10.2          | Kh?       | n?ng    | m?  | r?ng |     |
| --------------- | --------- | ------- | --- | ---- | --- |
| (cid:136) Scale | t?ng      | service |     |      |     |
| (cid:136) Load  | balancing |         |     |      |     |
| 4.10.3          | ?u        | ?i?m    |     |      |     |
(cid:136)
| Linh | ho?t |     |     |     |     |
| ---- | ---- | --- | --- | --- | --- |
(cid:136)
| D?  | m? r?ng |     |     |     |     |
| --- | ------- | --- | --- | --- | --- |
31

4.10.4 Nh??c ?i?m
(cid:136)
Ph?c t?p tri?n khai
(cid:136)
Debug kh�
4.11 B�i t?p th?c h�nh
(cid:136)
Tri?n khai c�c service b?ng Django
(cid:136)
K?t n?i qua API
(cid:136)
Docker h�a h? th?ng
(cid:136)
Test full flow mua h�ng + k?t qu? t? v?n
4.12 Checklist ?�nh gi�
(cid:136)
C� API Gateway
(cid:136)
C� JWT Auth
(cid:136)
C� Docker ch?y ???c
(cid:136)
C� flow order ? payment ? shipping
32

K?t lu?n
(cid:136)
| Microservices | ph� | h?p h? th?ng l?n |
| ------------- | --- | ---------------- |
(cid:136)
| DDD gi�p | thi?t k? | r� r�ng |
| -------- | -------- | ------- |
(cid:136)
| AI n�ng | cao tr?i nghi?m | ng??i d�ng |
| ------- | --------------- | ---------- |
33

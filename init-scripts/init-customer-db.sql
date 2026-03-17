CREATE DATABASE IF NOT EXISTS customer_db;
GRANT ALL PRIVILEGES ON customer_db.* TO 'bookstore_user'@'%';
FLUSH PRIVILEGES;

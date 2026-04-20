"""Seed data for product and category hierarchy."""

CATEGORY_TREE = {
    "Product": {
        "Electronics": ["Mobile", "Laptop", "Computer", "Tablet", "Accessory", "Audio"],
        "Fashion": ["Men", "Women", "Footwear"],
        "Home & Living": ["Kitchen", "Furniture", "Cleaning"],
        "Beauty": ["Skincare", "Makeup", "Personal Care"],
        "Sports": ["Fitness", "Outdoor"],
        "Books": ["Programming", "Business", "Self-help"],
    }
}

SAMPLE_PRODUCTS = [
    {"sku": "MOB-001", "name": "iPhone 15 Pro Max 256GB", "item_type": "mobile", "category": "Mobile", "price": 29900000, "stock": 12},
    {"sku": "MOB-002", "name": "Samsung Galaxy S24 Ultra", "item_type": "mobile", "category": "Mobile", "price": 27900000, "stock": 10},
    {"sku": "MOB-003", "name": "Xiaomi 14 Pro", "item_type": "mobile", "category": "Mobile", "price": 18900000, "stock": 9},
    {"sku": "LAP-001", "name": "MacBook Pro M3 14", "item_type": "laptop", "category": "Laptop", "price": 40900000, "stock": 6},
    {"sku": "LAP-002", "name": "Dell XPS 15", "item_type": "laptop", "category": "Laptop", "price": 35900000, "stock": 7},
    {"sku": "LAP-003", "name": "Lenovo ThinkPad X1 Carbon", "item_type": "laptop", "category": "Laptop", "price": 38900000, "stock": 8},
    {"sku": "COM-001", "name": "Gaming PC Ryzen 7 RTX 4070", "item_type": "computer", "category": "Computer", "price": 32900000, "stock": 5},
    {"sku": "COM-002", "name": "Workstation Intel i9 RTX 4080", "item_type": "computer", "category": "Computer", "price": 45900000, "stock": 3},
    {"sku": "TAB-001", "name": "iPad Air 11", "item_type": "tablet", "category": "Tablet", "price": 17900000, "stock": 11},
    {"sku": "AUD-001", "name": "Sony WH-1000XM5", "item_type": "audio", "category": "Audio", "price": 7990000, "stock": 18},
    {"sku": "AUD-002", "name": "JBL Flip 6", "item_type": "audio", "category": "Audio", "price": 2990000, "stock": 24},
    {"sku": "ACC-001", "name": "Mechanical Keyboard K8", "item_type": "accessory", "category": "Accessory", "price": 1590000, "stock": 30},
    {"sku": "FAS-001", "name": "Men Premium Oxford Shirt", "item_type": "fashion", "category": "Men", "price": 690000, "stock": 45},
    {"sku": "FAS-002", "name": "Women Linen Blazer", "item_type": "fashion", "category": "Women", "price": 990000, "stock": 38},
    {"sku": "FAS-003", "name": "Running Sneakers AirFlow", "item_type": "fashion", "category": "Footwear", "price": 1490000, "stock": 42},
    {"sku": "HOM-001", "name": "Air Fryer 6L Digital", "item_type": "home", "category": "Kitchen", "price": 2390000, "stock": 26},
    {"sku": "HOM-002", "name": "Stainless Cookware Set 10pcs", "item_type": "home", "category": "Kitchen", "price": 3290000, "stock": 16},
    {"sku": "HOM-003", "name": "Ergonomic Office Chair", "item_type": "home", "category": "Furniture", "price": 4290000, "stock": 14},
    {"sku": "HOM-004", "name": "Solid Wood Work Desk", "item_type": "home", "category": "Furniture", "price": 3890000, "stock": 12},
    {"sku": "HOM-005", "name": "Robot Vacuum Cleaner Pro", "item_type": "home", "category": "Cleaning", "price": 6990000, "stock": 9},
    {"sku": "BEA-001", "name": "Vitamin C Brightening Serum 30ml", "item_type": "beauty", "category": "Skincare", "price": 450000, "stock": 55},
    {"sku": "BEA-002", "name": "Hydrating Gel Moisturizer", "item_type": "beauty", "category": "Skincare", "price": 390000, "stock": 47},
    {"sku": "BEA-003", "name": "Longwear Matte Foundation", "item_type": "beauty", "category": "Makeup", "price": 520000, "stock": 33},
    {"sku": "BEA-004", "name": "Sonic Electric Toothbrush", "item_type": "beauty", "category": "Personal Care", "price": 890000, "stock": 27},
    {"sku": "SPT-001", "name": "Adjustable Dumbbell 20kg", "item_type": "sports", "category": "Fitness", "price": 2590000, "stock": 19},
    {"sku": "SPT-002", "name": "Yoga Mat Pro 8mm", "item_type": "sports", "category": "Fitness", "price": 420000, "stock": 60},
    {"sku": "SPT-003", "name": "Waterproof Hiking Backpack 35L", "item_type": "sports", "category": "Outdoor", "price": 790000, "stock": 28},
    {"sku": "BOO-001", "name": "Clean Code", "item_type": "book", "category": "Programming", "price": 189000, "stock": 70},
    {"sku": "BOO-002", "name": "Designing Data-Intensive Applications", "item_type": "book", "category": "Programming", "price": 299000, "stock": 52},
    {"sku": "BOO-003", "name": "The Lean Startup", "item_type": "book", "category": "Business", "price": 169000, "stock": 64},
    {"sku": "BOO-004", "name": "Atomic Habits", "item_type": "book", "category": "Self-help", "price": 189000, "stock": 80},
]

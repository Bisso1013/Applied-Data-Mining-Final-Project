import json
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

NUM_ORDERS = 200
STATUSES = ["delivered", "in-transit", "delayed", "returned", "cancelled"]

# Real Flipkart products — one entry per catalog CSV category
PRODUCTS = [
    # ── Laptops (flipkart_Laptops.csv) ────────────────────────────────────────
    {"product_id": "LAP-001", "name": "Lenovo IdeaPad 3 Core i3 11th Gen (8GB/256GB SSD)", "price": 445.99},
    {"product_id": "LAP-002", "name": "ASUS VivoBook 15 Core i3 10th Gen (8GB/512GB SSD)", "price": 399.99},
    {"product_id": "LAP-003", "name": "DELL Inspiron Core i3 11th Gen (8GB/1TB+256GB SSD)", "price": 469.99},
    {"product_id": "LAP-004", "name": "HP 14s Core i3 11th Gen (8GB/512GB SSD)", "price": 499.99},
    {"product_id": "LAP-005", "name": "Lenovo IdeaPad 3 Ryzen 5 (8GB/512GB SSD)", "price": 549.99},
    # ── Mobile Phones (flipkart_mobiles.csv) ──────────────────────────────────
    {"product_id": "MOB-001", "name": "SAMSUNG Galaxy F13 (Waterfall Blue, 64GB)", "price": 144.99},
    {"product_id": "MOB-002", "name": "POCO C31 (Shadow Gray, 64GB)", "price": 89.99},
    {"product_id": "MOB-003", "name": "REDMI 9i Sport (Coral Green, 64GB)", "price": 85.99},
    {"product_id": "MOB-004", "name": "REDMI 10 (Caribbean Green, 64GB)", "price": 114.99},
    {"product_id": "MOB-005", "name": "POCO M4 Pro (Power Black, 128GB)", "price": 144.99},
    # ── Refrigerators (flipkart_refrigerator.csv) ─────────────────────────────
    {"product_id": "REF-001", "name": "SAMSUNG 192L Direct Cool Single Door 3 Star Refrigerator", "price": 191.99},
    {"product_id": "REF-002", "name": "SAMSUNG 253L Frost Free Double Door 3 Star Refrigerator", "price": 294.99},
    {"product_id": "REF-003", "name": "Godrej 185L Direct Cool Single Door 4 Star Refrigerator", "price": 168.99},
    {"product_id": "REF-004", "name": "SAMSUNG 198L Direct Cool Single Door 4 Star Refrigerator", "price": 198.99},
    # ── Smart Watches (flipkart_smart_watch.csv) ──────────────────────────────
    {"product_id": "SWT-001", "name": "Noise ColorFit Caliber Go 1.69 inch HD Smartwatch", "price": 24.99},
    {"product_id": "SWT-002", "name": "boAt Storm Pro Bluetooth Calling 1.78 AMOLED Smartwatch", "price": 39.99},
    {"product_id": "SWT-003", "name": "Fire-Boltt Hurricane Smartwatch", "price": 21.99},
    {"product_id": "SWT-004", "name": "Amazfit GTR 2 1.39 AMOLED Bluetooth Smartwatch", "price": 72.99},
    {"product_id": "SWT-005", "name": "Noise ColorFit Icon 2 1.8 inch Bluetooth Calling Smartwatch", "price": 26.99},
    # ── Televisions (flipkart_tv.csv) ─────────────────────────────────────────
    {"product_id": "TV-001", "name": "Thomson 9R PRO 43 inch 4K LED Smart Android TV", "price": 241.99},
    {"product_id": "TV-002", "name": "SAMSUNG Crystal 4K Neo 55 inch LED Smart Tizen TV", "price": 567.99},
    {"product_id": "TV-003", "name": "SAMSUNG 32 inch HD Ready LED Smart Tizen TV", "price": 156.99},
    {"product_id": "TV-004", "name": "Mi 5A 32 inch HD Ready LED Smart Android TV", "price": 156.99},
    {"product_id": "TV-005", "name": "LG 32 inch HD Ready LED Smart TV", "price": 280.99},
    {"product_id": "TV-006", "name": "realme 32 inch HD Ready LED Smart Android TV", "price": 144.99},
    # ── Washing Machines (flipkart_washing_machine.csv) ───────────────────────
    {"product_id": "WM-001", "name": "Whirlpool 7kg Magic Clean 5 Star Fully Automatic Top Load Washing Machine", "price": 192.99},
    {"product_id": "WM-002", "name": "SAMSUNG 6.5kg Diamond Drum Fully Automatic Top Load Washing Machine", "price": 175.99},
    {"product_id": "WM-003", "name": "realme TechLife 7kg Semi Automatic Top Load Washing Machine", "price": 94.99},
    {"product_id": "WM-004", "name": "Whirlpool 7.5kg 5 Star Fully Automatic Top Load Washing Machine", "price": 138.99},
    {"product_id": "WM-005", "name": "MarQ 6.5kg Semi Automatic Top Load Washing Machine", "price": 82.99},
]

# Fixed demo order — preserved across regenerations for live demonstrations.
# Status "delayed" triggers the 15% delay refund policy in store_policies.md.
DEMO_ORDER = {
    "order_id": "07b571fd-f56c-4432-b236-093a8c0cc94c",
    "customer_id": "91a25c7c-40e8-42aa-8f8d-6c8dd90e9ffd",
    "timestamp": "2026-05-27T18:23:37.358635",
    "status": "delayed",
    "total_amount": 567.99,
    "items": [
        {
            "product_id": "TV-002",
            "name": "SAMSUNG Crystal 4K Neo 55 inch LED Smart Tizen TV",
            "price": 567.99
        }
    ],
    "shipping_address": {
        "street": "70454 Jennifer Ferry",
        "city": "Obrientown",
        "state": "CT",
        "zip_code": "04576"
    }
}


def generate_mock_orders(num_orders: int) -> list:
    orders = [DEMO_ORDER]

    for _ in range(num_orders - 1):
        num_items = random.randint(1, 2)
        selected = random.sample(PRODUCTS, k=num_items)
        order_items = [
            {"product_id": p["product_id"], "name": p["name"], "price": p["price"]}
            for p in selected
        ]
        total_amount = round(sum(item["price"] for item in order_items), 2)
        timestamp = datetime.now() - timedelta(days=random.randint(1, 60))

        orders.append({
            "order_id": fake.uuid4(),
            "customer_id": fake.uuid4(),
            "timestamp": timestamp.isoformat(),
            "status": random.choice(STATUSES),
            "total_amount": total_amount,
            "items": order_items,
            "shipping_address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip_code": fake.zipcode()
            }
        })

    return orders


if __name__ == "__main__":
    mock_orders = generate_mock_orders(NUM_ORDERS)
    with open("mock_orders.json", "w") as f:
        json.dump(mock_orders, f, indent=4)
    print(f"Generated {NUM_ORDERS} orders saved to mock_orders.json.")
    categories = {}
    for o in mock_orders:
        for item in o["items"]:
            cat = item["product_id"].split("-")[0]
            categories[cat] = categories.get(cat, 0) + 1
    print("Items by category:", categories)

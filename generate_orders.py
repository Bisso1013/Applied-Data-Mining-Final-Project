import json
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Configuration as per project requirements
NUM_ORDERS = 200
STATUSES = ["delivered", "in-transit", "delayed", "returned", "cancelled"]

# Sample product list for e-commerce context
PRODUCTS = [
    {"product_id": "P-1001", "name": "Wireless Noise-Canceling Headphones", "price": 199.99},
    {"product_id": "P-1002", "name": "Mechanical Gaming Keyboard", "price": 89.99},
    {"product_id": "P-1003", "name": "27-inch 4K Monitor", "price": 349.99},
    {"product_id": "P-1004", "name": "Ergonomic Mouse", "price": 45.00},
    {"product_id": "P-1005", "name": "1TB NVMe SSD", "price": 120.00},
    {"product_id": "P-1006", "name": "USB-C Hub", "price": 25.99}
]


def generate_mock_orders(num_orders):
    orders = []

    for _ in range(num_orders):
        # Generate 1 to 3 random items per order
        num_items = random.randint(1, 3)
        order_items = random.sample(PRODUCTS, k=num_items)

        # Calculate order total
        total_amount = round(sum(item["price"] for item in order_items), 2)

        # Generate a realistic timestamp within the last 60 days
        random_days_ago = random.randint(1, 60)
        timestamp = datetime.now() - timedelta(days=random_days_ago)

        order = {
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
        }
        orders.append(order)

    return orders


if __name__ == "__main__":
    # Generate the orders
    mock_orders = generate_mock_orders(NUM_ORDERS)

    # Save to JSON file
    output_filename = "mock_orders.json"
    with open(output_filename, "w") as json_file:
        json.dump(mock_orders, json_file, indent=4)

    print(f"Successfully generated {NUM_ORDERS} synthetic orders and saved to {output_filename}.")
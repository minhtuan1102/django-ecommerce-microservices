import pika
import json
import os
import sys
import django

# Setup Django environment
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cart_service.settings')
django.setup()

from app.models import Cart

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')

def callback(ch, method, properties, body):
    data = json.loads(body)
    customer_id = data.get('customer_id')
    if customer_id:
        # Create cart for new customer
        Cart.objects.get_or_create(customer_id=customer_id)
        print(f"Created cart for customer {customer_id}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def start_consuming():
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.queue_declare(queue='customer_created_queue', durable=True)
        channel.basic_consume(queue='customer_created_queue', on_message_callback=callback)
        print('Cart service is waiting for messages...')
        channel.start_consuming()
    except Exception as e:
        print(f"Failed to start consuming: {e}")

if __name__ == "__main__":
    start_consuming()

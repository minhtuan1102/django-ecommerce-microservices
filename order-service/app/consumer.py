import pika
import json
import os
import sys
import django

# Setup Django environment
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'order_service.settings')
django.setup()

from app.models import Order

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')

def callback_success(ch, method, properties, body):
    data = json.loads(body)
    order_id = data.get('order_id')
    try:
        order = Order.objects.get(id=order_id)
        order.status = 'confirmed'
        order.save()
        print(f"Order {order_id} confirmed by Saga success")
    except Order.DoesNotExist:
        pass
    ch.basic_ack(delivery_tag=method.delivery_tag)

import requests

BOOK_SERVICE_URL = os.environ.get('BOOK_SERVICE_URL', 'http://book-service:8002')

def callback_failed(ch, method, properties, body):
    data = json.loads(body)
    order_id = data.get('order_id')
    reason = data.get('reason')
    try:
        order = Order.objects.get(id=order_id)
        if order.status != 'cancelled':
            order.status = 'cancelled'
            order.save()
            print(f"Order {order_id} cancelled due to Saga failure: {reason}")
            
            # Compensation: Restore stock
            for item in order.items.all():
                try:
                    requests.post(
                        f"{BOOK_SERVICE_URL}/books/{item.book_id}/restore-stock/",
                        json={"quantity": item.quantity},
                        timeout=3
                    )
                    print(f"Restored stock for book {item.book_id}, quantity {item.quantity}")
                except Exception as e:
                    print(f"Failed to restore stock for book {item.book_id}: {e}")
                    
    except Order.DoesNotExist:
        pass
    ch.basic_ack(delivery_tag=method.delivery_tag)

def start_consuming():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    
    # Listen for success
    channel.queue_declare(queue='shipping_reserved_queue', durable=True)
    channel.basic_consume(queue='shipping_reserved_queue', on_message_callback=callback_success)
    
    # Listen for failure
    channel.queue_declare(queue='payment_failed_queue', durable=True)
    channel.basic_consume(queue='payment_failed_queue', on_message_callback=callback_failed)
    
    channel.queue_declare(queue='shipping_failed_queue', durable=True)
    channel.basic_consume(queue='shipping_failed_queue', on_message_callback=callback_failed)
    
    print('Order service is waiting for Saga results...')
    channel.start_consuming()

if __name__ == "__main__":
    start_consuming()

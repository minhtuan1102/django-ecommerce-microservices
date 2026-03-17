import pika
import json
import os
import sys
import django

# Setup Django environment
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ship_service.settings')
django.setup()

from app.models import Shipment

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')
FORCE_SHIPPING_FAIL = os.environ.get('FORCE_SHIPPING_FAIL', 'false').lower() == 'true'

def publish_shipping_reserved(data):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue='shipping_reserved_queue', durable=True)
    channel.basic_publish(
        exchange='',
        routing_key='shipping_reserved_queue',
        body=json.dumps(data),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    connection.close()

def publish_shipping_failed(data):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue='shipping_failed_queue', durable=True)
    channel.basic_publish(
        exchange='',
        routing_key='shipping_failed_queue',
        body=json.dumps(data),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    connection.close()

def callback(ch, method, properties, body):
    data = json.loads(body)
    order_id = data.get('order_id')
    address = data.get('shipping_address')

    try:
        if FORCE_SHIPPING_FAIL:
            raise RuntimeError('Forced shipping failure for fault simulation')

        # Create shipment record
        shipment = Shipment.objects.create(
            order_id=order_id,
            shipping_address=address,
            carrier='ghn',
            status='preparing'
        )

        # Accept normal user addresses; only fail when address is truly empty/invalid.
        normalized_address = (address or "").strip()
        if len(normalized_address) >= 8:
            shipment.status = 'shipped'
            shipment.save()
            publish_shipping_reserved(data)
            print(f"Shipping reserved for order {order_id}")
        else:
            shipment.status = 'cancelled'
            shipment.save()
            publish_shipping_failed({"order_id": order_id, "reason": "Invalid or empty shipping address"})
            print(f"Shipping failed for order {order_id}")
            
    except Exception as e:
        print(f"Error processing shipping: {e}")
        publish_shipping_failed({"order_id": order_id, "reason": str(e)})

    ch.basic_ack(delivery_tag=method.delivery_tag)

def start_consuming():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue='payment_reserved_queue', durable=True)
    channel.basic_consume(queue='payment_reserved_queue', on_message_callback=callback)
    print('Ship service is waiting for payment_reserved...')
    channel.start_consuming()

if __name__ == "__main__":
    start_consuming()

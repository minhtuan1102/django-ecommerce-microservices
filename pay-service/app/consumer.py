import pika
import json
import os
import sys
import django

# Setup Django environment
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pay_service.settings')
django.setup()

from app.models import Payment

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')
FORCE_PAYMENT_FAIL = os.environ.get('FORCE_PAYMENT_FAIL', 'false').lower() == 'true'

def publish_payment_reserved(data):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue='payment_reserved_queue', durable=True)
    channel.basic_publish(
        exchange='',
        routing_key='payment_reserved_queue',
        body=json.dumps(data),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    connection.close()

def publish_payment_failed(data):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue='payment_failed_queue', durable=True)
    channel.basic_publish(
        exchange='',
        routing_key='payment_failed_queue',
        body=json.dumps(data),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    connection.close()

def callback(ch, method, properties, body):
    data = json.loads(body)
    order_id = data.get('order_id')
    amount = data.get('amount')
    customer_id = data.get('customer_id')
    method_val = data.get('payment_method')

    try:
        if FORCE_PAYMENT_FAIL:
            raise RuntimeError('Forced payment failure for fault simulation')

        # Create payment record
        payment = Payment.objects.create(
            order_id=order_id,
            customer_id=customer_id,
            amount=amount,
            method=method_val,
            status='pending'
        )
        
        # In a real system, we would call a bank API here.
        # We now accept all payments to simulate a normal, working system.
        payment.status = 'completed'
        payment.save()
        publish_payment_reserved(data)
        print(f"Payment completed successfully for order {order_id}")
            
    except Exception as e:
        print(f"Error processing payment: {e}")
        publish_payment_failed({"order_id": order_id, "reason": str(e)})

    ch.basic_ack(delivery_tag=method.delivery_tag)

def start_consuming():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue='order_created_queue', durable=True)
    channel.basic_consume(queue='order_created_queue', on_message_callback=callback)
    print('Pay service is waiting for order_created...')
    channel.start_consuming()

if __name__ == "__main__":
    start_consuming()

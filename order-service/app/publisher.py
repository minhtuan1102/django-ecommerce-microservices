import pika
import json
import os

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')

def publish_order_created(order_data):
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.queue_declare(queue='order_created_queue', durable=True)
        
        message = json.dumps(order_data)
        channel.basic_publish(
            exchange='',
            routing_key='order_created_queue',
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2,
            )
        )
        connection.close()
    except Exception as e:
        print(f"Failed to publish order_created: {e}")

def publish_order_compensated(order_id, reason):
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.queue_declare(queue='order_compensated_queue', durable=True)
        
        message = json.dumps({"order_id": order_id, "reason": reason})
        channel.basic_publish(
            exchange='',
            routing_key='order_compensated_queue',
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2,
            )
        )
        connection.close()
    except Exception as e:
        print(f"Failed to publish order_compensated: {e}")

import pika
import json
import os

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')

def publish_customer_created(customer_data):
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.queue_declare(queue='customer_created_queue', durable=True)
        
        message = json.dumps(customer_data)
        channel.basic_publish(
            exchange='',
            routing_key='customer_created_queue',
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            )
        )
        connection.close()
    except Exception as e:
        print(f"Failed to publish message: {e}")

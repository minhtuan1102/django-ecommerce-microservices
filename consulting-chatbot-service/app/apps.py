from django.apps import AppConfig


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
    verbose_name = 'Consulting Chatbot Service'
    
    def ready(self):
        """Initialize app when Django starts"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Consulting Chatbot Service app initialized")

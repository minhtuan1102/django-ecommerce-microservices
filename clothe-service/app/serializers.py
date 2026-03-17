from rest_framework import serializers
from .models import Clothe

class ClotheSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clothe
        fields = '__all__'

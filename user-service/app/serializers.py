from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from .models import User, Job, Address

class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = '__all__'

class AddressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Address
        fields = '__all__'

class UserSerializer(serializers.ModelSerializer):
    job_info = JobSerializer(source='job', read_only=True)
    addresses = AddressSerializer(many=True, read_only=True)
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password', 'role', 'phone', 'staff_role', 'job', 'job_info', 'addresses', 'is_active', 'date_joined']
        extra_kwargs = {
            'password': {'write_only': True, 'required': False}
        }

    def create(self, validated_data):
        if 'password' in validated_data and validated_data['password']:
            validated_data['password'] = make_password(validated_data['password'])
        if 'username' not in validated_data:
            validated_data['username'] = validated_data['email']
        return super().create(validated_data)

class RegisterSerializer(serializers.Serializer):
    email = serializers.EmailField()
    username = serializers.CharField(required=False)
    password = serializers.CharField(min_length=6, write_only=True)
    role = serializers.ChoiceField(choices=['customer', 'staff', 'admin', 'service'], required=False)
    phone = serializers.CharField(required=False, allow_blank=True)
    staff_role = serializers.CharField(required=False, allow_blank=True, allow_null=True)

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

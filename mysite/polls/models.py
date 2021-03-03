from django.db import models

# Create your models here.
class DataDjango(models.Model):
    created_at = models.DateField()
    text = models.TextField()
    text_emotion = models.CharField(max_length=30)
    locations = models.CharField(max_length=100)
    language = models.CharField(max_length=50)

class DataDjango2(models.Model):
    created_at = models.DateField()
    text = models.TextField()
    text_emotion = models.CharField(max_length=30)
    locations = models.CharField(max_length=100)
    language = models.CharField(max_length=50)

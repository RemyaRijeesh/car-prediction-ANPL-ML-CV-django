from django.db import models

# Create your models here.


class ReviewAnalysis(models.Model):
    review = models.TextField()
    sentiment = models.IntegerField()
    
class CarDetails(models.Model):
    car_number = models.CharField(max_length=100)
    owner = models.CharField(max_length=100)
    address = models.TextField()
    
    def __str__(self):
        return self.owner
    

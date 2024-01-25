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
    
class CarCompany(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name

class CarModel(models.Model):
    id = models.AutoField(primary_key=True)
    car_company = models.ForeignKey(CarCompany, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.car_company.name} - {self.name}"
    

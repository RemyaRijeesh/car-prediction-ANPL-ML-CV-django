from django.db import models

# Create your models here.


class ReviewAnalysis(models.Model):
    review = models.TextField()
    sentiment = models.IntegerField()

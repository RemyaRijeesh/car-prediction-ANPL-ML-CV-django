
from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name="home"),
    path('prediction', predict_price, name="prediction"),
    path('number_detection', number_detection, name="number_detection"),
    path('review', sentimental_analysis, name="review"),
]
    

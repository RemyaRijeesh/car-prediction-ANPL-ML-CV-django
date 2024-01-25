
from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name="home"),
    path('prediction', predict_price, name="prediction"),
    path('number_detection', number_detection, name="number_detection"),
    path('review', sentimental_analysis, name="review"),
    path('get_model', get_model, name="get_model"),
]
    

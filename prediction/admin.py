from django.contrib import admin
from .models import ReviewAnalysis,CarDetails



@admin.register(ReviewAnalysis)
class ReviewAnalysisAdmin(admin.ModelAdmin):
    list_display = ('review', 'sentiment') 
    search_fields = ['sentiment']
    
admin.site.register(CarDetails)   
# myapp/admin.py

from django.contrib import admin
from .models import CarCompany, CarModel

admin.site.register(CarCompany)
admin.site.register(CarModel)
 
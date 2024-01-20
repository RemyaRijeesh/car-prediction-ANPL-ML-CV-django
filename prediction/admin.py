from django.contrib import admin
from .models import ReviewAnalysis



@admin.register(ReviewAnalysis)
class ReviewAnalysisAdmin(admin.ModelAdmin):
    list_display = ('review', 'sentiment') 
    search_fields = ['sentiment']
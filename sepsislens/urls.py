"""
URL patterns for sepsislens app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('upload/', views.upload_view, name='upload'),
    path('results/', views.results_view, name='results'),
    path('explain/', views.explain_view, name='explain'),
    path('model-comparison/', views.model_comparison_view, name='model_comparison'),
    path('about/', views.about_view, name='about'),
    path('export/csv/', views.export_results_csv, name='export_csv'),
]

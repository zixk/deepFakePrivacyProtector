from django.urls import path

from . import views

urlpatterns = [
    #ex: /obfus/
    path('', views.index, name='index'),

    #ex: /obfus/1/
    path('<int:participant_num>/', views.obfuscations, name='obfuscations'),
]
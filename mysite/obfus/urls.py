from django.urls import path

from . import views
from .views import CreatePostView

app_name="obfus"

urlpatterns = [
    #ex: /obfus/
    path('', CreatePostView.as_view(), name='add_post'),
    path('<int:participant_num>/', views.postDetail, name='post_detail'),
    path('<int:participant_num>/deepfake/', views.viewPicDeepfake, name='deepfake'),
    path('<int:participant_num>/pixelated/', views.viewPicPixelated, name='pixelated'),
    path('<int:participant_num>/blurred/', views.viewPicBlurred, name='blurred'),
    path('<int:participant_num>/masked/', views.viewPicMasked, name='masked'),
    path('<int:participant_num>/asis/', views.viewPicAsIs, name='asis')
]
"""pyweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.TitleGenPageFunc),
    path('index', views.TitleGenPageFunc),
    path('about', views.AboutFunc),
    path('content', views.ContentFunc),
    path('board', views.BoardFunc),
    path('board_write', views.BoardWriteFunc),
    path('board_save', views.BoardSaveFunc),
    path('board_view', views.BoardViewFunc),
    path('board_passwd_check', views.BoardPasswdCheckFunc),
    path('board_update', views.BoardUpdateFunc),
    path('board_update_save', views.BoardUpdateSaveFunc),
    path('board_delete', views.BoardDeleteFunc),
    path('comment_save', views.CommentSaveFunc),
    path('comment_passwd_check', views.CommentPasswdCheckFunc),
    path('comment_delete', views.CommentDeleteFunc),

    path('jaehong' , views.jaehong),
     
    path('youtube_data', views.DataViewFunc),  

    path('wordcloud', views.ShowWordCloudFunc),
    path('gentitle', views.TitleGenFunc),
    
    path('upload_date&views', views.AnalysisFunc),
    
    
    
    
    
    
    
    
]

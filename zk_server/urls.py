"""
URL configuration for zk_server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from zk_server_qeeg import views, views11, viewsx, views12, views11s

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('aeeg_monitor', views.aeeg_monitor),
    # path('aeeg_history', views.aeeg_history),
    # path('rbp_history', views.rbp_history),
    # path('abp_history', views.abp_history),
    # path('qeeg_history', views.qeeg_history),
    # path('qteeg_history', views.qteeg_history),
    # path('qteeg_history1', viewsx.qteeg_history),
    path('qteeg_history2', views11.qteeg_history),
    path('qteeg_history3', views12.qteeg_history),
    path('qteeg_history4', views11s.qteeg_history),
]

"""bitpyadm URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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

from django.views.generic import TemplateView
from django.contrib import admin
from django.urls import include
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views


urlpatterns = [
    path('', include('landingbitpy.urls')),
    path('admin/', admin.site.urls),
    path('admin/register/<str:plano>/<str:descricao>', views.register, name="sign-up"),
    path('admin/home', views.index, name="homeindex"),
    #path('admin/user_save', views.user_save)
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


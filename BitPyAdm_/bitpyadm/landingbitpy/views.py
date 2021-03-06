from django.shortcuts import render
from django import forms
from django.shortcuts import get_object_or_404
from .models import InvestorProfile
from .forms import ProfileEnquireForm
# Create your views here.
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.db.models import Q


def index(request):
    context = {
        'auth_user': ''
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'landing_page/index.html', context=context)


def register(request):
    context = {}
    return render(request, 'admin/register.html', context=context)


def login(request):
    context = {}
    return render(request, 'landing_page/login.html', context=context)


def investorprofile(request, id):
    item = InvestorProfile.objects.filter(Q(auth_user_id_id=id))
    form = ProfileEnquireForm(forms.Form(item))
    return render(request, 'admin/investorprofile.html', {'form': form})

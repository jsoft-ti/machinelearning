from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index(request):
    context = {
        'auth_user': ''
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'landing_page/index.html', context=context)

def register(request):
    context = {}
    return render(request, 'admin/register.html', context=context)
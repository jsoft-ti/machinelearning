from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required


# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
from .forms import RegistrationForm

@login_required
def index(request):
    return render(request,'admin/index.html')

def user_save(request):

    form = RegistrationForm(request.POST or None)
    user = request.user
    if form.is_valid():
        user = form.save(commit=False)
        password = form.cleaned_data.get("password")
        user.set_password(password)
        user.save()
        login(request, user)
    return render(request, '')


def register(request, plano, descricao):
    form = RegistrationForm(request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            user = form.save(commit=False)
            password = form.cleaned_data.get("password")
            user.set_password(password)
            user.save()
            login(request, user)
            return render(request, 'admin/index.html')
    form = RegistrationForm()
    args = {'form':form, 'plano':plano, 'descricao':descricao}
    return render(request,'admin/register.html',args)
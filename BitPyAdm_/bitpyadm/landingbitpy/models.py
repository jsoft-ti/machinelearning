from django.db import models
from django.contrib import auth
# Create your models here.

class Corretora(models.Model):
    name = models.CharField(max_length=100)
    status = models.IntegerField(default=0)
    description = models.CharField(max_length=100)

class StockRadar(models.Model):
    stock = models.CharField(max_length=5)
    status = models.IntegerField
    description = models.CharField(max_length=100)

class FavoriteStock(models.Model):
    auth_user_id = models.ForeignKey(auth.get_user_model(), on_delete=models.CASCADE)
    stock_radar_id = models.ForeignKey(StockRadar, on_delete=models.CASCADE)


class InvestorProfile(models.Model):
    auth_user_id = models.ForeignKey(auth.get_user_model(), on_delete=models.DO_NOTHING)
    r1 = models.IntegerField(default=0)
    r2 = models.IntegerField(default=0)
    r3 = models.IntegerField(default=0)
    r4 = models.IntegerField(default=0)
    r5 = models.IntegerField(default=0)
    r6 = models.IntegerField(default=0)
    r7 = models.IntegerField(default=0)
    r8 = models.IntegerField(default=0)
    r9 = models.IntegerField(default=0)
    r10 = models.IntegerField(default=0)
    r11 = models.IntegerField(default=0)
    r12 = models.IntegerField(default=0)
    corretora = models.ForeignKey(Corretora, on_delete=models.CASCADE)
    profiletype = models.CharField(max_length=10)

class ModelCandidateVote(models.Model):
    auth_user_id = models.ForeignKey(auth.get_user_model(), on_delete=models.CASCADE)
    stock_radar_id = models.ForeignKey(StockRadar, on_delete=models.CASCADE)

class Models(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=100)
    confidence = models.CharField(max_length=100)
    points = models.IntegerField(default=0)
    trainingdate = models.DateTimeField(auto_now_add=True, blank=True)

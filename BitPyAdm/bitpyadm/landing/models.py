from django.db import models
from django.contrib import auth


# Create your models here.
from .choices import *


class Broker(models.Model):
    name = models.CharField(max_length=100)
    status = models.IntegerField(default=0)
    description = models.CharField(max_length=100)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "Brokers"


class StockRadar(models.Model):
    name = models.CharField(max_length=5)
    status = models.IntegerField(default=1)
    description = models.CharField(max_length=100)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "Stocks Radar"


class ProfileType(models.Model):
    name = models.CharField(max_length=50)
    status = models.IntegerField(default=1)
    description = models.CharField(max_length=600)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "Profile Types"


class Models(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=100)
    confidence = models.CharField(max_length=100)
    points = models.IntegerField(default=0)
    training_date = models.DateTimeField(auto_now_add=True, blank=True)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "Models"


class FavoriteStock(models.Model):
    auth_user_id = models.ForeignKey(auth.get_user_model(), on_delete=models.CASCADE)
    stock_radar_id = models.ForeignKey(StockRadar, on_delete=models.CASCADE)

    class Meta:
        ordering = ["auth_user_id"]
        verbose_name_plural = "Favorite Stocks"


class InvestorProfile(models.Model):
    auth_user_id = models.ForeignKey(auth.get_user_model(), on_delete=models.DO_NOTHING, verbose_name='User')
    r1 = models.IntegerField(default=0, choices=R1_CHOICES, verbose_name=questions[0])
    r2 = models.IntegerField(default=0, choices=R2_CHOICES, verbose_name=questions[1])
    r3 = models.IntegerField(default=0, choices=R3_CHOICES, verbose_name=questions[2])
    r4 = models.IntegerField(default=0, choices=R4_CHOICES, verbose_name=questions[3])
    r5 = models.IntegerField(default=0, choices=R5_CHOICES, verbose_name=questions[4])
    r6 = models.IntegerField(default=0, choices=R6_CHOICES, verbose_name=questions[5])
    r7 = models.IntegerField(default=0, choices=R7_CHOICES, verbose_name=questions[6])
    r8 = models.IntegerField(default=0, choices=R8_CHOICES, verbose_name=questions[7])
    r9 = models.IntegerField(default=0, choices=R9_CHOICES, verbose_name=questions[8])
    r10 = models.IntegerField(default=0, choices=R10_CHOICES, verbose_name=questions[9])
    r11 = models.IntegerField(default=0, choices=R11_CHOICES, verbose_name=questions[10])
    r12 = models.IntegerField(default=0, choices=R12_CHOICES, verbose_name=questions[11])
    broker_id = models.ForeignKey(Broker, on_delete=models.CASCADE)
    profiletype = models.ForeignKey(ProfileType, on_delete=models.CASCADE)

    def calcProfile(self):
        w = [3, 2, 2, 1, 2, 3, 2, 3, 2, 2, 2, 2]
        choice = [self.r1, self.r2, self.r3, self.r4, self.r5, self.r6, self.r7, self.r8, self.r9, self.r10, self.r11, self.r12]
        q = sum(w * choice)

        if (q > 36 &  q <52):
            return 1 #– Perfil de Risco Conservador
        elif (q > 53 &  q <65):
            return 2 #– Perfil de Risco Moderado
        elif (q > 66 &  q <71):
            return 3 #– Perfil de Risco Dinâmico
        elif (q > 72 & q < 83):
            return 4 #– Perfil de Risco Arrojado
        elif (q > 84 & q < 104):
           return 5 #– Perfil de Risco Agressivo
        else:
           return 0 #Análise de perfil não realizada

    class Meta:
        verbose_name_plural = "Investor Profiles"


class ModelCandidateVote(models.Model):
    auth_user_id = models.ForeignKey(auth.get_user_model(), on_delete=models.CASCADE)
    stock_radar_id = models.ForeignKey(StockRadar, on_delete=models.CASCADE)

    class Meta:
        ordering = ["auth_user_id"]
        verbose_name_plural = "Model Candidate Votes"
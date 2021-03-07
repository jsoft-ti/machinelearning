from django.contrib import admin
from landing.models import Broker, StockRadar, ProfileType, Models, \
    FavoriteStock, InvestorProfile, ModelCandidateVote


# Register your models here.
# admin.site.register(Corretora)
class BrokerAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')


# admin.site.register(StockRadar)
class StockRadarAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')


# admin.site.register(ProfileType)
class ProfileTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')


# admin.site.register(Models)
class ModelsAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'confidence', 'points', 'training_date')


# admin.site.register(FavoriteStock)
class FavoriteStockAdmin(admin.ModelAdmin):
    list_display = ('auth_user_id', 'stock_radar_id')


# admin.site.register(InvestorProfile)
class InvestorProfileAdmin(admin.ModelAdmin):
    list_display = ('auth_user_id','broker_id', 'profiletype')


# admin.site.register(ModelCandidateVote)
class ModelCandidateVoteAdmin(admin.ModelAdmin):
    list_display = ('auth_user_id', 'stock_radar_id')


admin.site.register(Broker, BrokerAdmin)
admin.site.register(StockRadar, StockRadarAdmin)
admin.site.register(ProfileType, ProfileTypeAdmin)
admin.site.register(Models, ModelsAdmin)
admin.site.register(FavoriteStock, FavoriteStockAdmin)
admin.site.register(InvestorProfile, InvestorProfileAdmin)
admin.site.register(ModelCandidateVote, ModelCandidateVoteAdmin)

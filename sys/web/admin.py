from django.contrib import admin
from web.models import UserProfile, history, region
# Register your models here.

#secondname是证书编号
# 创建一个ModelAdmin的子类
class UserAdmin(admin.ModelAdmin):
    search_fields = ['username', 'user.first_name']

class HistoryAdmin(admin.ModelAdmin):
    date_hierarchy = 'query_time'

class RegionAdmin(admin.ModelAdmin):
    search_fields = ['city']
    exclude = ('latitude','longitude')




# 注册的时候，将原模型和ModelAdmin耦合起来
admin.site.register(UserProfile, UserAdmin)
admin.site.register(history, HistoryAdmin)
admin.site.register(region, RegionAdmin)

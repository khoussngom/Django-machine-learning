"""
URL configuration for memoire project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.urls import path,include
from prediction import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('acceuil/',views.accueil,name='accueil'),
    path('saisi/',views.Semestreview,name='saisi'),
    path('resultats/',views.resultat,name='resultats'),
    path('',views.userform,name='inscription'),
    path('connexion/',views.connexionform,name='connexion'),
    path('deconnexion/',views.deconnexion,name='deconnexion'),
    path('user/',views.liste_utilisateurs, name='liste_user'),
    path('supprimer-user/<int:user_id>/',views.supprimer_utilisateur, name='supprimer_user')

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root = settings.MEDIA_ROOT)

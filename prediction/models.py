from django.db import models
from django.contrib.auth.models import AbstractBaseUser,BaseUserManager,PermissionsMixin
from .managers import UtilisateurManager
# Create your models here.
class Semestre(models.Model):
    nom = models.CharField(max_length= 100)
    resultat_prevu = models.FileField(upload_to='learning_files/')
    resultat_actuel = models.FileField(upload_to = 'learning_files/')

    def __str__(self) :
        
     return self.nom
    


class Analyse(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    Semestre = models.ForeignKey(Semestre,on_delete=models.CASCADE)
    Analyse_file = models.FileField(upload_to = 'analyse_files/')


    def __str__(self) -> str:
        return self.title
    
class Utlisateurs(AbstractBaseUser,PermissionsMixin):
   objects = UtilisateurManager()
   prenom = models.CharField(max_length=25)
   nom = models.CharField(max_length=25)
   username = models.CharField(max_length=50,blank=True,null=True,unique=True)
   mdp = models.CharField(max_length=25,blank=True,null=True,unique=True)
   is_staff = models.BooleanField(default=False)  # Ce champ permet l'accès à l'admin
   is_superuser = models.BooleanField(default=False)  # Super utilisateur
   is_active = models.BooleanField(default=True)  # Utilisateur actif ou non
   date_joined = models.DateTimeField(auto_now_add=True)

   objects= UtilisateurManager()

   USERNAME_FIELD= 'username'
   REQUIRED_FIELDS= ['mdp']

   def __str__(self) -> str:
      return self.username
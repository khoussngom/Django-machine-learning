from typing import Any, Mapping
from django import forms
from django.core.files.base import File
from django.db.models.base import Model
from django.forms.utils import ErrorList
from  prediction.models import Utlisateurs


class UserForm(forms.ModelForm):
    class Meta:
        model = Utlisateurs
        fields = ['prenom','nom','username','mdp']
         
        widgets ={
           'prenom':forms.TextInput(attrs={'class':'form-control','placeholder':'entrer votre prénom'}),
           'nom':forms.TextInput(attrs={'class':'form-control','placeholder':'entrer votre nom'}),
           'username':forms.TextInput(attrs={'placeholder':'entrer votre nom d\'utilisateur'}),
           'mdp':forms.PasswordInput(attrs={'class':'form-control','placeholder':'entrer votre mot de passe'}),
        } 
    

    def __init__(self, *args, **kwargs):
        super(UserForm,self).__init__(*args,**kwargs)
        self.fields['prenom'].max_length = 30
        self.fields['nom'].max_length = 20
        self.fields['username'].max_length = 20
        self.fields['mdp'].max_length = 20

class ConnexionForm(forms.ModelForm):
    class Meta:
        model = Utlisateurs 
        fields=['username','mdp']   
        widgets ={
                   'mdp':forms.PasswordInput(attrs={'class':'form-control','placeholder':'entrer votre mot de passe'}),
                  }
        
        
from django.shortcuts import render,redirect, get_object_or_404
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required,user_passes_test
from django.contrib import messages
from .semestre.forms  import SemestreForm,AnalyseForm
from .models import Semestre
from .models import Utlisateurs
from .utilisateurs.forms import UserForm,ConnexionForm
from .ml_model import PythiaSystem
import pandas as pd
import base64
from django.contrib.auth.models import User

def accueil(request):
    return render(request,'accueil.html')

@login_required
def resultat(request):
    resultats = Semestre.objects.all()
    return render(request, 'success.html',{'resultats':resultats})
    

def userform(request):
    if request.method == 'POST':
        form =  UserForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['mdp'])
            user.save()
            return redirect('connexion')
    else:
        form = UserForm()

    return render(request, 'inscription.html',{'form':form})    

def connexionform(request):
    if request.method == 'POST':
            
            username = request.POST['username']
            password= request.POST['password']

            user = authenticate(request, username=username,password=password)
            if user is not None:
                login(request, user) 
                print("success")
                return redirect('accueil')
            else:
                messages.error(request,"identification non valide")

     
    return render(request,'connexion.html') 



@login_required
def Semestreview(request):
    if request.method == 'POST':
        form = SemestreForm(request.POST, request.FILES)
      
        
        if form.is_valid() :
            fichier_csv = form.save()
           
            csv_file = request.FILES['resultat_prevu']
            
            
   
            
            
            if not csv_file.name.endswith('.csv'):
                messages.error(request, 'Les fichiers ne sont pas au format CSV.')
                return redirect('saisi')
            
            try:
                csv_file.seek(0)
                data = pd.read_csv(csv_file,encoding='utf-8',header=0)
                # Liste pour stocker les informations des étudiants
                etudiants=data.to_dict(orient='records')
                

                if data.empty or data.shape[0] == 0 :
                    messages.error(request, 'Les fichiers CSV sont vides ou ne contiennent pas de données valides.')
                    return redirect('saisi')

            except pd.errors.ParserError:
                messages.error(request, 'Erreur lors de la lecture des fichiers CSV.')
                return redirect('saisi')
            except pd.errors.EmptyDataError:
                messages.error(request, 'le fichier CSV ne contient aucune donnée.')
                return redirect('saisi')

            # Initialiser Pythia
            pythia = PythiaSystem()

            # Prétraiter les données
            X, y = pythia.preprocess_data(data)

            # Évaluer les modèles de régression et de classification
            reg_results = pythia.evaluate_regression(X, y)
            clf_results = pythia.evaluate_classification(X, y)

            # Entraîner les meilleurs modèles
            best_models = pythia.train_best_models(X, y)

            # Générer des prédictions pour un échantillon (5 premiers étudiants)
            predictions = pythia.predict(best_models, X)
         
  
            graph=pythia.plot_roc_auc(y,predictions,best_models)

            # Vérifier que les longueurs sont les mêmes
            if len(etudiants) != len(predictions):
                messages.error(request, 'Erreur : le nombre de prédictions ne correspond pas au nombre d\'étudiants.')
                return redirect('saisi')
            # Ajouter les prédictions aux informations des étudiants
            for i, etudiant in enumerate(etudiants):
                etudiant['analyse'] = 'valide' if predictions[i]['classification'] else 'invalide'
            
            #graphic=pythia.plot_roc_auc( X,pred, best_models)
            
            # Afficher les résultats dans le template
            return render(request, 'analyse/resultats.html', {
                'etudiants': etudiants,
                'reg_results': reg_results,
                'clf_results': clf_results,
                'graph': graph
            })
        else:
            messages.error(request, 'Les formulaires ne sont pas valides.')
            return redirect('saisi')

    else:
        form = SemestreForm()

    return render(request, 'saisi.html', {'form': form})


def deconnexion(request):
    logout(request)
    return redirect('connexion')
        



# Vérifie si l'utilisateur connecté est un administrateur
def is_admin(user):
    return user.is_superuser

@user_passes_test(is_admin)  # Limite l'accès aux administrateurs uniquement
def supprimer_utilisateur(request, user_id):
    utilisateur = get_object_or_404(Utlisateurs, id=user_id)
    
    if request.method == 'POST':
        utilisateur.delete()
        messages.success(request, f"L'utilisateur {utilisateur.username} a été supprimé avec succès.")
        return redirect('liste_user')  # Rediriger vers la liste des utilisateurs après suppression

    return render(request, 'supprimer_user.html', {'utilisateur': utilisateur})        

def liste_utilisateurs(request):
    utilisateurs = Utlisateurs.objects.all()
    return render(request, 'liste_user.html', {'utilisateurs': utilisateurs})
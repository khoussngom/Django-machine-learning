from django import forms
from prediction.models import Semestre,Analyse

class SemestreForm (forms.ModelForm):
    class Meta:
        model = Semestre
        fields = ['resultat_prevu']
       # labels = {
          # 'resultat_prevu':'résultats antérieur',
       #}
#,'resultat_actuel'
class AnalyseForm(forms.ModelForm):
    class Meta:
       model= Analyse
       fields=['Analyse_file']
    


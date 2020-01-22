from django import forms
from .models import Participant

class PostForm(forms.ModelForm):

    class Meta:
        model = Participant
        fields = ['participant_num', 'participant_pic']
from django import forms
from .models import Participant



class PostForm(forms.ModelForm):
    class Meta:
        model = Participant
        fields = ['participant_num', 'participant_pic', 'not_participant_pic', 'gender', 'skin_tone']
        labels = {
            'participant_num': 'Participant Number',
            'participant_pic': "Participant's Picture",
            'not_participant_pic': "Bystander's Picture",
            'gender': 'Gender',
            'skin_tone': 'Skin Tone'
        }

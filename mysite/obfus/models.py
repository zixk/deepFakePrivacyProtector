from django.db import models

# Create your models here.
class Participant(models.Model):
    participant_num = models.IntegerField('participant number')

    def __str__(self):
        return self.participant_num
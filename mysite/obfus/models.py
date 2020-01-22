from django.db import models

# Create your models here.
class Participant(models.Model):
    participant_num = models.IntegerField('participant number', unique=True)
    participant_pic = models.ImageField(upload_to='images/')
    participant_pic_left = models.ImageField(upload_to='images/cropped/', blank=True)
    participant_pic_right = models.ImageField(upload_to='images/cropped/', blank=True)
    deepfake_obfus = models.ImageField(upload_to='images/deepfake', blank=True)
    blurred_obfus = models.ImageField(upload_to='images/blurred', blank=True)
    pixelated_obfus = models.ImageField(upload_to='images/blurred', blank=True)
    masked_obfus = models.ImageField(upload_to='images/blurred', blank=True)

    def __str__(self):
        return str(self.participant_num)
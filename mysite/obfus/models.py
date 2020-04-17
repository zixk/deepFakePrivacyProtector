from django.db import models

# Create your models here.
class Participant(models.Model):
    GENDER_CHOICES = [('Male', 'Male'),
                      ('Female', 'Female')]
    SKIN_TONE = [('Light', 'Light'),
                 ('Dark', 'Dark')]

    participant_num = models.IntegerField('participant number', unique=True)
    participant_pic = models.ImageField(upload_to='images/')
    not_participant_pic = models.ImageField(upload_to='images/')
    gender = models.CharField(max_length=64, choices=GENDER_CHOICES, default="")
    skin_tone = models.CharField(max_length=64, choices=SKIN_TONE, default="")
    participant_pic_left = models.ImageField(upload_to='images/cropped/', blank=True)
    not_participant_pic_left = models.ImageField(upload_to='images/cropped/', blank=True)
    participant_pic_right = models.ImageField(upload_to='images/cropped/', blank=True)
    not_participant_pic_right = models.ImageField(upload_to='images/cropped/', blank=True)
    deepfake_obfus = models.ImageField(upload_to='images/deepfake', blank=True)
    blurred_obfus = models.ImageField(upload_to='images/blurred', blank=True)
    pixelated_obfus = models.ImageField(upload_to='images/pixelated', blank=True)
    masked_obfus = models.ImageField(upload_to='images/masked', blank=True)
    avatar_obfus = models.ImageField(upload_to='images/avatar', blank=True)
    bystander_deepfake_obfus = models.ImageField(upload_to='images/deepfake', blank=True)
    bystander_blurred_obfus = models.ImageField(upload_to='images/blurred', blank=True)
    bystander_pixelated_obfus = models.ImageField(upload_to='images/pixelated', blank=True)
    bystander_masked_obfus = models.ImageField(upload_to='images/masked', blank=True)
    bystander_avatar_obfus = models.ImageField(upload_to='images/avatar', blank=True)

    def __str__(self):
        return str(self.participant_num)
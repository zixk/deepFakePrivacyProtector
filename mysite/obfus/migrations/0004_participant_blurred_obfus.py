# Generated by Django 3.0.1 on 2020-01-14 15:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('obfus', '0003_participant_deepfake_obfus'),
    ]

    operations = [
        migrations.AddField(
            model_name='participant',
            name='blurred_obfus',
            field=models.ImageField(blank=True, upload_to='images/blurred'),
        ),
    ]

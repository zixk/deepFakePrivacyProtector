# Generated by Django 3.0.1 on 2020-01-30 12:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('obfus', '0009_auto_20200130_1154'),
    ]

    operations = [
        migrations.AddField(
            model_name='participant',
            name='not_participant_pic',
            field=models.ImageField(default=None, upload_to='images/'),
            preserve_default=False,
        ),
    ]

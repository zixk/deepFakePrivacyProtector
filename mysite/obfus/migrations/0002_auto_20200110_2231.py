# Generated by Django 3.0.1 on 2020-01-10 22:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('obfus', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='participant',
            name='participant_pic_left',
            field=models.ImageField(blank=True, upload_to='images/cropped/'),
        ),
        migrations.AddField(
            model_name='participant',
            name='participant_pic_right',
            field=models.ImageField(blank=True, upload_to='images/cropped/'),
        ),
        migrations.AlterField(
            model_name='participant',
            name='participant_num',
            field=models.IntegerField(unique=True, verbose_name='participant number'),
        ),
    ]

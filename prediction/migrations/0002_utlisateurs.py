# Generated by Django 5.1.1 on 2024-09-29 12:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Utlisateurs',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('prenom', models.CharField(max_length=25)),
                ('nom', models.CharField(max_length=25)),
                ('username', models.CharField(max_length=25)),
                ('mdp', models.CharField(max_length=25)),
            ],
        ),
    ]

# Generated by Django 5.1.1 on 2024-09-26 20:59

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Semestre',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nom', models.CharField(max_length=100)),
                ('resultat_prevu', models.FileField(upload_to='learning_files/')),
                ('resultat_actuel', models.FileField(upload_to='learning_files/')),
            ],
        ),
        migrations.CreateModel(
            name='Analyse',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('description', models.TextField()),
                ('Analyse_file', models.FileField(upload_to='analyse_files/')),
                ('Semestre', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='prediction.semestre')),
            ],
        ),
    ]
from django.contrib.auth.models import BaseUserManager

class UtilisateurManager(BaseUserManager):
    def create_user(self, username, password=None, **extra_fields):
        """
        Crée et sauvegarde un utilisateur avec un nom d'utilisateur et un mot de passe.
        """
        if not username:
            raise ValueError('Le nom d\'utilisateur doit être défini')
        user = self.model(username=username, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, password=None, **extra_fields):
        """
        Crée et sauvegarde un superutilisateur avec un nom d'utilisateur et un mot de passe.
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser doit avoir is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser doit avoir is_superuser=True.')

        return self.create_user(username, password, **extra_fields)

    def get_by_natural_key(self, username):
        return self.get(username=username)
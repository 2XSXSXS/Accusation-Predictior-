from django import forms
from django.contrib.auth.models import User


class RegisterForm(forms.Form):
    username = forms.CharField(max_length=50)
    password = forms.CharField(max_length=500)
    first_name = forms.CharField(max_length=50)
    certificate = forms.CharField(max_length=150)
    work_type = forms.CharField(max_length=50)
    org = forms.CharField(max_length=50)
    pic_file = forms.ImageField()


class LoginForm(forms.Form):
    username = forms.CharField(max_length=50)
    password = forms.CharField(max_length=500)
    remember = forms.BooleanField(required=False)


class QueryForm(forms.Form):
    status = forms.CharField(max_length=4)
    query = forms.CharField(max_length=20000, required=False)
    advice = forms.CharField(max_length=20000, required=False)
    query_id = forms.IntegerField(required=False)


class GuestForm(forms.Form):
    query = forms.CharField(max_length=20000, required=False)


class BatchForm(forms.Form):
    status = forms.CharField(max_length=4)
    text_file = forms.FileField()


class ForgetForm(forms.Form):
    email = forms.CharField(max_length=100)


class ResetForm(forms.Form):
    password = forms.CharField(max_length=200)
    verify_code = forms.CharField(max_length=200)

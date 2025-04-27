# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

# Create your views here.
from django.shortcuts import render
from django.contrib.auth.views import LogoutView
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import LoginForm, SignUpForm


def login_view(request):
    form = LoginForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("/")
            else:
                msg = 'Invalid credentials'
        else:
            msg = 'Error validating the form'

    return render(request, "accounts/login.html", {"form": form, "msg": msg})


def register_user(request):
    msg = None
    success = False

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=username, password=raw_password)

            msg = 'User created - please <a href="/login">login</a>.'
            success = True

            # return redirect("/login/")

        else:
            msg = 'Form is not valid'
    else:
        form = SignUpForm()

    return render(request, "accounts/register.html", {"form": form, "msg": msg, "success": success})


class CustomLogoutView(LogoutView):
    def get(self, request, *args, **kwargs):
        return self.post(request, *args, **kwargs)


# home/views.py


def financial_exclusion_predictor(request):
    if request.method == 'POST':
        form = FinancialExclusionForm(request.POST)
        if form.is_valid():
            # Get prediction
            prediction, probability = predict_financial_exclusion(
                form.cleaned_data)

            # Prepare result message
            if prediction == 1:
                result = {
                    'type': 'error',
                    'message': f"Prediction: Financially Excluded",
                    'probability': f"Risk Score = {probability: .2%}"
                }
            else:
                result = {
                    'type': 'success',
                    'message': f"Prediction: Financially Included",
                    'probability': f"Confidence = {1 - probability: .2%}"
                }

            return render(request, 'home/predictor.html', {
                'form': form,
                'result': result
            })
    else:
        form = FinancialExclusionForm()

    return render(request, 'home/predictor.html', {'form': form})

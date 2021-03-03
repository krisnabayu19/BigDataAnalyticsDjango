from django import forms

class NameForm(forms.Form):
    teks_input2 = forms.CharField(label='teks_input', max_length=100)

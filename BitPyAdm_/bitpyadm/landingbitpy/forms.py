from django import forms
from django.utils.translation import gettext_lazy as _

from .models import InvestorProfile


class ProfileEnquireForm(forms.Form):
    class Meta:
        model = InvestorProfile
        fields = (
            'r1',
            'r2',
            'r3',
            'r4',
            'r5',
            'r6',
            'r7',
            'r8',
            'r9',
            'r10',
            'r11',
            'r12',
            'corretora',
            'profiletype'
        )
        labels = {
            'r1': _("Qual a principal finalidade de investir seu patrimônio conosco?"),
            'r2': _("Por quanto tempo pretende deixar seus recursos investidos conosco?"),
            'r3': _("Em relação aos seus investimentos na SLW, qual é a necessidade futura dos recursos aplicados?"),
            'r4': _(" Qual a sua renda mensal?"),
            'r5': _("Qual percentual da sua renda o (a) Sr.(a) investe regularmente?"),
            'r6': _(
                "Por conta de oscilações do mercado, considere que seus investimentos percam 10% do valor aplicado. Neste caso, o que o (a) Sr.(a) faria?"),
            'r7': _(
                "Quais dos produtos listados abaixo você tem familiaridade? (Esta questão permite múltiplas respostas. Para fins de cálculo de Perfil, deve ser utilizada a resposta de maior valor de pontuação entre as respostas assinaladas)."),
            'r8': _("Quais investimentos você realizou frequentemente nos últimos 24 meses?"),
            'r9': _(
                "Qual é a atual composição dos seus investimentos por categoria? (Esta questão permite múltiplas respostas Para fin de cálculo do Perfil, deve ser utilizada a resposta com o maior percentual e no caso de respostas com porcentagens igus, deve ser utilizada a resposta mais conservadora)."),
            'r10': _("Qual é o valor do seu Patrimônio?"),
            'r11': _(
                "Como você classificaria a relação de sua formação acadêmica e da sua experiência profissional em relação aos seus conhecimentos sobre o mercado financeiro?"),
            'r12': _("Qual das respostas abaixo mais se assemelha à sua personalidade como investidor?"),
            'corretora': _("Escolha sua corretora"),
            'profiletype': _("Perfil")
        }

# Generated by Django 3.1.7 on 2021-03-07 12:23

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Corretora',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('status', models.IntegerField(default=0)),
                ('description', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Models',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('description', models.CharField(max_length=100)),
                ('confidence', models.CharField(max_length=100)),
                ('points', models.IntegerField(default=0)),
                ('training_date', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='ProfileType',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('status', models.IntegerField(default=1)),
                ('description', models.CharField(max_length=600)),
            ],
        ),
        migrations.CreateModel(
            name='StockRadar',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=5)),
                ('status', models.IntegerField(default=1)),
                ('description', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='ModelCandidateVote',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('auth_user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('stock_radar_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='landing.stockradar')),
            ],
        ),
        migrations.CreateModel(
            name='InvestorProfile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('r1', models.IntegerField(choices=[('0', ''), ('1', 'Preservar meu patrimônio assumindo um risco menor.'), ('2', 'Uma combinação entre preservação do meu patrimônio e sua valorização.'), ('3', 'Maximizar o potencial de ganho assumindo um risco maior.')], default=0)),
                ('r2', models.IntegerField(choices=[('0', ''), ('1', 'Até 1 ano.'), ('2', '1 a 5 anos.'), ('3', 'Mais de 5 anos.'), ('4', 'Essa reserva não será utilizada, a não ser em caso de emergência.')], default=0)),
                ('r3', models.IntegerField(choices=[('0', ''), ('1', 'Preciso desse dinheiro como complemento de renda .'), ('2', 'Eventualmente posso precisar utilizar uma parte dele.'), ('3', 'Não tenho necessidade imediata desse dinheiro')], default=0)),
                ('r4', models.IntegerField(choices=[('0', ''), ('1', 'Até R$ 3.000,00'), ('2', 'Entre R$ 3.000,00 e R$ 5.000,00.'), ('3', 'Entre R$ 5.000,00 e R$ 10.000,00.'), ('4', 'Entre R$ 10.000,00 e R$ 30.000,00.'), ('5', 'Mais de R$ 30.000,00.')], default=0)),
                ('r5', models.IntegerField(choices=[('0', ''), ('1', 'Até 10%.'), ('2', 'De 10 a 20%.'), ('3', 'De 20% a 50%.'), ('4', 'Acima de 50%.')], default=0)),
                ('r6', models.IntegerField(choices=[('0', ''), ('1', 'Não sei o que faria.'), ('2', 'Venderia toda a posição.'), ('3', 'Manteria a posição.'), ('4', 'Aumentaria a posição.')], default=0)),
                ('r7', models.IntegerField(choices=[('0', ''), ('1', 'Poupança, Fundos DI, CDB, Fundos de Renda Fixa.'), ('2', 'Fundos Multimercados, Títulos Públicos, LCI, LCA.'), ('3', ' Fundos de Ações, Ações, Fundos Imobiliários, Debêntures, Fundos Cambiais, Clubes de Investimento.'), ('4', 'Fundos de Investimentos em Participações (FIP), Derivativos (Futuros, Opções e Swaps).')], default=0)),
                ('r8', models.IntegerField(choices=[('0', ''), ('1', 'Preservar meu patrimônio assumindo um risco menor.'), ('2', 'Uma combinação entre preservação do meu patrimônio e sua valorização.'), ('3', 'Maximizar o potencial de ganho assumindo um risco maior.')], default=0)),
                ('r9', models.IntegerField(choices=[('0', ''), ('1', 'Renda Variável (Ações e Fundos de Ações).'), ('2', 'Fundos de Investimento Multimercado.'), ('3', 'Renda Fixa (Fundos de Renda Fixa, DI, CDBs, Poupança).'), ('4', 'Imóveis.'), ('5', 'Outros.')], default=0)),
                ('r10', models.IntegerField(choices=[('0', ''), ('1', 'Até R$ 20.000,00.'), ('2', 'Entre R$ 20.000,01 e R$ 100.000,00.'), ('3', 'Entre R$ 100.000,01 a R$ 1.000.000,00.'), ('4', 'Acima de R$ 1.000.000,01.')], default=0)),
                ('r11', models.IntegerField(choices=[('0', ''), ('1', 'Não tenho formação acadêmica na área financeira, mas desejo operar no mercado de capitais e financeiro.'), ('2', 'Apesar de não ter a formação acadêmica na área financeira possuo experiência no mercado de capitais e financeiro.'), ('3', 'Tenho formação na área financeira e conheço as regras do mercado financeiro.'), ('4', 'Tenho formação acadêmica e experiência profissional na área financeira, por isto conheço profundamente o mercado financeiro, como operações de derivativos e estruturadas.')], default=0)),
                ('r12', models.IntegerField(choices=[('1', 'Não admito perder nada do capital investido. Procuro um retorno seguro e sem oscilações. Segurança é mais importante do que rentabilidade.'), ('2', 'Não admito perder nada do capital investido, no entanto posso arriscar uma parte do capital para alcançar resultados melhores que a renda fixa tradicional. Valorizo mais a segurança do que a rentabilidade.'), ('3', 'Posso correr riscos para conseguir uma rentabilidade acima da média, no entanto, prezo a preservação de 100% do capital investido. Divido minhas preferências entre segurança e rentabilidade, mas ainda prefiro segurança à rentabilidade..'), ('4', 'Admito perdas de até 20% do capital investido, se a proposta de investimento gerar possibilidade de altos retornos. A procura por rentabilidade é mais importante do que a segurança.'), ('5', 'Minha prioridade é maximizar a rentabilidade, com a segurança em segundo plano. Posso correr grande riscos para obter elevados retornos, admitindo perder mais de 20% do meu capital investido.')], default=0)),
                ('auth_user_id', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to=settings.AUTH_USER_MODEL)),
                ('corretora_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='landing.corretora')),
                ('profiletype', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='landing.profiletype')),
            ],
        ),
        migrations.CreateModel(
            name='FavoriteStock',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('auth_user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('stock_radar_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='landing.stockradar')),
            ],
        ),
    ]

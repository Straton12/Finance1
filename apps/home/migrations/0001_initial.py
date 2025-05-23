# Generated by Django 3.2.6 on 2025-03-22 03:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Data2021',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('respondent_id', models.IntegerField(blank=True, null=True)),
                ('age', models.IntegerField(blank=True, null=True)),
                ('gender', models.CharField(blank=True, max_length=10, null=True)),
                ('education_level', models.CharField(blank=True, max_length=50, null=True)),
                ('residence_type', models.CharField(blank=True, max_length=20, null=True)),
                ('marital_status', models.CharField(blank=True, max_length=50, null=True)),
                ('relationship_to_hh', models.CharField(blank=True, max_length=50, null=True)),
                ('region', models.CharField(blank=True, max_length=50, null=True)),
                ('population_weight', models.FloatField(blank=True, null=True)),
                ('mobile_money_registered', models.CharField(blank=True, max_length=5, null=True)),
                ('savings_mobile_banking', models.CharField(blank=True, max_length=5, null=True)),
                ('bank_account_current', models.CharField(blank=True, max_length=5, null=True)),
                ('bank_account_savings', models.CharField(blank=True, max_length=5, null=True)),
                ('bank_account_everyday', models.CharField(blank=True, max_length=5, null=True)),
                ('postbank_account', models.CharField(blank=True, max_length=5, null=True)),
                ('bank_overdraft', models.CharField(blank=True, max_length=5, null=True)),
                ('debit_card', models.CharField(blank=True, max_length=5, null=True)),
                ('credit_card', models.CharField(blank=True, max_length=5, null=True)),
                ('savings_microfinance', models.CharField(blank=True, max_length=5, null=True)),
                ('savings_sacco', models.CharField(blank=True, max_length=5, null=True)),
                ('savings_group_friends', models.CharField(blank=True, max_length=5, null=True)),
                ('savings_family_friend', models.CharField(blank=True, max_length=5, null=True)),
                ('savings_secret_place', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_bank', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_mobile_banking', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_sacco', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_microfinance', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_shylock', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_group_chama', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_govt', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_employer', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_family_friend', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_shopkeeper_cash', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_goods_credit', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_digital_app', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_agri_buyer_supplier', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_hire_purchase', models.CharField(blank=True, max_length=5, null=True)),
                ('loan_mortgage', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_motor', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_home', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_crop', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_livestock', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_nhif', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_health_other', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_life', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_education', models.CharField(blank=True, max_length=5, null=True)),
                ('insurance_other', models.CharField(blank=True, max_length=5, null=True)),
                ('pension_nssf', models.CharField(blank=True, max_length=5, null=True)),
                ('pension_mbao', models.CharField(blank=True, max_length=5, null=True)),
                ('pension_other', models.CharField(blank=True, max_length=5, null=True)),
                ('financially_excluded', models.CharField(blank=True, max_length=5, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Survey2016',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('respondent_id', models.BigIntegerField(blank=True, null=True)),
                ('age', models.IntegerField()),
                ('gender', models.CharField(max_length=10)),
                ('education_level', models.CharField(max_length=50)),
                ('residence_type', models.CharField(max_length=20)),
                ('marital_status', models.CharField(max_length=50)),
                ('relationship_to_hh', models.CharField(max_length=50)),
                ('region', models.CharField(max_length=50)),
                ('population_weight', models.IntegerField()),
                ('mobile_money_registered', models.CharField(max_length=10)),
                ('savings_mobile_banking', models.CharField(max_length=10)),
                ('bank_account_current', models.CharField(max_length=10)),
                ('bank_account_savings', models.CharField(max_length=10)),
                ('bank_account_everyday', models.CharField(max_length=10)),
                ('postbank_account', models.CharField(max_length=10)),
                ('bank_overdraft', models.CharField(max_length=10)),
                ('debit_card', models.CharField(max_length=10)),
                ('credit_card', models.CharField(max_length=10)),
                ('savings_microfinance', models.CharField(max_length=10)),
                ('savings_sacco', models.CharField(max_length=10)),
                ('savings_group_friends', models.CharField(max_length=10)),
                ('savings_family_friend', models.CharField(max_length=10)),
                ('savings_secret_place', models.CharField(max_length=10)),
                ('loan_bank', models.CharField(max_length=10)),
                ('loan_mobile_banking', models.CharField(max_length=10)),
                ('loan_sacco', models.CharField(max_length=10)),
                ('loan_microfinance', models.CharField(max_length=10)),
                ('loan_shylock', models.CharField(max_length=10)),
                ('loan_group_chama', models.CharField(max_length=10)),
                ('loan_govt', models.CharField(max_length=10)),
                ('loan_employer', models.CharField(max_length=10)),
                ('loan_family_friend', models.CharField(max_length=10)),
                ('loan_shopkeeper_cash', models.CharField(max_length=10)),
                ('loan_goods_credit', models.CharField(max_length=10)),
                ('loan_digital_app', models.CharField(max_length=10)),
                ('loan_agri_buyer_supplier', models.CharField(max_length=10)),
                ('loan_hire_purchase', models.CharField(max_length=10)),
                ('loan_mortgage', models.CharField(max_length=10)),
                ('insurance_motor', models.CharField(max_length=10)),
                ('insurance_home', models.CharField(max_length=10)),
                ('insurance_crop', models.CharField(max_length=10)),
                ('insurance_livestock', models.CharField(max_length=10)),
                ('insurance_nhif', models.CharField(max_length=10)),
                ('insurance_health_other', models.CharField(max_length=10)),
                ('insurance_life', models.CharField(max_length=10)),
                ('insurance_education', models.CharField(max_length=10)),
                ('insurance_other', models.CharField(max_length=10)),
                ('pension_nssf', models.CharField(max_length=10)),
                ('pension_mbao', models.CharField(max_length=10)),
                ('pension_other', models.CharField(max_length=10)),
                ('financially_excluded', models.CharField(max_length=10)),
            ],
        ),
    ]

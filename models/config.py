# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:49:01 2024

@author: gyuan
"""


cols_ad = dict()
cols_ad[0] = ["Age", "Gender","EducationLevel","BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality"]
cols_ad[1] = ["FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension"]
cols_ad[2] = ["SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides"]
cols_ad[3] = ["MMSE", "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL"]
cols_ad[4] = ["Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"]

cols_bcw = dict()
cols_bcw[0] = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean", "fractal_dimension_mean"]
cols_bcw[1] = ["radius_se", "texture_se","perimeter_se","area_se","smoothness_se", "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"]
cols_bcw[2] = ["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]


cols_cca = dict()
cols_cca[0] = ['FLAG_OWN_CAR',	'FLAG_OWN_REALTY',	'CNT_CHILDREN',	'AMT_INCOME_TOTAL',	'NAME_EDUCATION_TYPE',	'FLAG_WORK_PHONE',	'FLAG_PHONE',	'FLAG_EMAIL',	'CNT_FAM_MEMBERS',	'AGE',	'EMPLOYED_YEARS']
cols_cca[1] = ['recent_minor_dpd_count','recent_major_dpd_count','recent_minor_ratio','recent_major_ratio',
'recent_paid_ratio','recent_no_loan_ratio','recent_status_std',
'long_minor_ratio','long_major_ratio','total_history_length']


cols_hcdr=dict()
cols_hcdr[0] = ['FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL',
'AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
'AGE','EMPLOYED_YEARS','NAME_EDUCATION_TYPE', 'CNT_FAM_MEMBERS',
'EXT_SOURCE_2']
cols_hcdr[1] = ['CREDIT_MEAN','DEBT_MEAN','DEBT_RATIO','OVERDUE_RATIO',
'MAX_OVERDUE_LOG','ACTIVE_RATIO','HISTORY_YEARS','CREDIT_COUNT']
cols_hcdr[2] = ['DPD_MEAN', 'DPD_MAX', 
            'MINOR_RATIO', 'MAJOR_FLAG', 'PAY_RATIO_MEAN', 
            'INSTALL_COUNT']

#
cols = dict()
cols["AD"] = cols_ad
cols["BCW"] = cols_bcw
cols['CCA']= cols_cca
cols['HCDR'] = cols_hcdr
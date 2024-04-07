#!/usr/bin/env python
# coding: utf-8

# ## Подключение библиотек

# In[96]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from colorama import Fore, Back, Style 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj

init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff

get_ipython().run_line_magic('matplotlib', 'inline')

import xgboost
import lightgbm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


# ## Изучение данных

# In[39]:


heart_data = pd.read_csv('data.csv')
heart_data.head()


# In[40]:


heart_data.info()


# In[41]:


heart_data.describe()


# In[42]:


cmap = sns.diverging_palette(2, 165, s=80, l=55, n=9)
corrmat = heart_data.corr()
plt.subplots(figsize=(20,20))
sns.heatmap(corrmat, cmap=cmap, annot=True, square=True)


# 1. Фактор "time" (период наблюдения) является самым важным и имеет обратную зависимость от других факторов, поскольку очень важно как можно раньше диагностировать заболевание, чтобы получить своевременное лечение, тем самым снижая вероятность смертельного исхода.
# 2. Фактор "serum_creatinine" (уровень креатинина в сыворотке крови) не менее важен (корреляция со смертельным исходом 0.29), так как нормальное содержание креатинина облегчает работу сердца.
# 3. Фактор "ejecton_fraction" (фракции выброса) также оказывает значительное влияние на целевую переменную, потому что с фракциями выброса связана эффективность работы сердца.
# 4. Также можно заметить, что с возрастом работа сердца ухудшается

# #### Проверка факторов на потенциальные выбросы

# In[43]:


feature = ["age", "creatinine_phosphokinase", "ejection_fraction", 
           "platelets", "serum_creatinine", "serum_sodium", "time"]
for i in feature:
    plt.figure(figsize=(10,7))
    sns.swarmplot(x=heart_data["DEATH_EVENT"], y=heart_data[i], color='black', alpha=0.7)
    sns.boxenplot(x=heart_data["DEATH_EVENT"], y=heart_data[i], palette=["#CD5C5C","#FF0000"])
    plt.show()


# 1. Практически все факторы имеют выбросы.
# 2. Учитывая размер набора данных и его релевантность, не следует исключать такие выбросы при предварительной обработке данных, чтобы не совершить какую-либо статистическую ошибку.

# In[44]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = heart_data['age'],
    xbins = dict(
        start=40,
        end=95,
        size=2
    ),
    marker_color='#e8ab60',
    opacity=1
))

fig.update_layout(
    title_text='РАСПРЕДЕЛЕНИЕ ПО ВОЗРАСТУ',
    xaxis_title_text='ВОЗРАСТ', 
    yaxis_title_text='КОЛИЧЕСТВО',
    bargap=0.05,
    xaxis = {'showgrid' : False},
    yaxis = {'showgrid' : False},
    template = 'plotly_dark'
)

fig.show()


# Разброс данных достаточно высокий
# 

# In[45]:


fig = px.histogram(heart_data, x="age", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns, 
                   title ="Влияние ВОЗРАСТА на ВЫЖИВАЕМОСТЬ", 
                   labels={"age": "ВОЗРАСТ"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()


# Для визуализации данных использовались столбчатая и скрипичная диаграммы.
# Принцип работы скрипичных графиков: более широкие участки графика скрипки представляют более высокую вероятность заданного значения, более тонкие - меньшую вероятность.

# In[46]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = heart_data['creatinine_phosphokinase'],
    xbins=dict(
        start=23,
        end=582,
        size=15
    ),
    marker_color='#FE6F5E',
    opacity=1
))

fig.update_layout(
    title_text='РАСПРЕЛЕНИЕ ПО УРОВНЮ КРЕАТИНФОСФОКИНАЗЫ В КРОВИ',
    xaxis_title_text='УРОВЕНЬ КРЕАТИНФОСФОКИНАЗЫ (мкг/л)',
    yaxis_title_text='КОЛИЧЕСТВО',
    bargap=0.05,
    xaxis = {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()


# In[47]:


fig = px.histogram(heart_data, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns,
                   title ="Влияние УРОВНЯ КРЕАТИНФОСФОКИНАЗЫ на ВЫЖИВАЕМОСТЬ", 
                   labels={"creatinine_phosphokinase": "УРОВЕНЬ КРЕАТИНФОСФОКИНАЗЫ (мкг/л)"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[48]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = heart_data['ejection_fraction'],
    xbins=dict(
        start=14,
        end=70,
        size=2
    ),
    marker_color='#A7F432',
    opacity=1
))

fig.update_layout(
    title_text='РАСПРЕДЕЛЕНИЕ ПО ФРАКЦИИ ВЫБРОСА',
    xaxis_title_text='ФРАКЦИЯ ВЫБРОСА (%)',
    yaxis_title_text='КОЛИЧЕСТВО',
    bargap=0.05,
    xaxis = {'showgrid': False},
    yaxis = {'showgrid': False},
    template = 'plotly_dark'
)

fig.show()


# In[49]:


fig = px.histogram(heart_data, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns,
                   title ="Влияние ФРАКЦИИ ВЫБРОСА на ВЫЖИВАЕМОСТЬ", 
                   labels={"ejection_fraction": "ФРАКЦИЯ ВЫБРОСА (%)"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[50]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = heart_data['platelets'],
    xbins=dict(
        start=25000,
        end=300000,
        size=5000
    ),
    marker_color='#50BFE6',
    opacity=1
))

fig.update_layout(
    title_text='РАСПРЕДЕЛЕНИЕ ПО КОЛИЧЕСТВУ ТРОМБОЦИТОВ',
    xaxis_title_text='КОЛИЧЕСТВО ТРОМБОЦИТОВ (на 1 мл крови)',
    yaxis_title_text='КОЛИЧЕСТВО',
    bargap=0.05,
    xaxis = {'showgrid': False},
    yaxis = {'showgrid': False},
    template = 'plotly_dark'
)

fig.show()


# In[51]:


fig = px.histogram(heart_data, x="platelets", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns,
                   title ="Влияние КОЛИЧЕСТВА ТРОМБОЦИТОВ на ВЫЖИВАЕМОСТЬ", 
                   labels={"platelets": "КОЛИЧЕСТВО ТРОМБОЦИТОВ (на 1 мл крови)"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[52]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = heart_data['serum_creatinine'],
    xbins=dict(
        start=0.5,
        end=9.4,
        size=0.2
    ),
    marker_color='#E77200',
    opacity=1
))

fig.update_layout(
    title_text='РАСПРЕДЕЛЕНИЕ ПО УРОВНЮ КРЕАТИНИНА',
    xaxis_title_text='КРЕАТИНИН (мг/дл)',
    yaxis_title_text='КОЛИЧЕСТВО', 
    bargap=0.05,
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()


# In[53]:


fig = px.histogram(heart_data, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns,
                   title ="Влияние УРОВНЯ КРЕАТИНИНА на ВЫЖИВАЕМОСТЬ", 
                   labels={"serum_creatinine": "КРЕАТИНИН (мг/дл)"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[54]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = heart_data['serum_sodium'],
    xbins=dict(
        start=113,
        end=148,
        size=1
    ),
    marker_color='#AAF0D1',
    opacity=1
))

fig.update_layout(
    title_text='РАСПРЕДЕЛЕНИЕ ПО УРОВНЮ НАТРИЯ',
    xaxis_title_text='НАТРИЙ (мэкв/л)',
    yaxis_title_text='КОЛИЧЕСТВО', 
    bargap=0.05,
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()


# In[55]:


fig = px.histogram(heart_data, x="serum_sodium", color="DEATH_EVENT", marginal="violin",hover_data=heart_data.columns,
                   title ="Влияние УРОВНЯ НАТРИЯ на ВЫЖИВАЕМОСТЬ", 
                   labels={"serum_sodium": "НАТРИЙ (мэкв/л)"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[56]:


d1 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["sex"]==1)]
d2 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["sex"]==1)]
d3 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["sex"]==0)]
d4 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["sex"]==0)]

label1 = ["Мужчина", "Женщина"]
label2 = ["Мужчина - Выжил", "Мужчина - Умер", "Женщина - Выжила", "Женщина - Умерла"]
values1 = [(len(d1)+len(d2)), (len(d3)+len(d4))]
values2 = [len(d1), len(d2), len(d3), len(d4)]

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="ПОЛ"), 1, 1)
fig.add_trace(go.Pie(labels=label2, values=values2, name="ПОЛ - ВЫЖИВАЕМОСТЬ"), 1, 2)

fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="РАСПРЕДЕЛЕНИЕ ПО ПОЛУ В НАБОРЕ ДАННЫХ \
                    ПОЛ - ВЫЖИВАЕМОСТЬ",
    annotations=[dict(text='ПОЛ', x=0.21, y=0.5, font_size=10, showarrow=False),
                dict(text='ПОЛ - ВЫЖИВАЕМОСТЬ', x=0.837, y=0.5, font_size=9, showarrow=False)],
    autosize=False, width=1200, height=500, paper_bgcolor="white")

fig.show()


# #### ВЫВОД: Из приведённых выше диаграмм можно сделать вывод что, в наборе данных 64,9% мужчин (из которых 44,1% выжило и 20,7% умерло) и 35,1% женщин (из которых 23,7% выжило и 11,4% умерло).

# In[57]:


d1 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["diabetes"]==0)]
d2 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["diabetes"]==1)]
d3 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["diabetes"]==0)]
d4 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["diabetes"]==1)]

label1 = ["НЕТ диабета","ЕСТЬ диабет"]
label2 = ["НЕТ диабета - Выжил(-а)", "ЕСТЬ диабет - Выжил(-а)", "НЕТ диабета - Умер(-ла)", "ЕСТЬ диабет - Умер(-ла)"]
values1 = [(len(d1)+len(d3)), (len(d2)+len(d4))]
values2 = [len(d1),len(d2),len(d3),len(d4)]

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="ДИАБЕТ"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=values2, name="ДИАБЕТ - ВЫЖИВАЕМОСТЬ"),
              1, 2)
              
fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="РАСПРЕДЕЛЕНИЕ ПО НАЛИЧИЮ ДИАБЕТА \
                  ДИАБЕТ - ВЫЖИВАЕМОСТЬ",
    annotations=[dict(text='ДИАБЕТ', x=0.196, y=0.5, font_size=12, showarrow=False),
                 dict(text='ДИАБЕТ - ВЫЖИВАЕМОСТЬ', x=0.84, y=0.5, font_size=8, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")

fig.show()


# #### ВЫВОД: В наборе данных 58,2% пациентов не страдают диабетом (из которых 39,5% выжило и 18,7% умерло) и 41,8% пациентов больны диабетом (из которых 28,4% выжило и 13,4% умерло).

# In[58]:


d1 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["anaemia"]==0)]
d2 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["anaemia"]==0)]
d3 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["anaemia"]==1)]
d4 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["anaemia"]==1)]

label1 = ["НЕТ анемии","ЕСТЬ анемия"]
label2 = ['НЕТ анемии - Выжил(-а)','НЕТ анемии - Умер(-ла)', "ЕСТЬ анемия - Выжил(-а)", "ЕСТЬ анемия - Умер(-ла)"]
values1 = [(len(d1)+len(d2)), (len(d3)+len(d4))]
values2 = [len(d1),len(d2),len(d3),len(d4)]

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="АНЕМИЯ"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=values2, name="АНЕМИЯ - ВЫЖИВАЕМОСТЬ"),
              1, 2)

fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="РАСПРЕДЕЛЕНИЕ ПО НАЛИЧИИ АНЕМИИ \
                  АНЕМИЯ - ВЫЖИВАЕМОСТЬ",
    annotations=[dict(text='АНЕМИЯ', x=0.194, y=0.5, font_size=12, showarrow=False),
                 dict(text='АНЕМИЯ - ВЫЖИВАЕМОСТЬ', x=0.842, y=0.5, font_size=8, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")

fig.show()


# #### ВЫВОД: В наборе данных 56,9% пациентов не больны анемией (из них 40,1% выжило и 16,7% умерло) и 43,1% пациентов страдают от анемии (из них 27,8% выжило и 15,4% умерло).

# In[59]:


d1 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["high_blood_pressure"]==0)]
d2 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["high_blood_pressure"]==0)]
d3 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["high_blood_pressure"]==1)]
d4 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["high_blood_pressure"]==1)]
label1 = ["НЕТ гипертонии","ЕСТЬ гипертония"]
label2 = ['НЕТ гипертонии - Выжил(-а)', 'НЕТ гипертонии - Умер(-ла)', "ЕСТЬ гипертония - Выжил(-а)", "ЕСТЬ гипертония - Умер(-ла)"]
values1 = [(len(d1)+len(d2)), (len(d3)+len(d4))]
values2 = [len(d1),len(d2),len(d3),len(d4)]

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="ГИПЕРТОНИЯ"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=values2, name="ГИПЕРТОНИЯ - ВЫЖИВАЕМОСТЬ"),
              1, 2)

fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="РАСПРЕДЕЛЕНИЕ ПО НАЛИЧИЮ ГИПЕРТОНИИ \
                  ГИПЕРТОНИЯ - ВЫЖИВАЕМОСТЬ",
    annotations=[dict(text='ГИПЕРТОНИЯ', x=0.177, y=0.5, font_size=12, showarrow=False),
                 dict(text='ГИПЕРТОНИЯ - ВЫЖИВАЕМОСТЬ', x=0.845, y=0.5, font_size=7, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")
fig.show()


# #### ВЫВОД: В наборе данных у 64,9% пациентов нет гипертонии (из которых 45,8% выжило и 19,1% умерло) и у 35,1% - есть гипертония (из которых 22,1% выжило и 13% умерло).

# In[60]:


d1 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["smoking"]==0)]
d2 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["smoking"]==0)]
d3 = heart_data[(heart_data["DEATH_EVENT"]==0) & (heart_data["smoking"]==1)]
d4 = heart_data[(heart_data["DEATH_EVENT"]==1) & (heart_data["smoking"]==1)]

label1 = ["НЕ курит","Курит"]
label2 = ['НЕ курит - Выжил(-а)','НЕ курит - Умер(-ла)', "Курит - Выжил(-а)", "Курит - Умер(-ла)"]
values1 = [(len(d1)+len(d2)), (len(d3)+len(d4))]
values2 = [len(d1),len(d2),len(d3),len(d4)]

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="КУРЕНИЕ"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=values2, name="КУРЕНИЕ - ВЫЖИВАЕМОСТЬ"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="РАСПРЕДЕЛЕНИЕ ПО КУРЕНИЮ \
                  КУРЕНИЕ - ВЫЖИВАЕМОСТЬ",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='КУРЕНИЕ', x=0.195, y=0.5, font_size=12, showarrow=False),
                 dict(text='КУРЕНИЕ - ВЫЖИВАЕМОСТЬ', x=0.842, y=0.5, font_size=8, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")
fig.show()


# #### ВЫВОД: В наборе входных данных 67,9% некурящих пациентов (из которых 45,8% выжило и 22,1% умерло) и 32,1% - курящих (из которых 22,1% выжило и 10% умерло).

# In[61]:


fig = px.histogram(heart_data, x="age", color="diabetes", marginal="violin",hover_data=heart_data.columns,
                   title ="РАСПРЕДЕЛЕНИЕ ПО ВОЗРАСТУ И НАЛИЧИЮ ДИАБЕТА", 
                   labels={"diabetes": "ДИАБЕТ", "age": "ВОЗРАСТ"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[62]:


fig = px.histogram(heart_data, x="age", color="anaemia", marginal="violin",hover_data=heart_data.columns,
                   title ="РАСПРЕДЕЛЕНИЕ ПО ВОЗРАСТУ И НАЛИЧИЮ АНЕМИИ", 
                   labels={"anaemia": "АНЕМИЯ", "age": "ВОЗРАСТ"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[63]:


fig = px.histogram(heart_data, x="age", color="high_blood_pressure", marginal="violin",hover_data=heart_data.columns,
                   title ="РАСПРЕДЕЛЕНИЕ ПО ВОЗРАСТУ И НАЛИЧИЮ ГИПЕРТОНИИ", 
                   labels={"high_blood_pressure": "ГИПЕРТОНИЯ", "age": "ВОЗРАСТ"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# In[64]:


fig = px.histogram(heart_data, x="age", color="smoking", marginal="violin",hover_data=heart_data.columns,
                   title ="РАСПРЕДЕЛЕНИЕ ПО ВОЗРАСТУ И КУРЕНИЮ", 
                   labels={"smoking": "КУРЕНИЕ", "age": "ВОЗРАСТ"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()


# ## Моделирование данных и обучение

# In[313]:


# Деление датасета на обучающую и тестовую выборки
x = heart_data.iloc[:, [4,7,11]].values
y = heart_data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=2)


# In[314]:


print(x_train)


# In[315]:


print(y_test)


# In[419]:


# Масштабирование объектов

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[420]:


accuracy_list = []


# ### 1. Логистическая регрессия (Logistic Regression) 

# In[421]:


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
accuracy_list.append(100*log_reg_acc)


# In[422]:


print(Fore.GREEN + "Точность логистической регрессии:", "{:.2f}%".format(100* log_reg_acc))


# ### 2. Support Vector (SVC)

# In[423]:


sv_clf = SVC()
sv_clf.fit(x_train, y_train)
sv_clf_pred = sv_clf.predict(x_test)
sv_clf_acc = accuracy_score(y_test, sv_clf_pred)
accuracy_list.append(100*sv_clf_acc)


# In[424]:


print(Fore.GREEN + "Точность SVC:", "{:.2f}%".format(100* sv_clf_acc))


# ### 3. Метод k-ближайших соседей (K Neighbors Classifier)

# In[425]:


kn_clf = KNeighborsClassifier(n_neighbors=6)
kn_clf.fit(x_train, y_train)
kn_pred = kn_clf.predict(x_test)
kn_acc = accuracy_score(y_test, kn_pred)
accuracy_list.append(100*kn_acc)


# In[426]:


print(Fore.GREEN + "Точность метода k-ближайших соседей:", "{:.2f}%".format(100* kn_acc))


# ### 4. Дерево решений (Decision Tree Classifier)

# In[427]:


dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
dt_acc = accuracy_score(y_test, dt_pred)
accuracy_list.append(100*dt_acc)


# In[428]:


print(Fore.GREEN + "Точность дерева решений:", "{:.2f}%".format(100* dt_acc))


# ### 5. Случайный лес (Random Forest Classifier)

# In[429]:


r_clf = RandomForestClassifier(n_estimators=50, max_features=0.5, max_depth=15, random_state=1)
r_clf.fit(x_train, y_train)
r_pred = r_clf.predict(x_test)
r_acc = accuracy_score(y_test, r_pred)
accuracy_list.append(100*r_acc)


# In[430]:


print(Fore.GREEN + "Точность случайного леса:", "{:.2f}%".format(100* r_acc))


# ### 6. Градиентный бустинг (Gradient Boosting Classifier)

# In[431]:


gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=1)
gradientboost_clf.fit(x_train,y_train)
gradientboost_pred = gradientboost_clf.predict(x_test)
gradientboost_acc = accuracy_score(y_test, gradientboost_pred)
accuracy_list.append(100*gradientboost_acc)


# In[432]:


print(Fore.GREEN + "Точность градиентного бустинга:", "{:.2f}%".format(100* gradientboost_acc))


# ### 7. XGBoost Classifier

# In[433]:


xgb_clf = xgboost.XGBRFClassifier(max_depth=3, random_state=1)
xgb_clf.fit(x_train,y_train)
xgb_pred = xgb_clf.predict(x_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
accuracy_list.append(100*xgb_acc)


# In[434]:


print(Fore.GREEN + "Точность XGBoost Classifier:", "{:.2f}%".format(100* xgb_acc))


# ### 8. Усиленный световой градиент (LightGBM Classifier)

# In[435]:


lgb_clf = lightgbm.LGBMClassifier(max_depth=2, random_state=4)
lgb_clf.fit(x_train,y_train)
lgb_pred = lgb_clf.predict(x_test)
lgb_acc = accuracy_score(y_test, lgb_pred)
accuracy_list.append(100*lgb_acc)


# In[436]:


print(Fore.GREEN + "Точность LightGBM Classifier:","{:.2f}%".format(100* lgb_acc))


# ### 9. CatBoost Classifier

# In[437]:


cat_clf = CatBoostClassifier()
cat_clf.fit(x_train,y_train)
cat_pred = cat_clf.predict(x_test)
cat_acc = accuracy_score(y_test, cat_pred)
accuracy_list.append(100*cat_acc)


# In[438]:


print(Fore.GREEN + "Точность CatBoost Classifier:","{:.2f}%".format(100* cat_acc))


# ## Заключение

# In[439]:


model_list = ['Logistic Regression', 'SVC', 'K Nearest Neighbors', 'Decision Tree', 'Random Forest',
             'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']


# In[445]:


plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=accuracy_list, palette = "crest", saturation =2.0)
plt.xlabel('Модели классификации', fontsize = 20 )
plt.ylabel('Точность (%)', fontsize = 20)
plt.title('Точность различных моделей классификации', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


#     • Логистическая регрессия (Logistic Regression): 90.00%
#     • Support Vector (SVC): 91.67%
#     • Метод k-ближайших соседей (K-Nearest Neigbors): 88.33%
#     • Дерево решений (Decision Tree): 90.00%
#     • Случайный лес (Random Forest): 90.00%
#     • Градиентный бустинг (Gradient Boosting): 93.33%
#     • XGBoost: 93.33%
#     • Усиленный световой градиент (LightGBM Classifier: 86.67%
#     • CatBoost: 93.33%

# In[ ]:





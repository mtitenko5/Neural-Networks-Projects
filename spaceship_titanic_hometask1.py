import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',50)
train=pd.read_csv('C:/Users/titen/Downloads/train1.csv')
test=pd.read_csv('C:/Users/titen/Downloads/test1.csv')
print("Dimension train: ", train.shape)
print("Dimension test: ", test.shape)

train["Transported"]=train["Transported"].astype(int)
#print(train.describe())
#print(train.describe(include=['O']))
#print(train.info())
#print(train.isnull().sum())

transported = train[train['Transported'] == 1]
not_transported = train[train['Transported'] == 0]

print ("Transported: %i (%.1f%%)" %(len(transported), train['Transported'].mean()*100.0))
print ("Not Transported: %i (%.1f%%)" %(len(not_transported), (1-train['Transported'].mean())*100.0))
print ("Total: %i"%len(train))

fig1 = plt.figure(figsize=(15,10))
ax11 = fig1.add_subplot(221)
ax12 = fig1.add_subplot(222)
ax13 = fig1.add_subplot(223)
ax14 = fig1.add_subplot(224)
#print(train[['VIP', 'Transported']].groupby('VIP').value_counts())
vip_train=train[['VIP', 'Transported']].groupby('VIP', as_index = False).mean()
sns.barplot(data=vip_train, x='VIP', y='Transported', ax=ax11)
#print(train.groupby('HomePlanet').Transported.value_counts())
home_train=train[['HomePlanet', 'Transported']].groupby('HomePlanet', as_index = False).mean()
sns.barplot(data=home_train, x='HomePlanet', y='Transported',ax=ax12)
sns.pointplot(x='HomePlanet', y='Transported', hue='CryoSleep', data=train, errorbar=None, ax=ax13)
cross_tab = pd.crosstab(train['HomePlanet'], train['CryoSleep'])
cross_tab.div(cross_tab.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, ax=ax14)
ax14.set_xlabel('HomePlanet')
ax14.set_ylabel('Percentage')
plt.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(15,10))
ax21 = fig2.add_subplot(211)
ax22 = fig2.add_subplot(212)
sns.violinplot(x="HomePlanet", y="Age", hue="Transported", data=train, split=True, ax=ax21)
ax22.set_xlabel('Age')
sns.histplot(data=transported['Age'].dropna(), bins=range(0, 80, 1), kde=False, color='blue', ax = ax22)
sns.histplot(data=not_transported['Age'].dropna(), bins=range(0, 80, 1), kde=False, color='red', ax = ax22)
plt.tight_layout()
plt.show()

train_test_data = [train, test]
train["VIP"]=train["VIP"].fillna(0).astype(int)
train["CryoSleep"]=train["CryoSleep"].fillna(0).astype(int)
test["VIP"]=test["VIP"].fillna(0).astype(int)
test["CryoSleep"]=test["CryoSleep"].fillna(0).astype(int)

#print(train['HomePlanet'].value_counts())
for dataset in train_test_data:
    dataset['HomePlanet'] = dataset['HomePlanet'].fillna('Earth')
home_mapping = {"Earth": 1, "Europa": 2, "Mars": 3}
for dataset in train_test_data:
    dataset['HomePlanet'] = dataset['HomePlanet'].map(home_mapping)
    dataset['HomePlanet']=dataset['HomePlanet'].fillna(1)

#print(train['Destination'].value_counts())
for dataset in train_test_data:
    dataset['Destination'] = dataset['Destination'].fillna('TRAPPIST-1e')
dest_mapping = {"TRAPPIST-1e": 1, "55 Cancri e": 2, "PSO J318.5-22": 3}
for dataset in train_test_data:
    dataset['Destination'] = dataset['Destination'].map(dest_mapping)
    dataset['Destination']=dataset['Destination'].fillna(1)

for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_nan_count = dataset['Age'].isnull().sum()
    age_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_nan_count)
    null_indices=dataset['Age'].isnull()
    dataset.loc[null_indices, 'Age'] = age_random_list
    dataset['Age'] = dataset['Age'].astype(int)
for dataset in train_test_data:
    dataset['AgeGroup'] = pd.cut(train['Age'],5)

#print(train[['AgeGroup', 'Transported']].groupby(['AgeGroup'], as_index=False).mean())
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 15.8, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 15.8) & (dataset['Age'] <= 31.6), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 31.6) & (dataset['Age'] <= 47.4), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 47.4) & (dataset['Age'] <= 63.2), 'Age'] = 3
    dataset.loc[dataset['Age'] > 63.2, 'Age'] = 4

print("Number of people who didn''t buy smth on FoodCourt: ", train[train['FoodCourt']==0.0].shape[0])
foodcourt_median=train['FoodCourt'].median()
for dataset in train_test_data:
    dataset['FoodCourt'] = dataset['FoodCourt'].fillna(foodcourt_median)
for dataset in train_test_data:
    dataset['FoodGroup'] = pd.qcut(train['FoodCourt'], 4, duplicates='drop')
for dataset in train_test_data:
      dataset.loc[dataset['FoodCourt'] <= 61.0, 'FoodCourt'] = 0
      dataset.loc[dataset['FoodCourt'] > 61.0, 'FoodCourt'] = 1
      dataset['FoodCourt'] = dataset['FoodCourt'].astype(int)

mall_median=train['ShoppingMall'].median()
for dataset in train_test_data:
    dataset['ShoppingMall'] = dataset['ShoppingMall'].fillna(mall_median)
for dataset in train_test_data:
    dataset['MallGroup'] = pd.qcut(train['ShoppingMall'], 4, duplicates='drop')
for dataset in train_test_data:
      dataset.loc[dataset['ShoppingMall'] <= 22.0, 'ShoppingMall'] = 0
      dataset.loc[dataset['ShoppingMall'] > 22.0, 'ShoppingMall'] = 1
      dataset['ShoppingMall'] = dataset['ShoppingMall'].astype(int)

features_drop = ['Name', 'RoomService', 'Spa', 'VRDeck', 'Cabin', 'AgeGroup', 'FoodGroup', 'MallGroup']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

corr_matrix = train.corr(numeric_only=True)
fig=plt.figure(figsize=[6,6])
sns.heatmap(corr_matrix, vmax=0.6, square=True, annot=True)
plt.show()

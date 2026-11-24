# NAME : A.DIVIYADHARSHINI
# REG NO : 212224040080
# EXNO2DS
# AIM:
To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dt = pd.read_csv("/content/titanic_dataset.csv")
dt
```

<img width="1551" height="527" alt="image" src="https://github.com/user-attachments/assets/7f2194c7-8f75-4434-8df2-ef16bf4abfec" />


```
dt.info()
```

<img width="1357" height="480" alt="image" src="https://github.com/user-attachments/assets/04d2b8e3-2604-47b8-b830-13740908ed67" />


```
dt.shape

```
<img width="1292" height="85" alt="image" src="https://github.com/user-attachments/assets/9e74bf5f-5608-4c1c-b1fd-44f7c17b3232" />


```
dt.set_index("PassengerId",inplace=True)
dt

```
<img width="1498" height="631" alt="image" src="https://github.com/user-attachments/assets/fa69aeb7-1aa1-478b-92b2-cf3c75934c36" />

```
print("Data types:")
display(dt.dtypes.to_frame("Data Type"))

```
<img width="400" height="513" alt="image" src="https://github.com/user-attachments/assets/cd160f4c-20dc-4e14-8840-3dcbdd7e8e7e" />

```

print("Missing values:")
missing=pd.DataFrame({
    "Missing count":dt.isnull().sum(),
    "Missing percentage":(dt.isna().mean()*100).round(2)
})
display(missing.sort_values("Missing percentage",ascending=False))

```
<img width="655" height="512" alt="image" src="https://github.com/user-attachments/assets/ea0a50f6-f118-425b-971c-0a71bcca7e3a" />

```
print("Exact duplicate rows:",dt.duplicated().sum())
print("Numeric summary")
display(dt.describe())

```
<img width="875" height="502" alt="image" src="https://github.com/user-attachments/assets/116c1801-451b-424d-837d-e32216ee3029" />

```
dt.nunique()

```
<img width="438" height="571" alt="image" src="https://github.com/user-attachments/assets/777dbe80-de06-4ece-a979-2706a08687cb" />

```

dt["Survived"].value_counts()

```
<img width="931" height="262" alt="image" src="https://github.com/user-attachments/assets/e6143a9a-4fa8-4c73-814b-9b41e08701a3" />

```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per

```
<img width="1052" height="283" alt="image" src="https://github.com/user-attachments/assets/6a58234f-63eb-4258-bfd0-8bf8ebc02266" /> 

```
sns.countplot(data=dt, x="Survived")

```
<img width="915" height="615" alt="image" src="https://github.com/user-attachments/assets/d6683d05-7905-4b07-975a-f9c4b922c54b" /> 

```
dt.Pclass.unique()

```
<img width="953" height="85" alt="image" src="https://github.com/user-attachments/assets/bfad25b3-805a-4e43-9d7d-c708363eff4e" />

```
dt.rename(columns={'Sex':'Gender'},inplace=True) 
dt
```

<img width="1497" height="635" alt="image" src="https://github.com/user-attachments/assets/060b6b87-8369-4b85-89db-aea153cdbaa3" />

```
sns.catplot(x='Gender',col='Survived',kind='count',data=dt,height=5,aspect=.7)

```
<img width="1060" height="685" alt="image" src="https://github.com/user-attachments/assets/22cd4b4f-dd41-4985-b734-13952e5bb5a7" />

```
sns.catplot(x='Survived', hue="Gender", data=dt, kind = "count")

```

<img width="960" height="688" alt="image" src="https://github.com/user-attachments/assets/f30c3808-dd16-42c7-887f-79e6bbc1e784" />

```
dt.boxplot(column='Age',by='Survived')

```

<img width="1183" height="650" alt="image" src="https://github.com/user-attachments/assets/4fb85f2b-8eef-45fe-8d40-70a90ef1aae0" /> 

```
sns.scatterplot(x=dt['Age'],y=dt['Fare'])

```
<img width="953" height="575" alt="image" src="https://github.com/user-attachments/assets/a351c1d7-54db-4b93-8b11-57a98ba497c7" />

```
sns.jointplot(x="Age",y="Fare", data=dt) 

```
<img width="1460" height="737" alt="image" src="https://github.com/user-attachments/assets/2e07d308-4782-4e74-952f-03089dbd5091" />

```
fig, ax1= plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x="Pclass",y='Age', hue='Gender', data=dt)

```
<img width="1125" height="632" alt="image" src="https://github.com/user-attachments/assets/a5e8e2e0-1b4f-459e-abfe-4fb34901d709" />

```

sns.catplot(data=dt, col ="Survived", x = "Gender", hue="Pclass", kind = "count")

```
<img width="1567" height="695" alt="image" src="https://github.com/user-attachments/assets/d4dd2147-c2dd-427a-bb4f-942b8ed6ca97" />

```
corr=dt.corr(numeric_only=True)

sns.heatmap(corr,annot=True,cmap='coolwarm')

```
<img width="1027" height="645" alt="image" src="https://github.com/user-attachments/assets/106121d4-f7dd-4d38-8d0b-7eed4afadb82" />

<img width="867" height="572" alt="image" src="https://github.com/user-attachments/assets/87342870-1a28-453e-87de-8aaf45e23390" />





# RESULT

The Program and The Output has been sucessfully Verified along with the Visualization .

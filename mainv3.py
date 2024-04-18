import numpy as np
import pandas as pd
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik, kategorik fakat carinal ve numerik görünümlü fakat kategorik olan değişkenleri belirler
    Parameters
    ----------
    dataframe: Dataframe
        Değişkenlerin türlerinin belirleneceği veri seti

    cat_th: int, optional
        Numerik fakat kategorik olan değişkenler için eşik değeri.

    car_th: int, optional
        Kategorik fakat kardinal olan değişkenler için eşik değeri.

    Returns
    -------
    cat_cols: list
        Kategorik değişkenlerin listesi.

    num_cols: list
        Numerik değişkenlerin listesi.

    cat_but_car: list
        Kategorik fakat karinal değişkenlerin listesi.

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

def analyze_col_by_target(dataframe, target, col):
    arr = dataframe[col].dropna().unique()
    for value in np.delete(arr, np.argwhere(arr == np.nan)):
        mean = dataframe[dataframe[col] == value][target].mean()
        print(f"Value : {value}, mean : {mean}")

    if dataframe[col].isnull().any():
        print(f"Value : NULL, mean : {dataframe[dataframe[col].isnull()][target].mean()}")

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 2000)
################################################################################################################################
df = pd.read_csv("train.csv")
################################################################################################################################
df.head()
df.shape # (1460, 81)
null_degisken_yüzdeleri = df.isnull().sum()[df.isnull().sum() != 0]/df.shape[0]*100
dropable_features = null_degisken_yüzdeleri[null_degisken_yüzdeleri > 45] # çok fazla null değer içeren değişkenler.
df.drop(dropable_features.index, axis=1, inplace=True) # Çok fazla eksik değer içeren değişkenleri veri setinden drop ettim.
################################################################################################################################
##Garage Related Cols(GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond)
#GarageType
analyze_col_by_target(df, "SalePrice", "GarageType")

df.loc[df["GarageType"] == "CarPort", "GarageType"] = 1
df.loc[df["GarageType"] == "Detchd", "GarageType"] = 2
df.loc[df["GarageType"] == "2Types", "GarageType"] = 3
df.loc[df["GarageType"] == "Basment", "GarageType"] = 4
df.loc[df["GarageType"] == "Attchd", "GarageType"] = 5
df.loc[df["GarageType"] == "BuiltIn", "GarageType"] = 6
df["GarageType"].fillna(0, inplace=True)

#GarageYrBlt
df["GarageYrBltQCut"] = pd.qcut(df["GarageYrBlt"], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df["GarageYrBltQCut"] = df["GarageYrBltQCut"].cat.set_categories([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df["GarageYrBltQCut"].fillna(0, inplace=True)
df.drop(columns="GarageYrBlt", inplace=True)

#GarageFinish
analyze_col_by_target(df, "SalePrice", "GarageFinish")

df.loc[df["GarageFinish"] == "Unf", "GarageFinish"] = 1
df.loc[df["GarageFinish"] == "RFn", "GarageFinish"] = 2
df.loc[df["GarageFinish"] == "Fin", "GarageFinish"] = 3
df["GarageFinish"].fillna(0, inplace=True)

#GarageQual
analyze_col_by_target(df, "SalePrice", "GarageQual")

df.loc[df["GarageQual"] == "Po", "GarageQual"] = 1
df.loc[df["GarageQual"] == "Fa", "GarageQual"] = 2
df.loc[df["GarageQual"] == "TA", "GarageQual"] = 3
df.loc[df["GarageQual"] == "Gd", "GarageQual"] = 4
df.loc[df["GarageQual"] == "Ex", "GarageQual"] = 5
df["GarageQual"].fillna(0, inplace=True)

#GarageCond
analyze_col_by_target(df, "SalePrice", "GarageCond")
df.loc[df["GarageCond"] == "Po", "GarageCond"] = 1
df.loc[df["GarageCond"] == "Fa", "GarageCond"] = 2
df.loc[df["GarageCond"] == "TA", "GarageCond"] = 3
df.loc[df["GarageCond"] == "Gd", "GarageCond"] = 4
df.loc[df["GarageCond"] == "Ex", "GarageCond"] = 5
df["GarageCond"].fillna(0, inplace=True)

#GarageArea
df["GarageAreaQCut"] = pd.qcut(df["GarageArea"], 5, labels=[1, 2, 3, 4, 5])
df["GarageAreaQCut"] = df["GarageAreaQCut"].cat.set_categories([0, 1, 2, 3, 4, 5])
df["GarageAreaQCut"].fillna(0, inplace=True)
df.drop(columns="GarageArea", inplace=True)

#Creating a single garage variable from garage related cols(GarageAreaQCut, GarageQual, GarageYrBltQCut, GarageType, GarageCars, GarageCond, GarageFinish)
garage_cols = [col for col in df.columns if "Garage" in col]
col_coefficients = [20, 4, 7, 10, 10, 12, 15]
df["GaragePoint"] = 0
for index in range(len(garage_cols)):
    df["GaragePoint"] = df["GaragePoint"] + df[garage_cols[index]].astype('int') * col_coefficients[index]

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
properties_with_missing_values = df.isnull().sum()[df.isnull().sum() != 0].index
#properties_with_missing_values = [col for col in properties_with_missing_values if "Garage" not in col]
for i in properties_with_missing_values:
    if i in num_cols:
        df[i].fillna(df[i].mean(), inplace=True) # numerik değişkenleri ortalama ile
    else:
        df[i].fillna(df[i].mode()[0], inplace=True) # kategorik değişkenleri modu ile
################################################################################################################################
def outlier_thresholds(dataframe, col_name, q1_rate=0.25, q3_rate= 0.75):
    """
    Veri setindeki belirtilen değişken için aykırı değerlerin alt ve üst sınırını hesaplar.
    Args:
        dataframe: Dataframe
                Aykırı değer sınırları belirlenecek olan değişkenin bulunduğu veri seti
        col_name: String
                Aykırı değer sınırları belirlenecek olan değişken
        q1_rate: float
                IQR hesabı yapmak için kullanılacak olan ilk yüzdelik.
        q3_rate: float
                IQR hesabı yapmak için kullanılacak olan ikinci yüzdelik.

    Returns:
        low_limit : float
                Aykırı değerlerin alt sınırı.
        up_limit : float
                Aykırı değerlerin üst sınırı.

    """
    quartile1 = dataframe[col_name].quantile(q1_rate)
    quartile3 = dataframe[col_name].quantile(q3_rate)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, return_index=False,q1_rate=0.25, q3_rate= 0.75):
    """
    Verilen veri setindeki belirtilen değişkende aykırı değer olup olmadığını varsa index'ini döndürür.
    Args:
        dataframe: dataframe
                Aykırı değer kontrolü yapılacak değişkenin içinde bulunduğu veri seti
        col_name: string
                Aykırı değer kontrolü yapılacak değişkenin ismi.
        return_index: bool
                Aykırı değer kontrolü yapıldıktan sonra aykırı değerlerin index'i döndürülecek mi kontrolü yapan bool değişken

    Returns:
        True : bool
            Değişken aykırı değer içeriyordiği anlamına gelir

        False : bool
            Değişken aykırı değer içermediği anlamına gelir

        outlier_index : liste
            Aykırı değerlerin indexlerini barındıran liste

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1_rate, q3_rate)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        if return_index:
            outlier_index = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].index
            return outlier_index
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
check_outlier(df,"MSSubClass")

num_cols = [col for col in num_cols if "Garage" not in col]
for i in num_cols:
    #if df[i].dtype != 'O':
    print(i, check_outlier(df,i))

features_with_outliers = [col for col in num_cols if check_outlier(df, col) ]

for i in features_with_outliers:
   replace_with_thresholds(df, i)
################################################################################################################################
df.head()
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
for i in  cat_cols:
    print(i+"\n",df[i].value_counts())
################################################################################################################################
def mean_count_fonk(dataframe,col_name):
    mean_df = dataframe.groupby(col_name).agg({"SalePrice": "mean"})
    count_df = dataframe.groupby(col_name).agg({"SalePrice": "count"}) / df.shape[0] * 100
    return pd.concat([mean_df, count_df], axis=1)
def rare_fonk(dataframe, col_name, rare_ratio=3):
    ratio_df = dataframe.groupby(col_name).agg({"SalePrice": "count"}) / df.shape[0] * 100
    index = ratio_df[ratio_df[ratio_df < rare_ratio].notnull()["SalePrice"]].index
    return index


################################################################################################################################
mean_count_fonk(df,"MSZoning") #C, FV, RH sınıflarını rare olarak atadıktan sonra RL ve RM arasındaki büyüklük tanımlanabilir bundan dolayı label encoding yapacağım.
df["MSZoning"] = df["MSZoning"].apply(lambda x: "rare" if x in ['C (all)', 'FV', 'RH'] else x)
df["MSZoning"].unique()
MSZoning_order = {'RL': 0, 'RM': 1, 'rare': 2}
df['MSZoning_LabelEncoded'] = df['MSZoning'].map(MSZoning_order)
df.drop("MSZoning",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Street")
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Street_LabelEncoded'] = label_encoder.fit_transform(df['Street'])
label_encoder.inverse_transform([0,1]) # ['Grvl', 'Pave']
df.drop("Street",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"LotShape")
df["LotShape"] = df["LotShape"].apply(lambda x: "rare" if x in ["IR2","IR3"] else x)
LotShape_order = {"Reg":0, "IR1":1, "rare":2}
df["LotShape_LabelEncoded"] = df["LotShape"].map(LotShape_order)
df.drop("LotShape",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"LandContour")
df["LandContour"] = df["LandContour"].apply(lambda x: "rare" if x in rare_fonk(df,"LandContour",5) else x)
LandContour_order = {"Lvl":1, "rare":0}
df['LandContour_LabelEncoded'] = df['LandContour'].map(LandContour_order)
df.drop("LandContour",axis=1,inplace=True)
################################################################################################################################
df["Utilities"].value_counts() # Sadece bir gözlemdeki değeri NoSeWa diğerlerinin hepsi ALLPub olduğu için bir ayırt edicilik sağlamaıyor bundan dolayı drop ediyorum.
df.drop("Utilities", axis=1, inplace=True)
################################################################################################################################
df["LotConfig"].head()
mean_count_fonk(df,"LotConfig")
df["LotConfig"] = df["LotConfig"].apply(lambda x: "rare" if x in rare_fonk(df,"LotConfig",5) else x)
LotConfig_order = {"CulDSac":0, "Inside":1, "Corner":2, "rare":3 }
df["LotConfig_LabelEncoded"] = df["LotConfig"].map(LotConfig_order)
df.drop("LotConfig",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"LandSlope")
df["LandSlope"] = df["LandSlope"].apply(lambda x: "rare" if x in rare_fonk(df,"LandSlope",5) else x)
LotConfig_order = {"Gtl":0, "rare":1 }
df["LandSlope_LabelEncoded"] = df["LandSlope"].map(LotConfig_order)
df.drop("LandSlope",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Condition1")
df["Condition1"] = df["Condition1"].apply(lambda x: "rare" if x in rare_fonk(df,"Condition1",2) else x)
Condition1_oder = {"Artery":0, "Feedr":1, "Norm":2, "rare":3 }
df["Condition1_LabelEncoded"] = df["Condition1"].map(Condition1_oder)
df.drop("Condition1",axis=1,inplace=True)
################################################################################################################################
df.drop("Condition2",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"BldgType")
one_hot_encoded = pd.get_dummies(df['BldgType'], prefix='BldgType', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("BldgType",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"HouseStyle")
df["HouseStyle"] = df["HouseStyle"].apply(lambda x: "2.5_rare" if x in rare_fonk(df,"HouseStyle",1) else x)
df["HouseStyle"] = df["HouseStyle"].apply(lambda x: "Split_rare" if ((x in rare_fonk(df,"HouseStyle",5)) & (x not in ["2.5_rare"])) else x)
HouseStyle_oder = {"Split_rare":0, "1Story":1, "1.5Fin":2, "2Story":3, "2.5_rare":4 }
df["HouseStyle_LabelEncoded"] = df["HouseStyle"].map(HouseStyle_oder)
df.drop("HouseStyle",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"RoofStyle")
one_hot_encoded = pd.get_dummies(df['RoofStyle'], prefix='RoofStyle', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("RoofStyle",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"RoofMatl")
df["RoofMatl"] = df["RoofMatl"].apply(lambda x: "rare" if x in rare_fonk(df,"RoofMatl",1) else x)
RoofMatl_oder = {"CompShg":0, "rare":1}
df["RoofMatl_LabelEncoded"] = df["RoofMatl"].map(RoofMatl_oder)
df.drop("RoofMatl",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Exterior1st")
one_hot_encoded = pd.get_dummies(df['Exterior1st'], prefix='Exterior1st', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("Exterior1st",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Exterior2nd")
one_hot_encoded = pd.get_dummies(df['Exterior2nd'], prefix='Exterior2nd', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("Exterior2nd",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"MasVnrType")
one_hot_encoded = pd.get_dummies(df['MasVnrType'], prefix='MasVnrType', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("MasVnrType",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"ExterQual")
ExterQual_oder = {"Fa":0, "TA":1, "Gd":2, "Ex":3}
df["ExterQual_LabelEncoded"] = df["ExterQual"].map(ExterQual_oder)
df.drop("ExterQual",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"ExterCond")
ExterCond_oder = {"Po":0, "Fa":1, "TA":2, "Gd":3, "Ex":4}
df["ExterCond_LabelEncoded"] = df["ExterCond"].map(ExterCond_oder)
df.drop("ExterCond",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Foundation")
one_hot_encoded = pd.get_dummies(df['Foundation'], prefix='Foundation', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("Foundation",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"BsmtQual")
BsmtQual_oder = {"Fa":0, "TA":1, "Gd":2, "Ex":3}
df["BsmtQual_LabelEncoded"] = df["BsmtQual"].map(BsmtQual_oder)
df.drop("BsmtQual",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"BsmtCond")
BsmtCond_oder = {"Po":0, "Fa":1, "TA":2, "Gd":3}
df["BsmtCond_LabelEncoded"] = df["BsmtCond"].map(BsmtCond_oder)
df.drop("BsmtCond",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"BsmtExposure")
BsmtExposure_oder = {"No":0, "Mn":1, "Av":2, "Gd":3}
df["BsmtExposure_LabelEncoded"] = df["BsmtExposure"].map(BsmtExposure_oder)
df.drop("BsmtExposure",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"BsmtFinType1")
BsmtFinType1_oder = {"Unf":0, "LwQ":1, "Rec":2, "BLQ":3, "ALQ":4, "GLQ":5}
df["BsmtFinType1_LabelEncoded"] = df["BsmtFinType1"].map(BsmtFinType1_oder)
df.drop("BsmtFinType1",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"BsmtFinType2")
BsmtFinType2_oder = {"Unf":0, "LwQ":1, "Rec":2, "BLQ":3, "ALQ":4, "GLQ":5}
df["BsmtFinType2_LabelEncoded"] = df["BsmtFinType2"].map(BsmtFinType1_oder)
df.drop("BsmtFinType2",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Heating")
df["Heating"] = df["Heating"].apply(lambda x: "rare" if x in rare_fonk(df,"Heating",2) else x)
Heating_oder = {"GasA":0, "rare":1}
df["Heating_LabelEncoded"] = df["Heating"].map(Heating_oder)
df["Heating_LabelEncoded"].value_counts()
df.drop("Heating",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"HeatingQC")
HeatingQC_oder = {"Po":0, "Fa":1, "TA":2, "Gd":3, "Ex":4}
df["HeatingQC_LabelEncoded"] = df["HeatingQC"].map(HeatingQC_oder)
df.drop("HeatingQC",axis=1,inplace=True)
df["HeatingQual"] = df["Heating_LabelEncoded"] + df["HeatingQC_LabelEncoded"]
################################################################################################################################
mean_count_fonk(df,"CentralAir")
CentralAir_oder = {"N":0, "Y":1}
df["CentralAir_LabelEncoded"] = df["CentralAir"].map(CentralAir_oder)
df.drop("CentralAir",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Electrical")
Electrical_oder = {"Mix":0, "FuseP":1, "FuseF":2, "FuseA":3, "SBrkr":4}
df["Electrical_LabelEncoded"] = df["Electrical"].map(Electrical_oder)
df.drop("Electrical",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"KitchenQual")
KitchenQual_oder = {"Fa":0, "TA":1, "Gd":2, "Ex":3}
df["KitchenQual_LabelEncoded"] = df["KitchenQual"].map(KitchenQual_oder)
df.drop("KitchenQual",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Functional")
Functional_oder = {"Sev":0, "Maj2":1, "Maj1":2, "Mod":3, "Min2":4, "Min1":5, "Typ":6}
df["Functional_LabelEncoded"] = df["Functional"].map(Functional_oder)
df.drop("Functional",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"PavedDrive")
PavedDrive_oder = {"N":0, "P":1, "Y":2}
df["PavedDrive_LabelEncoded"] = df["PavedDrive"].map(PavedDrive_oder)
df.drop("PavedDrive",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"SaleType")
df["SaleType"] = df["SaleType"].apply(lambda x: "rare" if x in rare_fonk(df,"SaleType",2) else x)
one_hot_encoded = pd.get_dummies(df['SaleType'], prefix='SaleType', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("SaleType",axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"SaleCondition")
df["SaleCondition"] = df["SaleCondition"].apply(lambda x: "rare" if x in rare_fonk(df,"SaleCondition",2) else x)
one_hot_encoded = pd.get_dummies(df['SaleCondition'], prefix='SaleCondition', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("SaleCondition",axis=1,inplace=True)
################################################################################################################################
df.drop("BsmtFinSF2",axis=1,inplace=True)
################################################################################################################################
df.drop("LowQualFinSF",axis=1,inplace=True)
################################################################################################################################
df.drop(["EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal"],axis=1,inplace=True)
################################################################################################################################
mean_count_fonk(df,"Neighborhood")
one_hot_encoded = pd.get_dummies(df['Neighborhood'], prefix='Neighborhood', drop_first=True)
df = pd.concat([df, one_hot_encoded], axis=1)
df.drop("Neighborhood",axis=1,inplace=True)
################################################################################################################################
df.isnull().sum()[df.isnull().sum()!=0]
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['HighQualitySF'] = df['GrLivArea'] + df['BsmtFinSF1']
df["TotalQual"] = df["OverallQual"] + df["OverallCond"]
df["HeatingQual"] = df["Heating_LabelEncoded"] + df["HeatingQC_LabelEncoded"]
df["Basement_Total-Rate"] = df["BsmtFinType1_LabelEncoded"] + df["BsmtFinType2_LabelEncoded"]
df["Basement_attr1"] = df["BsmtExposure_LabelEncoded"] + df["BsmtQual_LabelEncoded"]
df["Basement_attr2"] = df["BsmtQual_LabelEncoded"] + df["BsmtCond_LabelEncoded"]
df["ExteriorState"] = df["ExterCond_LabelEncoded"] + df["ExterQual_LabelEncoded"]
df['LotInfo'] = df['LotShape_LabelEncoded'] + df['LotConfig_LabelEncoded']
df['PropertyInfo'] = df['MSSubClass'] + df['MSZoning_LabelEncoded']
df['TerrainInfo'] = df['LandContour_LabelEncoded'] + df['LandSlope_LabelEncoded']
df['ConfigInfo'] = df['LotConfig_LabelEncoded'] + df['LandSlope_LabelEncoded']
df['ShapeConfigInfo'] = df['LotShape_LabelEncoded'] + df['LotConfig_LabelEncoded'] + df['LandSlope_LabelEncoded']
df['CombinedInfo'] = df['MSSubClass'] + df['MSZoning_LabelEncoded'] + df['LandContour_LabelEncoded']
################################################################################################################################
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

X = df.drop("SalePrice", axis=1)
y = df[["SalePrice"]]
reg_model = LinearRegression().fit(X, y)
y_pred = reg_model.predict(X)
np.sqrt(mean_squared_error(y, y_pred))
reg_model.score(X, y)
y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
reg_model.score(X_train, y_train)

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
reg_model.score(X_test, y_test)
# Basic model sonucunda train ve test hataları gayet iyi R2 skoru .90
################################################################################################################################
# Random Forests, GBM, XGBoost, LightGBM,
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
warnings.simplefilter(action='ignore', category=Warning)
################################################################################################################################
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
robust_scaled_X = scaler.fit_transform(X)
robust_scaled_X_train = scaler.fit_transform(X_train)
robust_scaled_X_test = scaler.fit_transform(X_test)
################################################################################################################################
rf_model = RandomForestRegressor(random_state=17)
np.mean(np.sqrt(-cross_val_score(rf_model, X,y,cv=5,scoring="neg_mean_squared_error"))) # 21884.017903014224

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
np.mean(np.sqrt(-cross_val_score(rf_final, X,y,cv=5,scoring="neg_mean_squared_error")))# 21758.548785026815
################################################################################################################################
gbm_model = GradientBoostingRegressor(random_state=17)
np.mean(np.sqrt(-cross_val_score(gbm_model, X,y,cv=5,scoring="neg_mean_squared_error"))) # 20272.444310327508
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
np.mean(np.sqrt(-cross_val_score(gbm_final, X,y,cv=5,scoring="neg_mean_squared_error"))) # 19481.649634801226
################################################################################################################################
xgboost_model = XGBRegressor(random_state=17, use_label_encoder=False)
X.drop(["GarageYrBltQCut","GarageAreaQCut"], axis=1,inplace=True)# bu değişkenler kategorik olduğu için model kurulamdı zaten temel modleimizde de gördük bu değişkenler önemsiz değişkenler olduğundan drop ettik.
xgboost_model.fit(X,y)
np.mean(np.sqrt(-cross_val_score(xgboost_model, X,y,cv=5,scoring="neg_mean_squared_error"))) # 21383.56173785801
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)
np.mean(np.sqrt(-cross_val_score(gbm_final, X,y,cv=5,scoring="neg_mean_squared_error"))) # 19475.24509546236
################################################################################################################################
lgbm_model = LGBMRegressor(random_state=17)
X.drop("Id",axis=1,inplace=True)
np.mean(np.sqrt(-cross_val_score(lgbm_model, X,y,cv=5,scoring="neg_mean_squared_error"))) # 20346.899242427735 - 20447.81895173941 - 20600.24416175796
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
np.mean(np.sqrt(-cross_val_score(lgbm_final, X,y,cv=5,scoring="neg_mean_squared_error"))) # 19205.332278843875 - 19374.761056833784 - 19481.929215424178 - 19409.5445615286
################################################################################################################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
plot_importance(lgbm_final, X)
################################################################################################################################

#############################################
# ÖDEV 3 PROJE: KURAL TABANLI SINIFLANDIRMA
#############################################

##############################
# PROBLEM NEDİR ?
##############################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları
# (persona) oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni
# gelebilecek müşterilerin şirkete ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

##############################
# VERİ SETİ HAKKINDA BİLGİ
##############################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını
# ve bu ürünleri satın alan kullanıcıların bazı demografik bilgilerini barındırmaktadır.
# Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo tekilleştirilmemiştir.
# Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

##############################
# DEĞİŞKENLER
##############################
# PRICE – Müşterinin harcama tutarı
# SOURCE – Müşterinin bağlandığı cihaz türü
# SEX – Müşterinin cinsiyeti
# COUNTRY – Müşterinin ülkesi
# AGE – Müşterinin yaşı


# LEVEL BASED PERSONA TANIMLAMA, BASIT SEGMENTASYON ve KURAL TABANLI SINIFLANDIRMA

# Proje Amacı:
# - Persona kavramını düşünmek.
# - LEVEL BASED PERSONA TANIMLAMA: Kategori seviyelerine (Level) göre yeni müşteri tanımları yapabilmek.
# - BASIT SEGMENTASYON: qcut fonksiyonunu kullanarak basitçe yeni müşteri tanımlarını segmentlere ayırmak.
# - KURAL TABANLI SINIFLANDIRMA: Yeni bir müşteri geldiğinde bu müşteriyi segmentlere göre sınıflandırmak.

################# Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 15)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', True)


#############################################
# GÖREV 1: Veri Setinin Yüklenmesi
#############################################


def load_dataset():
    return pd.read_csv("datasets/persona.csv")


df = load_dataset()
df.head()


#############################################
# GÖREV 2: EDA Analizi
#############################################


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
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

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col)


def data_analysis(dataframe):
    print("##########################################")
    # Kaç unique SOURCE vardır?:
    print("Unique Values of Source:\n", dataframe[["SOURCE"]].nunique())

    print("##########################################")
    # SOURCE'un frekansları nedir:
    print("Frequency of Source:\n", dataframe["SOURCE"].value_counts())

    print("##########################################")
    # Kaç unique PRICE vardır?:
    print("Unique Values of Price:\n", dataframe[["PRICE"]].nunique())

    print("##########################################")
    #  Hangi PRICE'dan kaçar tane satış gerçekleşmiş:
    print("Number of product sales by sales price:\n", dataframe["PRICE"].value_counts())

    print("##########################################")
    # Hangi ülkeden kaçar tane satış olmuş:
    print("Number of product sales by country:\n", dataframe["COUNTRY"].value_counts(ascending=False))

    print("##########################################")
    #  Ülkelere göre satışlardan toplam ve ortalama ne kadar kazanılmış
    print("Total & average amount of sales by country:\n", dataframe.groupby("COUNTRY").agg({"PRICE": ["mean", "sum"]}))

    print("##########################################")
    # SOURCE'lara göre PRICE ortalamaları nedir:
    print("Average amount of sales by source:\n", dataframe.groupby("SOURCE").agg({"PRICE": "mean"}))

    print("##########################################")
    # Average amount of sales by source and country:
    print("Average amount of sales by source and country:\n", dataframe.pivot_table(values=['PRICE'],
                                                                                    index=['COUNTRY'],
                                                                                    columns=["SOURCE"],
                                                                                    aggfunc=["mean"]))


data_analysis(df)


#############################################
# GÖREV 3: Personaların Tanımlanması
#############################################


def define_persona(dataframe):
    agg_df = dataframe.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE",
                                                                                                          ascending=False).reset_index()
    bins = [agg_df["AGE"].min(), 18, 23, 30, 40, agg_df["AGE"].max()]
    labels = [str(agg_df["AGE"].min()) + '_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
    agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins, labels=labels)

    agg_df["CUSTOMERS_LEVEL_BASED"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + str(row[5]).upper() for row in agg_df.values]

    # Calculating average amount of personas:
    df_persona = agg_df.groupby("CUSTOMERS_LEVEL_BASED").agg({"PRICE": "mean"})
    df_persona = df_persona.reset_index()

    return df_persona


define_persona(df)


#############################################
# GÖREV 4: Segmentelerin Tanımlanması
#############################################


def create_segments(dataframe):
    df_persona = define_persona(dataframe)
    df_persona["SEGMENT"] = pd.qcut(df_persona["PRICE"], 4, labels=["D", "C", "B", "A"])

    return df_persona


create_segments(df)


#############################################
# GÖREV 5: Yeni Bir Müşteri Geldi Bakalım Ne Kadar Kazandıracak Bize?
#############################################


def AGE_CAT(age):
    if age <= 18:
        AGE_CAT = "0_18"
        return AGE_CAT
    elif (age > 18 and age <= 23):
        AGE_CAT = "19_23"
        return AGE_CAT
    elif (age > 23 and age <= 30):
        AGE_CAT = "24_30"
        return AGE_CAT
    elif (age > 30 and age <= 40):
        AGE_CAT = "31_40"
        return AGE_CAT
    elif (age > 40 and age <= agg_df["AGE"].max()):
        AGE_CAT = '41_' + str(agg_df["AGE"].max())
        return AGE_CAT


def new_users(dataframe):
    df_segment = create_segments(dataframe)

    COUNTRY = input("Enter a country name (USA/EUR/BRA/DEU/TUR/FRA):")
    SOURCE = input("Enter the operating system of phone (IOS/ANDROID):")
    SEX = input("Enter the gender (FEMALE/MALE):")
    AGE = int(input("Enter the age:"))
    AGE_SEG = AGE_CAT(AGE)
    new_user = COUNTRY.upper() + '_' + SOURCE.upper() + '_' + SEX.upper() + '_' + AGE_SEG

    print(new_user)
    print("Segment:" + str(df_segment[df_segment["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "SEGMENT"].values[0]))
    print("Price:" + str(df_segment[df_segment["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "PRICE"].values[0]))

    return new_user


new_users(df)

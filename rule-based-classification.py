#####################################
# Rule-Based Classification
#####################################


# Reading Data

import pandas as pd
import numpy as np

def load_persona(DataFrame):
    df = pd.read_csv(DataFrame)
    return df

df = load_persona('datasets/csv_path/persona.csv')


# 1. General information about the dataset

def check_df(dataframe, head = 5):

    """

    Task
    ---
    Task will give information about:
    - shape: number of elements per axis
    - types: array data type
    - head: first n observations
    - tail: last n observations
    - NA: null value
    - describe().T: Summary information of numeric variables
    - Quantiles: divides by the specified range values
    in the dataset.

    Parameters
    ----------
    dataframe: DataFrame
            Dataframe to get variable names
    head: int, optional
            first n observation

    Exapmle:
    ----------
            df = pd.read_csv("datasets/csv_path/persona.csv")
            check_df(df, 10)
    """

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


# 2. Number and frequencies of Unique SOURCE
df["SOURCE"].value_counts() # value_count -> The classes of a categorical variable give the frequencies of these classes.
df["SOURCE"].nunique()

# 3. Number of unique PRICE
df["PRICE"].unique()

# 4. How many price at which price?
df[["PRICE"]].value_counts()

# 5. How many sales from which country?
df[["PRICE","COUNTRY"]].groupby("COUNTRY").agg({"count"})

# 6. How much was earned in total from price by country?
df.loc[:,["COUNTRY","PRICE"]].groupby("COUNTRY").agg({"sum"})

# 7. What are the sales numbers by source types?
df.loc[:,["SOURCE","PRICE"]].groupby("SOURCE").agg({"count"})

# 8. What are the price averages by country?
df.loc[:,["COUNTRY","PRICE"]].groupby("COUNTRY").agg({"mean"})

# 9. What are the price averages based on sources?
df.loc[:,["PRICE","SOURCE"]].groupby("SOURCE").agg({"PRICE":["mean"]}).astype(int)

# 10. What are the price averages in the country-source breakdown?
df.loc[:,["PRICE","SOURCE","COUNTRY"]].groupby(["COUNTRY","SOURCE"]).agg({"PRICE":["mean"]}).astype(int).head()

#####################################
# TASK 2
#####################################

# What are the total earnings broken down by country, source, gender, age?
agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).sum().head()

#####################################
# TASK 3
#####################################

# Sorting output by price
agg_df = agg_df.sort_values("PRICE", ascending=False).head()

#####################################
# TASK 4
#####################################

# Converting index names to variable names
# All variables except price in the output of the third question are index names.
agg_df = agg_df.reset_index()

#####################################
# TASK 5
#####################################

# Converting age variable to categorical variable and adding 'agg_df'

agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins = [0,18,23,30,40,69],labels = ['0_18','19_23','24_30','31_40','41_70'])
agg_df

#####################################
# TASK 6
#####################################

# Identify new level-based customers (personas).

agg_df["customer_level_based"] = [agg_df["COUNTRY"][i].upper() + "_"+
                                  agg_df["SOURCE"][i].upper() + "_"+
                                  agg_df["SEX"][i].upper() + "_" +
                                  agg_df["age_cat"][i].upper() + "_"
                                  for i in range(0,len(agg_df))]
agg_df = agg_df.groupby("customer_level_based").agg({"PRICE":"mean"})

#####################################
# TASK 7
#####################################

# Segment new customers (personas).
# Divide new customers into 4 segments based on price.

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels = ["D","C","B","A"])
agg_df.groupby("SEGMENT").agg({"PRICE":["mean","min","max","sum"]})
agg_df[agg_df["SEGMENT"] == "C"].describe().T

#####################################
# TASK 8
#####################################

# Which segment does a Turkish woman using android at the age of 33 belong to and how much income is expected to earn on average?

new_user1 = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == new_user1].groupby("new_user1").agg({"PRICE":["mean"]})

new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == new_user2].groupby("new_user1").agg({"PRICE":["mean"]})
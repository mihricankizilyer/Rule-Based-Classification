#####################################
# Rule-Based Classification
#####################################

######################
#  TASK 1
######################

# Reading Data
import pandas as pd
import numpy as np
df = pd.read_csv('datasets/csv_path/persona.csv')

# 1. General information about the dataset
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

check_df(df, 5)

# 2. Number and frequencies of Unique SOURCE
df["SOURCE"].nunique() # value_count -> The classes of a categorical variable give the frequencies of these classes.
df["SOURCE"].value_counts()

# 3. Number of unique PRICE
df["PRICE"].nunique()

# 4. How many price at which price?
df["PRICE"].value_counts()

# 5. How many sales from which country?
df["COUNTRY"].value_counts()

# 6. How much was earned in total from price by country?
df.groupby("COUNTRY").agg({"PRICE":"sum"})

# 7. What are the sales numbers by source types?
df["SOURCE"].value_counts()

# 8. What are the price averages by country?
df.groupby(by = ['COUNTRY']).agg({"PRICE":"mean"})

# 9. What are the price averages based on sources?
df.groupby(by = ["SOURCE"]).agg({"PRICE":"mean"})

# 10. What are the price averages in the country-source breakdown?
df.groupby(by = ["COUNTRY","SOURCE"]).agg({"PRICE":"mean"})

######################
#  TASK 2
######################

# What are the total earnings broken down by country, source, gender, age?
agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "sum"})

######################
#  TASK 3
######################

# Sorting output by price
agg_df = agg_df.sort_values(["PRICE"], ascending=False) # ascending = False -> declining value

######################
#  TASK 4
######################

# Converting index names to variable names
# All variables except price in the output of the third question are index names.
agg_df = agg_df.reset_index()

######################
#  TASK 5
######################

# Converting age variable to categorical variable and adding 'agg_df'

# Specifying where to split the AGE variable
my_bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Expressing what the nomenclature will be in response to the divided values
my_labels = ['0_18','19_23','24_30','31_40','41_' + str(agg_df["AGE"].max())]

agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins = my_bins, labels = my_labels)

agg_df.head()

######################
#  TASK 6
######################

# Identify new level-based customers (personas).
agg_df["customer_level_based"] = [  row[0].upper() + "_" +
                                    row[1].upper() + "_" +
                                    row[2].upper() + "_" +
                                    row[0].upper()
                                    for row in agg_df.values] 

agg = agg_df[["customer_level_based","PRICE"]]

agg_df = agg_df.groupby("customer_level_based").agg({"PRICE":"mean"})

agg_df = agg_df.reset_index()

######################
#  TASK 7
######################

# Segment new customers (personas).
# Divide new customers into 4 segments based on price.

agg_df["SEGMENT"] = pd.cut(agg_df["PRICE"], 4, labels=["D","C","B","A"])
agg_df.groupby("SEGMENT").agg({"PRICE":["mean","max","sum"]})
agg_df[agg_df["SEGMENT"] == "C"]


######################
#  TASK 8
######################

# Estimating how much revenue the classification of new customers can bring

agg_df.reset_index(inplace = True)


# Which segment does a Turkish woman using android at the age of 33 belong to and how much income is expected to earn on average?
new_user1 = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customer_level_based"]  == new_user1]


# What segment does a 33-year-old French woman using ios belong to and what is the average income expected to earn?
new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == new_user2]



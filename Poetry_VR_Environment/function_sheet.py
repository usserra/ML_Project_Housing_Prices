import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer

def nan_calculator(column):
    summation = column.isna().sum()
    number = column.notna().sum()
    nan_num = print(["Number of NaN values:", summation, "Number of non-NaN values:", number]) 
    for number in nan_num:
        return summation > 730

def category_calculator(column):
    if pd.api.types.is_object_dtype(column):
        try:
            categories = column.unique()
            print('Categories in the column are:', categories)
        except TypeError:
            print("No categorical data present")
    else:
        print("No categorical data present")


def category_calculator_old(column):
    try:
        categories = column.unique('object')
        category = print('Categories in the column are :', categories)
        return category
    except TypeError:
        print("No categorical data present")


def outlier_remover(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    outlier1 = q1 - (iqr * 1.5)
    outlier2 = q3 + (iqr * 1.5)
    mask = (column < outlier1) | (column > outlier2)
    column = column[~mask]
    return column

def delete_columns(data): 
    deleted_columns_data = data.dropna(axis = 1, inplace = True)
    return deleted_columns_data

def info_columns_deleted(data):
    original = data.columns
    new = data.dropna(axis= 1).columns
    removed_columns = list(set(original) - set(new))
    print("Number of columns removed: ", len(removed_columns))
    return removed_columns

def ordinal_encoder(data):  
    label_data  = data.copy()
    object_columns = [col for col in label_data.columns if label_data[col].dtype == "object"]
    label_data[object_columns] = OrdinalEncoder().fit_transform(data[object_columns])
    return label_data

def mi_scores(x, y): #one can make a function so that one doesn't need to copy the code over and over again if needed to be sued
    mi = mutual_info_regression(x, y)
    mi = pd.Series(mi, index = x.columns) #to create a table looking output to see mi scores for each column by name
    mi = mi.sort_values(ascending = False) #highest mi scored values will be on top
    return mi

def new_features(data):
    data['AgeHouse'] = data['YrSold'] - data['YearBuilt']
    data['TotalSF'] = data['1stFlrSF'] +  data['2ndFlrSF'] + data['TotalBsmtSF']
    data['HasPool'] = [1 if area > 0 else 0 for area in data['PoolArea']]
    new_data = data[['AgeHouse', 'TotalSF', 'HasPool']]
    return new_data

def imputer(data): #method 3
    import pandas as pd
    knn_imputer = KNNImputer(n_neighbors = 5) #it will look at 5 values next to it, distance between them and then the average 
    imputed_data = pd.DataFrame(knn_imputer.fit_transform(data))
    imputed_data.columns = data.columns 
    return imputed_data
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
from xgboost import XGBClassifier
# import random forest regressor
from sklearn.ensemble import RandomForestRegressor
#import linear regression
from sklearn.linear_model import LinearRegression
# import tqdm
from tqdm import tqdm
import tqdm
#import r2_score
from sklearn.metrics import r2_score
#import confusion matrix
from sklearn.metrics import confusion_matrix
# import roc auc score
from sklearn.metrics import roc_auc_score
import re
import geemap, ee
from scipy.spatial import KDTree
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import glob
import sys
from collections import defaultdict
sys.path.append('C:/Program Files/Stata18/utilities')
X_drop_set = ['lat', 'lon', 'month','year','area_id','elevation','market_access','nitrogen_5-15cm_mean',
'phh2o_5-15cm_mean',
'cec_5-15cm_mean',
'cfvo_5-15cm_mean',
'soc_5-15cm_mean','aez_groupid_4000',
'aez_groupid_7000',
'aez_groupid_9000',
'aez_groupid_10000',
'aez_groupid_12000',
'aez_groupid_17000',
'aez_groupid_19000',
'aez_groupid_25000',
'aez_groupid_30000',
'aez_groupid_31000',
'aez_groupid_32000',
'aez_groupid_33000',
'aez_groupid_34000',
'aez_groupid_36000',
'aez_groupid_40000',
'aez_groupid_43000',
'slope',
'estimated_population',
'cropland',
'rangeland','area',
'es_urban_pop',
'urban_area',
'distance_to_river',
'ruggedness_index']

X_stable_set = ['elevation','market_access', 'nitrogen_5-15cm_mean',
'phh2o_5-15cm_mean',
'cec_5-15cm_mean',
'cfvo_5-15cm_mean',
'soc_5-15cm_mean','aez_groupid_4000',
'aez_groupid_7000',
'aez_groupid_9000',
'aez_groupid_10000',
'aez_groupid_12000',
'aez_groupid_17000',
'aez_groupid_19000',
'aez_groupid_25000',
'aez_groupid_30000',
'aez_groupid_31000',
'aez_groupid_32000',
'aez_groupid_33000',
'aez_groupid_34000',
'aez_groupid_36000',
'aez_groupid_40000',
'aez_groupid_43000',
'slope',
'estimated_population',
'cropland',
'rangeland','area',
'es_urban_pop',
'urban_area',
'distance_to_river',
'ruggedness_index']

def calculate_finetune_metric(y_test, y_pred, cm, y_pred_test):
    accuracy_score_new = accuracy_score(y_test, y_pred)
    correct_3_more = np.sum(cm[2:, 2:])
    total_3_more = np.sum(cm[2:, :])
    sensitivity = correct_3_more / total_3_more if total_3_more != 0 else 0
    precise_3_more = np.sum(cm[2:, 2:])
    total_prec_3_more = np.sum(cm[:, 2:])
    precision = precise_3_more / total_prec_3_more if total_prec_3_more != 0 else 0
    overall_r2 = r2_score(y_pred_test['phase3_test'], y_pred_test['phase3_pred'])
    finetune_metric = accuracy_score_new + sensitivity + precision + overall_r2
    return floor(finetune_metric)
def all_metrics(y_test, y_pred, cm, y_pred_test):
    accuracy_score_new = accuracy_score(y_test, y_pred)
    correct_3_more = np.sum(cm[2:, 2:])
    total_3_more = np.sum(cm[2:, :])
    sensitivity = correct_3_more / total_3_more if total_3_more != 0 else 0
    correct_1_2 = np.sum(cm[2:, 2:])
    total_1_2 = np.sum(cm[:, 2:])
    precision = correct_1_2 / total_1_2 if total_1_2 != 0 else 0
    overall_r2 = r2_score(y_pred_test['phase3_test'], y_pred_test['phase3_pred'])
    return accuracy_score_new, sensitivity, precision, overall_r2
def calculate_fine_metric(accuracy_score_list, sensitivity_list, precision_list, overall_r2_list):
    accuracy_score_mean = round(np.mean(accuracy_score_list),3)
    sensitivity_mean = round(np.mean(sensitivity_list),3)
    precision_mean = round(np.mean(precision_list),3)
    overall_r2_mean = round(np.mean(overall_r2_list),3)
    accuracy_score_std = round(np.std(accuracy_score_list),3)
    sensitivity_std = round(np.std(sensitivity_list),3)
    overall_r2_std = round(np.std(overall_r2_list),3)
    precision_std = round(np.std(precision_list),3)
    return accuracy_score_mean, accuracy_score_std, sensitivity_mean, sensitivity_std, precision_mean, precision_std, overall_r2_mean, overall_r2_std
def convert_prob_to_phase(y_pred_test, th=0.2):
    y_pred_test['y_pred'] = y_pred_test['y_pred'].round(2)
    phase_list = [y_pred_test[y_pred_test['phase']==i].drop(['phase'], axis=1).rename(columns={'y_pred': 'phase{}_pred'.format(i), 'y_test': 'phase{}_test'.format(i)}) for i in range(1, 6)]
    phase_list = [phase_list[i].reset_index(drop=True) for i in range(5)]
    y_pred_test = pd.concat(phase_list, axis=1)
    # phase1_test = 1-phase2_test-phase3_test-phase4_test-phase5_test, phase1_pred = 1-phase2_pred-phase3_pred-phase4_pred-phase5_pred
    y_pred_test = y_pred_test.drop(['phase1_test', 'phase1_pred'], axis=1)
    #y_pred_test['phase3_pred'] = 1 - y_pred_test['phase1_pred'] - y_pred_test['phase2_pred'] - y_pred_test['phase4_pred'] - y_pred_test['phase5_pred']
    # fillna with 0
    y_pred_test = y_pred_test.fillna(0)
    # find row while phase1_pred+phase2_pred+phase3_pred+phase4_pred+phase5_pred<0, drop these rows
    y_pred_test = y_pred_test.drop(y_pred_test[( y_pred_test['phase2_pred'] + y_pred_test['phase3_pred'] + y_pred_test['phase4_pred'] + y_pred_test['phase5_pred'] <= 0)].index)
    # find row while phase1_test+phase2_test+phase3_test+phase4_test+phase5_test<0, drop these rows
    y_pred_test = y_pred_test.drop(y_pred_test[(y_pred_test['phase2_test'] + y_pred_test['phase3_test'] + y_pred_test['phase4_test'] + y_pred_test['phase5_test'] <= 0)].index)
    # set true overall_label = 5 if phase5_test >= 0.2, = 4 if phase4_test >= 0.2, = 3 if phase3_test >= 0.2, = 2 if phase2_test >= 0.2, = 1 if phase1_test >= 0.2.
    for row, idx in zip(y_pred_test.itertuples(), y_pred_test.index):
        if row.phase5_test >= th:
            y_pred_test.loc[idx, 'overall_phase'] = 5
        elif row.phase4_test>= th:
            y_pred_test.loc[idx, 'overall_phase'] = 4
        elif row.phase3_test>= th:
            y_pred_test.loc[idx, 'overall_phase'] = 3
        elif row.phase2_test>= th:
            y_pred_test.loc[idx, 'overall_phase'] = 2
        else:
            y_pred_test.loc[idx, 'overall_phase'] = 1

        if row.phase5_pred >= th:
            y_pred_test.loc[idx, 'overall_phase_pred'] = 5
        elif row.phase4_pred>= th:
            y_pred_test.loc[idx, 'overall_phase_pred'] = 4
        elif row.phase3_pred>= th:
            y_pred_test.loc[idx, 'overall_phase_pred'] = 3
        elif row.phase2_pred>= th:
            y_pred_test.loc[idx, 'overall_phase_pred'] = 2
        else:
            y_pred_test.loc[idx, 'overall_phase_pred'] = 1
    return y_pred_test
def new_function(files):
    p=re.compile(r"\b[-a-z]{2,40}\s?\r\n")
    def list_all_files(dir):
        import os.path
        _files=[]
        list=os.listdir(dir)
        for i in range(0,len(list)):
            path=os.path.join(dir,list[i])
            if os.path.isdir(path):
                _files.extend(list_all_files(path))
            if os.path.isfile(path):
                _files.append(path)
        return _files
    count=0
    with open("listofwords.txt","w") as f:
        for file in files:
            f.write("\n"+file+"\n")
            with open(file,"rb") as fr:
                str=fr.read().decode("gbk","ignore")
                words=re.findall(p,str)
                word_remove_duplication=[]
                for word in words:
                    if word not in word_remove_duplication:
                        word_remove_duplication.append(word)
                for s in word_remove_duplication:
                    f.write(s)
                    count+=1
    import os
    list1=os.listdir()
    for i in list1:
        x=i[:-1]+"rar"
        os.rename(i,x)
    import zipfile
    list1=os.listdir()
    # specify the path to the zip file
    for i in list1:
        zip_file_path = i
        # create a ZipFile object
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # extract all the contents of the zip file to the current directory
            zip_ref.extractall()
def drop_cols(df,cols_to_drop,patterns_to_drop):
    for col in cols_to_drop:
        if col in df.columns.values.tolist():
            df.drop(col, axis = 1, inplace = True)
        else:
            pass
    for pattern in patterns_to_drop:
        for col in df.columns.values.tolist():
            if re.search(pattern,col):
                df.drop(col, axis = 1, inplace = True)
            else:
                pass
    return df
# reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernal
def forecasting_pipeline(train_df,test_df,i):
    train_df['month'] = pd.to_datetime(train_df['date']).dt.month
    test_df['month'] = pd.to_datetime(test_df['date']).dt.month
    train_df['year'] = pd.to_datetime(train_df['date']).dt.year
    test_df['year'] = pd.to_datetime(test_df['date']).dt.year
    train_df = train_df.drop(['date'], axis=1)
    test_df = test_df.drop(['date'], axis=1)
    train_df_new = train_df.drop(['phase{}_worse'.format(j) for j in range(2, 6) if j != i], axis=1)
    test_df_new = test_df.drop(['phase{}_worse'.format(j) for j in range(2, 6) if j != i], axis=1)
    train_df_new = train_df_new.dropna(subset=['phase{}_worse'.format(i)])
    test_df_new = test_df_new.dropna(subset=['phase{}_worse'.format(i)])
    test_index = test_df_new.index
    X_train = train_df_new.drop('phase{}_worse'.format(i), axis=1)
    y_train = train_df_new['phase{}_worse'.format(i)]
    X_pred = test_df_new.drop('phase{}_worse'.format(i), axis=1)
    y_pred = test_df_new['phase{}_worse'.format(i)]
    X__pred = X_pred.copy()
    X__train = X_train.copy()
    y__train = y_train.copy()
    y__pred = y_pred.copy()
    X_pred_loc = X_pred[['lat', 'lon', 'month','year','area_id']]
    X_pred_loc = pd.concat([X_pred_loc, pd.get_dummies(X_pred_loc['month'], prefix='month'),pd.get_dummies(X_pred_loc['area_id'],prefix="area")], axis=1)
    for k in range(1, 13):
        if 'month_{}'.format(k) not in X_pred_loc.columns:
            X_pred_loc['month_{}'.format(k)] = False
    for k in range(1, 1876):
        if 'area_{}'.format(k) not in X_pred_loc.columns:
            X_pred_loc['area_{}'.format(k)] = False
    X_pred_loc = X_pred_loc.drop(['month','area_id'], axis=1)
    X_stable = X_pred[X_stable_set]
    X_pred = X_pred.drop(X_drop_set, axis=1)
    X_pred = pd.DataFrame(np.nan, index=X_pred.index, columns=X_pred.columns)
    X_pred = pd.concat([X_pred_loc,X_stable,X_pred], axis=1)
    X_pred = X_pred.loc[:,~X_pred.columns.duplicated()]
    X_pred = X__pred.drop([ 'month','year'], axis=1)
    #X_pred = X_pred.fillna(0)
    y_test = test_df_new['phase{}_worse'.format(i)]
    y_train = y__train.copy()
    y_pred = y__pred.copy()
    #for i in range(len(date_split)-1):
    #df_train_new = train_df[train_df['date'] <= date_split[i]]
    #df_train_new = df_train_new.drop(['date','area_id'], axis=1)
    #train_list.append(df_train_new)
    #df_val_new = train_df[(train_df['date'] > date_split[i]) & (train_df['date'] <= date_split[i+1])]
    #df_val_new = df_val_new.drop(['date','area_id'], axis=1)
    #val_list.append(df_val_new)
    #for train_index, val_index in zip(train_list, val_list):
    fews_ipc_ha_test = X_pred['fews_ipc_ha']
    X_train_loc = X_train[['lat', 'lon', 'month','year','area_id']]
    X_train_loc = pd.concat([X_train_loc, pd.get_dummies(X_train_loc['month'], prefix='month'),pd.get_dummies(X_train_loc['area_id'],prefix="area")], axis=1)
    for k in range(1, 13):
        if 'month_{}'.format(k) not in X_train_loc.columns:
            X_train_loc['month_{}'.format(k)] = False
    for k in range(1, 1876):
        if 'area_{}'.format(k) not in X_train_loc.columns:
            X_train_loc['area_{}'.format(k)] = False
    X_train_loc = X_train_loc.drop(['month','area_id'], axis=1)
    X_train = X_train.drop(['lat', 'lon', 'month','year','area_id'], axis=1)
    X_train = pd.concat([X_train_loc, X_train], axis=1)
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_train = X__train.drop([ 'month','year'], axis=1)
    #X_train = X_train.drop(['lat', 'lon', 'month','year'], axis=1)
    #X_train = X_train[X_pred.columns]
    return X_train, X_pred, y_train, y_pred, y_test, test_index,fews_ipc_ha_test

def extract_from_location(image_collection,location,band,start_time,end_time):
    result = pd.DataFrame(columns=['date', 'mean','stdDev','lat','lon'])
    def poi_mean(img):
        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30, bestEffort=True).get(band)
        return img.set('date', img.date().format()).set('mean', mean)
    def poi_sd(img):
        stdDev = img.reduceRegion(reducer=ee.Reducer.stdDev(), geometry=poi, scale=30, bestEffort=True).get(band)
        return img.set('date', img.date().format()).set('stdDev', stdDev)
    for i in tqdm.tqdm(range(len(location))):
        # identify a 500 meter buffer around our Point Of Interest (POI)
        poi = ee.Geometry.Point(location['lat'][i], location['lon'][i]).buffer(100)
        viirs = ee.ImageCollection(image_collection).filterDate(start_time,end_time)
        poi_reduced_imgs_mean = viirs.map(poi_mean)
        poi_reduced_imgs_sd = viirs.map(poi_sd)
        nested_list_mean = poi_reduced_imgs_mean.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
        nested_list_sd = poi_reduced_imgs_sd.reduceColumns(ee.Reducer.toList(2), ['date','stdDev']).values().get(0)
        # dont forget we need to call the callback method "getInfo" to retrieve the data
        df_1 = pd.DataFrame(nested_list_mean.getInfo(), columns=['date','mean'])
        df_2 = pd.DataFrame(nested_list_sd.getInfo(), columns=['date','stdDev'])
        # merge on date
        df = pd.merge(df_1, df_2, on='date')
        df['lat'] = location['lat'][i]
        df['lon'] = location['lon'][i]
        df['date'] = pd.to_datetime(df['date'])
        result = pd.concat([result, df], axis=0)
    return result
def extract_from_location(image_collection,location,band,start_time,end_time):
    result = pd.DataFrame(columns=['date', 'mean','stdDev','lat','lon'])
    def poi_mean(img):
        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30, bestEffort=True).get(band)
        return img.set('date', img.date().format()).set('mean', mean)
    def poi_sd(img):
        stdDev = img.reduceRegion(reducer=ee.Reducer.stdDev(), geometry=poi, scale=30, bestEffort=True).get(band)
        return img.set('date', img.date().format()).set('stdDev', stdDev)
    for i in tqdm.tqdm(range(len(location))):
        # identify a 500 meter buffer around our Point Of Interest (POI)
        poi = ee.Geometry.Point(location['lat'][i], location['lon'][i]).buffer(100)
        viirs = ee.ImageCollection(image_collection).filterDate(start_time,end_time)
        poi_reduced_imgs_mean = viirs.map(poi_mean)
        poi_reduced_imgs_sd = viirs.map(poi_sd)
        nested_list_mean = poi_reduced_imgs_mean.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
        nested_list_sd = poi_reduced_imgs_sd.reduceColumns(ee.Reducer.toList(2), ['date','stdDev']).values().get(0)
        # dont forget we need to call the callback method "getInfo" to retrieve the data
        df_1 = pd.DataFrame(nested_list_mean.getInfo(), columns=['date','mean'])
        df_2 = pd.DataFrame(nested_list_sd.getInfo(), columns=['date','stdDev'])
        # merge on date
        df = pd.merge(df_1, df_2, on='date')
        df['lat'] = location['lat'][i]
        df['lon'] = location['lon'][i]
        df['date'] = pd.to_datetime(df['date'])
        result = pd.concat([result, df], axis=0)
    return result
def match_nearest(df_left,df_match,dat):
    new_df = pd.DataFrame()
    df_match = df_match[df_match['date'].dt.date == dat]
    df_match = df_match[['latitude', 'longitude']]
    #select df_IPC data for a specific date
    df_left = df_left[df_left['date'].dt.date == dat]
    df_left = df_left[['date','latitude','longitude']]
    # for each longitude and longitude in df_chirps_gdd_sd, match for the nearest longitude and latitude in aggregated_gdf
    location = df_match[['longitude', 'latitude']].values
    tree = KDTree(location)
    df_left[['longitude', 'latitude']] = df_left[['lon', 'lat']].astype(float)
    # find the nearest longitude and latitude in aggregated_gdf for each longitude and latitude in df_chirps_gdd_sd
    df_left['nearest_neighbor'] = df_left.apply(lambda x: tree.query([x['longitude'], x['latitude']])[1], axis=1)
    df_left['longitude_match'] = df_left['nearest_neighbor'].apply(lambda x: location[x][0])
    df_left['latitude_match'] = df_left['nearest_neighbor'].apply(lambda x: location[x][1])
    new_df = pd.concat([new_df, df_left], axis=0)
    return new_df
def floor(x):
    return abs(x)
def plot_missing_value(df):
# Assuming 'baseline' is your DataFrame
    plt.figure(figsize=(16,9))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    # create legend
    missing_data_patch = mpatches.Patch(color='yellow', label='missing data')
    not_missing_data_patch = mpatches.Patch(color='purple', label='not missing data')
    plt.legend(handles=[missing_data_patch, not_missing_data_patch], loc='lower right')
    return plt.show()
def read_directory(directory,extension):
    # read all dta files,using glob
    files = glob.glob(directory + "/*"+extension)
    # use pandas to read from glob, using original name as dataframe name,if there are spaces in the name, use underscore instead
    for file in files:
        name = file.split('\\')[-1].split('.')[0].replace(' ','_')
        vars()[name] = pd.read_stata(file)        
# replace columns,
def replace_columns(data_frame,character):
    for column in data_frame.columns:
        if data_frame[column].dtype == object:  # Checks if the column is of object type (typically string)
            data_frame[column] = data_frame[column].str.replace(character, "'")
    return data_frame
def find_index_of_C1(columns,string):
    list = []
    for i in range(len(columns)):
        if columns[i].startswith(string):
            list.append(i)
    return list
import subprocess
import os
def upload_to_bucket(bucket_path,collection,gsutil_path):
    earthengine_path = 'earthengine'
    # List all the TIFF files in the bucket using gsutil.
    # Note: If you encounter issues, you might need to add shell=True
    tiff_files = subprocess.check_output([gsutil_path, 'ls', f'{bucket_path}/*.tif'], shell=True).decode('utf-8').split('\n')
    # The GEE collection to which you want to upload your TIFF files.
    # Iterate over the list of TIFF files and upload each to GEE.
    for tiff_path in tiff_files:
        if tiff_path:  # This check is to skip any empty strings in the list.
            filename = os.path.basename(tiff_path)
            # extract  GPP_mean_201601 from GOSIF_GPP_2016.M01_Mean.tif as asset_id
            new_filename = filename.split('_')[1] + '_' + filename.split('_')[2] + '_' + filename.split('_')[3].split('.')[0]
            #replace . with _
            new_filename = new_filename.replace('.', '_')
            asset_id = f'{collection}/{new_filename}'
            # Earth Engine upload command.
            # Include additional properties as needed.
            # Example: --time_start and --time_end are optional
            upload_command = [
                earthengine_path,
                'upload',
                'image',
                '--asset_id={}'.format(asset_id),
                tiff_path
                # '--time_start="START_TIME"',
                # '--time_end="END_TIME"'
            ]
            # Run the upload command.
            # Note: If you encounter issues, you might need to add shell=True
            print(f'Uploading {filename} to {asset_id}')
            subprocess.run(upload_command, shell=True)

    print('Batch upload to GEE initiated.')

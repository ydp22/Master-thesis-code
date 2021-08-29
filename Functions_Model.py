# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:17:04 2020

@author: Yash's PC
"""

import pandas as pd
import numpy as np

""" This file is for collection of all the functions used for data processing
which helps in creating a final datasets for the model to use. For different types of data 
the function are changed a little bit to adapt the other changes in the raw data."""


# =============================================================================
# This section is mostly used to process the demand data. First part filters the data 
#for number of missing values and the other part fills out these missing values by weekly
#averages.
# =============================================================================

# Creating Datetime_index:
def DateTime_index(df,freq_in_quotation):
    df[df.columns[0]] = df[df.columns[0]].str.split('-',n=2,expand=True)
    df.index=pd.to_datetime(df[df.columns[0]]).dt.to_period(freq_in_quotation)
    df=df.drop([df.columns[0],df.columns[1]],axis=1)
    return df

#Find out the number of consecutive gaps in data:

def gap_occurances(df, per_year=False, write=True):
    def consecutive_nans(ds):
        return (ds.isnull().astype(int)
                  .groupby(ds.notnull().astype(int).cumsum()).sum())
    if not per_year:
        return df.apply(lambda d: consecutive_nans(d).value_counts()).fillna(0)
    tables=[]
    for year, dff in df.reset_index().groupby(df.reset_index().DateTime.dt.year):

        cons_nans = dff.apply(consecutive_nans)
        table = pd.DataFrame(index=range(len(df)))

        for column in dff:
            table[column] = cons_nans[column].value_counts()

        table=table.fillna(0).drop(0).drop('DateTime', axis=1)
        if write:
            table.to_csv('RAW DATA_'+str(year)+'.csv')
        tables.append(table)
    return tables

# Filter the data to see that whether the number of gap is not greater than a certain number

def filter_data(df, alpha, threshold, per_year=False):
    if per_year:
        res = sum(gap_occurances(df, per_year=per_year))
    else:
        res = gap_occurances(df)
    res = res.apply(lambda x : x * x.index**alpha).sum().sort_values()
    return df.drop(res[res>threshold].index, axis=1)

#Get weekly average 

def transform_to_average_week(df):
    groups = [df.index.dayofweek, df.index.hour]
    return df.groupby(groups).transform('mean')

#Create a factor by dividing original dataFrame with interpolated data.
#Multipling Factor with interpolated data to scale the data

def scale_data(df, PS,freq_in_quotation):
    stats = PS
    factors = df.resample('M').sum().pipe(lambda df: stats.resample('M').sum().div(df)).resample(freq_in_quotation).ffill().reindex(df.index).ffill()
    return df.multiply(factors),factors

#Change the index back to original

def Standard_index(df):
    df.index= Std_index
    return df 

#Total Demand(Demand-Import+Export)
def Total_Demand(df,Import_df,Export_df):
    df['Total_Demand']=df[df.columns[1]]-Import_df.sum(axis=1)+Export_df.sum(axis=1)
    return df
    

# =============================================================================
# Functions for powerplant lists
# =============================================================================

def Powerplants_list(Country_code):
    z=[]
    for i in Powerplants.index:
        if Powerplants['Country'].loc[i]==Country_code:
            pass
        else:
            z.append(i)
    return Powerplants.drop(z,axis=0)

#To remove Decommissioned Powerplants

def Decommissioned_Hydro_Drop(df,year):
    x=[]
    for i in df.index:
        if df['YearDecommissioning'].loc[i] <= year:
            x.append(i)
        else:
            pass
    df=df.drop(x,axis=0)
    y=[]
    for i in df.index:
        if df['Fueltype'].loc[i]=='Hydro':
            y.append(i)
        else:
            pass
    df=df.drop(y,axis=0)    
    return df

# =============================================================================
# Fuel data Fucntions
# =============================================================================

#Adding efficiency to different powerplants
Fuel_Data = pd.read_csv('Input/Fuel Costs.csv',index_col=0)

Efficiency_Range=pd.DataFrame(index=range(10000),columns=Fuel_Data.index)

def Efficiency_Cost(df):
    df['Min_Operation_Time'] = np.nan
    df['Startup_Costs[euro/MW]'] = np.nan
    df['Min_Down_Time'] = np.nan
    df['pu_min'] = np.nan
    for i in Efficiency_Range.columns:
        np.random.seed([2])
        eff=[]
        for j in range(10000):
            eff.append(np.random.uniform(Fuel_Data['Efficiency_Min'].loc[i],Fuel_Data['Efficiency_Max'].loc[i]))
        Efficiency_Range[i]=eff
        
    def Efficiency(df):
        for i in df.index:
            for j in Efficiency_Range.columns:
                if df['Fueltype'].loc[i] == j:
                    df['Efficiency'].loc[i] = Efficiency_Range[j].loc[i]
                    df['pu_min'].loc[i] = Fuel_Data['pu_min'].loc[j]
                    df['Min_Operation_Time'].loc[i] = Fuel_Data['Min_Operation_Time'].loc[j]
                    df['Startup_Costs[euro/MW]'].loc[i] = Fuel_Data['Startup_Costs[euro/MW]'].loc[j] * df['pu_min'].loc[i] * df['Capacity'].loc[i]
                    df['Min_Down_Time'].loc[i] = Fuel_Data['Min_Down_Time'].loc[j]
                else:
                    pass
        return df
    df=Efficiency(df)
    
#Adding marginal cost for different powerplants

    x=np.random.seed([1])
    y=np.random.random_sample([10000,1])
    WhiteNoise=(y)/10

    def Marginal_Cost(df):
        def Cost(row):
            Power_Plant_Efficiency,Fuel_type = row.loc['Efficiency'],row.loc['Fueltype']
            Cost = (Fuel_Data['Fuel Cost/Mwh'].loc[Fuel_type] / Power_Plant_Efficiency) + Fuel_Data['Variable Cost'].loc[Fuel_type] +(Fuel_Data['Emission_Factor'].loc[Fuel_type]/Power_Plant_Efficiency)*Fuel_Data['Fuel Cost/Mwh'].loc['Co2']   
            return Cost
        df['Marginal_Cost']=df.apply(Cost,axis=1)
        for i in df.index:
            df['Marginal_Cost'].loc[i]=df['Marginal_Cost'].loc[i] + WhiteNoise[i]
        return df
    df=Marginal_Cost(df)
    return df
# =============================================================================
# For Total renewable generation and per unit values
# =============================================================================

# As Pypsa only take one value per generator for p_nom so taking one big value for 
# renewable Generation and Total_RG with that value to generate the p_max_pu values
# so that we can add Renewable Generation as a generator and allow curtailment
'''Constant'''
#For Austria 15000
#For Belgium 5000
#For France 18000
#For Germany 60000
#For NL 26000

def Total_Renewable_Generation(df,Constant):
    df['Total_Renewable_Generation']=df.sum(axis=1)
    df['p_max_pu']=df['Total_Renewable_Generation']/Constant
    return df


# =============================================================================
# CWE cross_border_flow data
# =============================================================================

def NTC_pu_max_values(df,installed_Cap):
    df['Total_Flow']=df.sum(axis=1)
    df['Directional_Flow']=df[df.columns[2]]-df[df.columns[1]]
    df['p_max_pu']=df['Total_Flow']/installed_Cap
    return df

#def Capacity_Factors(df):
#    for i in df.index:
#        for j in Fuel_Data.index:
#            if df['Fueltype'][i]==j:
#                df['Capacity'][i]=df['Capacity'][i]*Fuel_Data['Capacity Factor'][j]
#            else:
#                pass
#    return df
        
def Hydro(df,constant):
    df['PU Values'] = df['Total Generation']/constant
    return df
        
        
def Shadow_ATC_fns(df,df2):
    df["AT-DE"] = df2["AT=>DE Initial"]
    df["BE-NL"] = df2["BE=>NL Initial"]
    df["BE-FR"] = df2["BE=>FR Initial"]
    df["NL-DE"] = df2["NL=>DE Initial"]
    df["FR-DE"] = df2["FR=>DE Initial"]
    df["DE-AT"] = df2["DE=>AT Initial"]
    df["NL-BE"] = df2["NL=>BE Initial"]
    df["FR-BE"] = df2["FR=>BE Initial"]
    df["DE-NL"] = df2["DE=>NL Initial"]
    df["DE-FR"] = df2["DE=>FR Initial"]
    return df
      
        
        
        
        
        
        
        
        
        
        
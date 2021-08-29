# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:23:05 2020

@author: Yash's PC
"""

import pandas as pd
import pypsa
import numpy as np
import Functions_Model as fns
from tqdm import tqdm
from pyomo.environ import Constraint, Var, Reals, Binary, Objective , ConcreteModel ,NonNegativeReals, minimize ,Set , Integers
from pypsa.descriptors import (get_switchable_as_dense, get_switchable_as_iter,
                          allocate_series_dataframes, zsum)
from pypsa.opt import (l_constraint, l_objective, LExpression, LConstraint,
                  patch_optsolver_record_memusage_before_solving,
                  empty_network, free_pyomo_initializers)
from pyomo.opt import SolverFactory


# ==================================

# =============================================================================
#Importing data and Processing
# =============================================================================
#Demand Data

Austria_Demand = pd.read_csv('Input/Austria/Austria_Demand.csv')
Belgium_Demand = pd.read_csv('Input/Belgium/Belgium_Demand.csv')
France_Demand = pd.read_csv('Input/France/France_Demand.csv')
Germany_Demand = pd.read_csv('Input/Germany/Germany_Demand.csv')
NL_Demand = pd.read_csv('Input/NL/NL_Demand.csv')

#Import Data

Austria_Import = pd.read_csv('Input/Austria/Austria_Import.csv')
Belgium_Import = pd.read_csv('Input/Belgium/Belgium_Import.csv')
France_Import = pd.read_csv('Input/France/France_Import.csv')
Germany_Import = pd.read_csv('Input/Germany/Germany_Import.csv')
NL_Import = pd.read_csv('Input/NL/NL_Import.csv')

#Export Data

Austria_Export=pd.read_csv('Input/Austria/Austria_Export.csv')
Belgium_Export=pd.read_csv('Input/Belgium/Belgium_Export.csv')
France_Export=pd.read_csv('Input/France/France_Export.csv')
Germany_Export=pd.read_csv('Input/Germany/Germany_Export.csv')
NL_Export=pd.read_csv('Input/NL/NL_Export.csv')

# Total_Demand

Austria_Demand = fns.Total_Demand( Austria_Demand, Austria_Import, Austria_Export)
Belgium_Demand = fns.Total_Demand( Belgium_Demand, Belgium_Import, Belgium_Export)
France_Demand =  fns.Total_Demand( France_Demand, France_Import, France_Export)
Germany_Demand =  fns.Total_Demand( Germany_Demand, Germany_Import, Germany_Export)
NL_Demand =  fns.Total_Demand( NL_Demand, NL_Import, NL_Export)

# =============================================================================
#Powerplant_Data and Fuel data
#Adding Marginal cost and Efficiency
# =============================================================================

Austria_Powerplants = pd.read_csv('Input/Austria/Austria_Powerplants.csv',index_col=0)
Belgium_Powerplants = pd.read_csv('Input/Belgium/Belgium_Powerplants.csv',index_col=0)
France_Powerplants = pd.read_csv('Input/France/France_Powerplants.csv',index_col=0)
Germany_Powerplants = pd.read_csv('Input/Germany/Germany_Powerplants.csv',index_col=0)
NL_Powerplants = pd.read_csv('Input/NL/NL_Powerplants.csv',index_col=0)
Fuel_Data = pd.read_csv('Input/Fuel Costs.csv',index_col=0)

#Removing decommisioned powerplant and hydro powerplants

Austria_Powerplants = fns.Decommissioned_Hydro_Drop(Austria_Powerplants,2018)
Belgium_Powerplants = fns.Decommissioned_Hydro_Drop(Belgium_Powerplants,2018)
France_Powerplants = fns.Decommissioned_Hydro_Drop(France_Powerplants,2018)
Germany_Powerplants = fns.Decommissioned_Hydro_Drop(Germany_Powerplants,2018)
NL_Powerplants = fns.Decommissioned_Hydro_Drop(NL_Powerplants,2018)

#Marginal Cost and efficiency 

Austria_Powerplants = fns.Efficiency_Cost(Austria_Powerplants)
Belgium_Powerplants = fns.Efficiency_Cost(Belgium_Powerplants)
France_Powerplants = fns.Efficiency_Cost(France_Powerplants)
Germany_Powerplants = fns.Efficiency_Cost(Germany_Powerplants)
NL_Powerplants = fns.Efficiency_Cost(NL_Powerplants)
# =============================================================================
# Renewable data and processing
# Calculation total generation and per unit(pu) values
# =============================================================================

Austria_Renewable_Generation = pd.read_csv('Input/Austria/Austria_Renewable_Generation.csv')
Belgium_Renewable_Generation = pd.read_csv('Input/Belgium/Belgium_Renewable_Generation.csv')
France_Renewable_Generation = pd.read_csv('Input/France/France_Renewable_Generation.csv')
Germany_Renewable_Generation = pd.read_csv('Input/Germany/Germany_Renewable_Generation.csv')
NL_Renewable_Generation = pd.read_csv('Input/NL/NL_Renewable_Generation.csv')

Austria_Renewable_Generation = fns.Total_Renewable_Generation(Austria_Renewable_Generation,15000)
Belgium_Renewable_Generation = fns.Total_Renewable_Generation(Belgium_Renewable_Generation,5000)
France_Renewable_Generation = fns.Total_Renewable_Generation(France_Renewable_Generation,18000)
Germany_Renewable_Generation = fns.Total_Renewable_Generation(Germany_Renewable_Generation,60000)
NL_Renewable_Generation = fns.Total_Renewable_Generation(NL_Renewable_Generation,26000)

# =============================================================================
# Cross_border_Flow (ENTSO-e data)
# =============================================================================
#Austria_Germany_CBF = pd.read_csv('Input/Cross_Border_Flow/Austria_Germany.csv')
#Belgium_Germany_CBF = pd.read_csv('Input/Cross_Border_Flow/Belgium_Germany.csv')
#Belgium_France_CBF = pd.read_csv('Input/Cross_Border_Flow/Belgium_France.csv')
#Belgium_NL_CBF = pd.read_csv('Input/Cross_Border_Flow/Belgium_NL.csv')
#France_Germany_CBF = pd.read_csv('Input/Cross_Border_Flow/France_Germany.csv')
#NL_Germany_CBF = pd.read_csv('Input/Cross_Border_Flow/NL_Germany.csv')

#Processing (Trail for NTC value estimation for CWE)

#Austria_Germany_CBF=fns.NTC_pu_max_values(Austria_Germany_CBF,5000)
#Belgium_Germany_CBF=fns.NTC_pu_max_values(Belgium_Germany_CBF,500)
#Belgium_France_CBF=fns.NTC_pu_max_values(Belgium_France_CBF,5000)
#Belgium_NL_CBF=fns.NTC_pu_max_values(Belgium_NL_CBF,5000)
#France_Germany_CBF=fns.NTC_pu_max_values(France_Germany_CBF,5500)
#NL_Germany_CBF=fns.NTC_pu_max_values(NL_Germany_CBF,5500)

# =============================================================================
# Hydro Generation
# =============================================================================

Austria_Hydro=pd.read_csv('Input/Austria/Austria_Hydro.csv')
Belgium_Hydro=pd.read_csv('Input/Belgium/Belgium_Hydro.csv')
France_Hydro=pd.read_csv('Input/France/France_Hydro.csv')
Germany_Hydro=pd.read_csv('Input/Germany/Germany_Hydro.csv')
NL_Hydro=pd.read_csv('Input/NL/NL_Hydro.csv')

# Processing data

Austria_Hydro = fns.Hydro(Austria_Hydro,10000)
Belgium_Hydro = fns.Hydro(Belgium_Hydro,1500)
France_Hydro = fns.Hydro(France_Hydro,17000)
Germany_Hydro = fns.Hydro(Germany_Hydro,11000)
NL_Hydro = fns.Hydro(NL_Hydro,100)

# =============================================================================
# Building Network
# =============================================================================

#Adding new variables to the model

override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Bus"].loc["NEX"] = ["static or series","MW",0,"Net exchange","Output"]


Bus=['Austria','Belgium','France','Germany','NL']
x=[0 for i in range(10968)]

network=pypsa.Network(override_component_attrs=override_component_attrs)
network.set_snapshots(range(10968))
for i in Bus:
    network.add('Bus', i)
 
#Adding loads to buses (nodes)

network.add('Load' , 'Austria_Total_Demand' , bus = 'Austria' ,
                         p_set = Austria_Demand['Total_Demand'])
network.add('Load' , 'Belgium_Total_Demand' , bus = 'Belgium' ,
                         p_set = Belgium_Demand['Total_Demand'])
network.add('Load' , 'France_Total_Demand' , bus = 'France' ,
                         p_set = France_Demand['Total_Demand'])
network.add('Load' , 'Germany_Total_Demand' , bus = 'Germany' ,
                         p_set = Germany_Demand['Total_Demand'])
network.add('Load' , 'NL_Total_Demand' , bus = 'NL' ,
                         p_set = NL_Demand['Total_Demand'])

#Adding Conventional Generators 

for i in Austria_Powerplants.index:
    network.add('Generator' , str(i) + '.' + Austria_Powerplants['Name'][i] , bus='Austria' ,
                             p_nom = Austria_Powerplants['Capacity'][i],
                             marginal_cost = Austria_Powerplants['Marginal_Cost'][i])
for i in Belgium_Powerplants.index:
    network.add('Generator' , str(i) + '.' + Belgium_Powerplants['Name'][i] , bus='Belgium' ,
                             p_nom = Belgium_Powerplants['Capacity'][i],
                             marginal_cost = Belgium_Powerplants['Marginal_Cost'][i])
for i in France_Powerplants.index:
    network.add('Generator' , str(i) + '.' + France_Powerplants['Name'][i] , bus='France' ,
                             p_nom = France_Powerplants['Capacity'][i],
                             marginal_cost = France_Powerplants['Marginal_Cost'][i])
for i in Germany_Powerplants.index:
    network.add('Generator' , str(i) + '.' + Germany_Powerplants['Name'][i] , bus='Germany' ,
                             p_nom = Germany_Powerplants['Capacity'][i],
                             marginal_cost = Germany_Powerplants['Marginal_Cost'][i])
for i in NL_Powerplants.index:
    network.add('Generator' , str(i) + '.' + NL_Powerplants['Name'][i] , bus='NL' ,
                             p_nom = NL_Powerplants['Capacity'][i],
                             marginal_cost = NL_Powerplants['Marginal_Cost'][i])

#Adding Renewable Generation as Generators:

network.add('Generator' , 'Austria_Renewable_Generation' ,
                  bus = 'Austria' , p_nom = 15000 ,
                  p_min_pu = x ,
                  p_max_pu = Austria_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)
network.add('Generator' , 'Belgium_Renewable_Generation' ,
                  bus = 'Belgium' , p_nom = 5000 ,
                  p_min_pu = x ,
                  p_max_pu = Belgium_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)
network.add('Generator' , 'France_Renewable_Generation' ,
                  bus = 'France' , p_nom = 18000 ,
                  p_min_pu = x ,
                  p_max_pu = France_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)
network.add('Generator' , 'Germany_Renewable_Generation' ,
                  bus = 'Germany' , p_nom = 60000 ,
                  p_min_pu = x ,
                  p_max_pu = Germany_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)
network.add('Generator' , 'NL_Renewable_Generation' ,
                  bus = 'NL' , p_nom = 26000 ,
                  p_min_pu = x ,
                  p_max_pu = NL_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)

#Adding Hydro Generator:

network.add('Generator' , 'Austria_Hydro' ,
            bus = 'Austria' , p_nom = 10000 ,
            p_min_pu = [0 for i in range(10968)] ,
            p_max_pu = Austria_Hydro['PU Values'] ,
            marginal_cost = 0)
network.add('Generator' , 'Belgium_Hydro' ,
            bus = 'Belgium' , p_nom = 1500 ,
            p_min_pu = [0 for i in range(10968)] ,
            p_max_pu = Belgium_Hydro['PU Values'] ,
            marginal_cost = 0)
network.add('Generator' , 'France_Hydro' ,
            bus = 'France' , p_nom = 17000 ,
            p_min_pu = [0 for i in range(10968)] ,
            p_max_pu = France_Hydro['PU Values'] ,
            marginal_cost = 0)
network.add('Generator' , 'Germany_Hydro' ,
            bus = 'Germany' , p_nom = 11000 ,
            p_min_pu = [0 for i in range(10968)] ,
            p_max_pu = Germany_Hydro['PU Values'] ,
            marginal_cost = 0)
network.add('Generator' , 'NL_Hydro' ,
            bus = 'NL' , p_nom = 100 ,
            p_min_pu = [0 for i in range(10968)] ,
            p_max_pu = NL_Hydro['PU Values'] ,
            marginal_cost = 0)
                  
# Adding lines (Very high flow capacity, Redundant)

network.add('Link' , 'Austria-Germany',
                         bus0 = 'Austria',
                         bus1 = 'Germany',
                         efficiency = 1,
                         marginal_cost = 0,
                         p_nom = 10000,
                         p_min_pu = -1 )

network.add('Link' , 'France-Germany',
                         bus0 = 'France',
                         bus1 = 'Germany',
                         efficiency = 1,
                         marginal_cost = 0,
                         p_nom = 10000,
                         p_min_pu = -1 )
network.add('Link' , 'NL-Germany',
                         bus0 = 'NL',
                         bus1 = 'Germany',
                         efficiency = 1,
                         marginal_cost = 0,
                         p_nom = 10000,
                         p_min_pu = -1)
network.add('Link' , 'Belgium-NL',
                         bus0 = 'Belgium',
                         bus1 = 'NL',
                         efficiency = 1,
                         marginal_cost = 0,
                         p_nom = 10000,
                         p_min_pu = -1)
network.add('Link' , 'Belgium-France' ,
                         bus0 = 'Belgium',
                         bus1 = 'France',
                         efficiency = 1 ,
                         marginal_cost = 0,
                         p_nom = 10000,
                         p_min_pu = -1)

#Common Dataframe for directional cross border flow

#CBF = pd.DataFrame(index = range(10968), columns = list(network.links.index))
#CBF["Austria-Germany"] = Austria_Germany_CBF["Directional_Flow"]
#CBF["Belgium-Germany"] = Belgium_Germany_CBF["Directional_Flow"]
#CBF["France-Germany"] = France_Germany_CBF["Directional_Flow"]
#CBF["NL-Germany"] = NL_Germany_CBF["Directional_Flow"]
#CBF["Belgium-NL"] = Belgium_NL_CBF["Directional_Flow"]
#CBF["Belgium-France"] = Belgium_France_CBF["Directional_Flow"]

# =============================================================================
# For Flow based Market coupling
# =============================================================================


Ptdf = pd.read_csv("Input/PTDF.csv")

# Generating list of Critical lines for each hour

Ptdf['Diff'] = Ptdf['ticks'].diff()
CB_Count = list(Ptdf[Ptdf['Diff']==1].index)
CB_Count.insert(0,0)
CB_Count.insert(len(CB_Count),len(Ptdf.index))
Ptdf['ticks'] = Ptdf['ticks'].astype(int)

#%% test  

# =============================================================================
# Creating constraints to add flow based parameter
# =============================================================================
def extra_functionality(network,snapshots):
    print(snapshots)
    fixed_links_i = network.links.index[~ network.links.p_nom_extendable]
    
    model = network.model
    load_p_set = get_switchable_as_dense(network, 'Load', 'p_set', snapshots)

    
    model.bus_NEX = Var(list(network.buses.index), list(snapshots), domain = Reals)
    def NEX_Rule(model,bus,sn):
        count["0"] += 1
        x=0
        for i in network.generators[network.generators.bus == bus].index:
            x += model.generator_p[i,sn]
        load = list(network.loads[network.loads.bus == bus].index)
        return model.bus_NEX[bus,sn] == x - load_p_set.at[sn,load[0]]
    model.NEX_cons = Constraint(list(network.buses.index), list(snapshots), rule = NEX_Rule)  
    
    # line limit variable for the Critical Branch(cb)
       
    model.line_limit = Var(list(range(CB_Count[snapshots[0]],CB_Count[snapshots[-1]+1])), domain = Reals)
    # Relation with the net exchange value. Gives us the flow on that line
    
    def Limits(model, cb):
        return model.line_limit[cb] == (model.bus_NEX["Austria",Ptdf['ticks'].loc[cb]]*Ptdf["AT"].loc[cb] + 
                                       model.bus_NEX["Belgium",Ptdf['ticks'].loc[cb]]*Ptdf["BE"].loc[cb] +
                                       model.bus_NEX["France",Ptdf['ticks'].loc[cb]]*Ptdf["FR"].loc[cb] +
                                       model.bus_NEX["Germany",Ptdf['ticks'].loc[cb]]*Ptdf["DE"].loc[cb] + 
                                       model.bus_NEX["NL",Ptdf['ticks'].loc[cb]]*Ptdf["NL"].loc[cb])
        
    model.line_limit_cons = Constraint(list(range(CB_Count[snapshots[0]],CB_Count[snapshots[-1]+1])), rule = Limits)
    
    # Constraints that keep the flow on the line within the Remaining Available Margin(RAM)
     
    def FB_Limits_pos(model, cb):
        return model.line_limit[cb] <= Ptdf["RemainingAvailableMargin"].loc[cb]
    
    model.FB_limits_pos_cons = Constraint(list(range(CB_Count[snapshots[0]],CB_Count[snapshots[-1]+1])), rule = FB_Limits_pos)
    
    def FB_Limits_neg(model, cb):
        return model.line_limit[cb] >= -Ptdf["RemainingAvailableMargin"].loc[cb]
    
    model.FB_limits_neg_cons = Constraint(list(range(CB_Count[snapshots[0]],CB_Count[snapshots[-1]+1])), rule = FB_Limits_neg)
    def total_nex(model,sn):
        count["1"] += 1 
        return model.bus_NEX["Austria",sn] + model.bus_NEX["Belgium",sn] + model.bus_NEX["France",sn] + model.bus_NEX["Germany",sn] + model.bus_NEX["NL",sn] == 0
    model.total_nex_cons = Constraint(list(snapshots), rule = total_nex)      


Bus_NEX = pd.DataFrame(index = range(10968), columns = network.buses.index)\

# Dual values of constraint
duals_df = pd.DataFrame(columns = ["Duals"])
def extra_postprocessing(network,sanpshots,duals):
    global duals_df
    df_temp = pd.DataFrame()
    df_temp["Duals"] = duals
    df_temp.index = df_temp.index.astype(str)
    duals_df = pd.concat([duals_df,df_temp])
    return duals_df

# =============================================================================
# Solving in steps of 20 hours at a time
# =============================================================================
    
#range_exceed=False
#f = 20
#ind = 0
#timestep=range(0,f)
#while range_exceed==False:    
#    if max(timestep) < 10968: 
#        network.lopf( timestep, solver_name='gurobi', extra_functionality = extra_functionality, extra_postprocessing = extra_postprocessing)
#        for i in tqdm(timestep):
#            for j in Bus_NEX.columns:
#                Bus_NEX[j].loc[i] = network.model.bus_NEX[j,i].value
#        f+=20
#        timestep=range(f-20,f)
#        ind+= 1
#    elif max(timestep) > 10968:
#        timestep = range(f-20,10968)
#        network.lopf( timestep, solver_name='gurobi', extra_functionality = extra_functionality, extra_postprocessing = extra_postprocessing)
#        for i in timestep:
#            for j in Bus_NEX.columns:
#                Bus_NEX[j].loc[i] = network.model.bus_NEX[j,i].value
#        print('done')
#        range_exceed = True

# =============================================================================
# Test runs
# =============================================================================

for i in tqdm(range(2)):
    network.lopf(network.snapshots[i],solver_name='gurobi', extra_functionality = extra_functionality, extra_postprocessing = extra_postprocessing)
    for j in Bus_NEX.columns:
        Bus_NEX[j].loc[i] = network.model.bus_NEX[j,i].value
#%%

CrossBorderFlow = network.links_t.p0

#marginal cost for each zone, Generator Output

Marginal_Cost = network.buses_t.marginal_price
Generator_Output = network.generators_t.p

#%%
# Getting generator output for each zone
GO_Austria = network.generators_t.p[network.generators[network.generators.bus == "Austria"].index]
GO_Belgium = network.generators_t.p[network.generators[network.generators.bus == "Belgium"].index]
GO_France = network.generators_t.p[network.generators[network.generators.bus == "France"].index]
GO_Germany = network.generators_t.p[network.generators[network.generators.bus == "Germany"].index]
GO_NL = network.generators_t.p[network.generators[network.generators.bus == "NL"].index]

# =============================================================================
# Trail for estimating marginal cost
# =============================================================================

# Calculates the marginal cost of the most expensive powerplant in the zone for each hour

#MC = pd.DataFrame(index = range(10968), columns = network.buses.index)
#
#def MC_replace(df,column):
#    def DF_Replace(row):
#        for i in df.columns:
#            if row[i] == 0:
#                pass
#            else:
#                row[i] = network.generators.marginal_cost[i]
#        return row
#    df = df.apply(DF_Replace, axis = 1)
#    MC[column] = df.max(axis=1)
#    return MC[column]
#        
#
#MC["Austria"] = MC_replace(GO_Austria, "Austria")
#MC["Belgium"] = MC_replace(GO_Belgium, "Belgium")
#MC["France"] = MC_replace(GO_France, "France")
#MC["Germany"] = MC_replace(GO_Germany, "Germany")
#MC["NL"] = MC_replace(GO_NL, "NL")

# =============================================================================
# Correct method for MC
# =============================================================================
# As FBMC creates a new more restrainting power balance, calculating the correct marginal costs

new_mc = pd.DataFrame(index = Marginal_Cost.index, columns = Marginal_Cost.columns)
for i in tqdm(range(10968)):
    for j in new_mc.columns:
        new_mc[j].loc[i] = duals_df["Duals"].loc['power_balance['+str(j)+","+str(i)+"]"] - duals_df["Duals"].loc["NEX_cons["+str(j)+","+str(i)+"]"] 

##
#
#
#MC.to_csv("Output/FBMC/MarginalCost.csv")
#BUS_NEX.to_csv("Output/FBMC/NEX_FBMC.csv")

#%%

# calculating market results
        
total_cost = network.generators_t.p * network.generators.marginal_cost 

revenue = pd.DataFrame(index = range(10968), columns = network.generators.index)

for j in revenue.columns:
    revenue[j] = network.generators_t.p[j] * new_mc[network.generators.bus[j]] 


consumer_cost = pd.DataFrame(index = range(10968), columns = network.buses.index)
consumer_cost["Austria"] = new_mc["Austria"]*Austria_Demand["Total_Demand"] 
consumer_cost["Belgium"] = new_mc["Belgium"]*Belgium_Demand["Total_Demand"] 
consumer_cost["France"] = new_mc["France"]*France_Demand["Total_Demand"] 
consumer_cost["Germany"] = new_mc["Germany"]*Germany_Demand["Total_Demand"] 
consumer_cost["NL"] = new_mc["NL"]*NL_Demand["Total_Demand"]         
        
total_cost = network.generators_t.p * network.generators.marginal_cost 

revenue = pd.DataFrame(index = range(10968), columns = network.generators.index)

for j in revenue.columns:
    revenue[j] = network.generators_t.p[j] * new_mc[network.generators.bus[j]]        
        
cost_gen_austria = network.generators_t.p[network.generators[network.generators.bus == "Austria"].index]
mc_gen_austria = Austria_Powerplants["Marginal_Cost"]

cost_gen_belgium = network.generators_t.p[network.generators[network.generators.bus == "Belgium"].index]
mc_gen_belgium = Belgium_Powerplants["Marginal_Cost"]

cost_gen_france = network.generators_t.p[network.generators[network.generators.bus == "France"].index]
mc_gen_france = France_Powerplants["Marginal_Cost"]

cost_gen_germany = network.generators_t.p[network.generators[network.generators.bus == "Germany"].index]
mc_gen_germany = Germany_Powerplants["Marginal_Cost"]

cost_gen_nl = network.generators_t.p[network.generators[network.generators.bus == "NL"].index]
mc_gen_nl = NL_Powerplants["Marginal_Cost"]
renewable_mc = pd.DataFrame([0,0])
def cost_gen(cost_gen_df,mc):
    mc = mc.append(renewable_mc)
    for i in tqdm(range(len(cost_gen_df.columns))):
        cost_gen_df[cost_gen_df.columns[i]] = cost_gen_df[cost_gen_df.columns[i]] * mc[0].loc[mc.index[i]]
    return cost_gen_df

cost_gen_austria = cost_gen(cost_gen_austria,mc_gen_austria)
cost_gen_belgium = cost_gen(cost_gen_belgium,mc_gen_belgium)
cost_gen_france = cost_gen(cost_gen_france,mc_gen_france)
cost_gen_germany = cost_gen(cost_gen_germany,mc_gen_germany)
cost_gen_nl = cost_gen(cost_gen_nl,mc_gen_nl)


revenue_austria = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "Austria"].index)

for j in tqdm(revenue_austria.columns):
    revenue_austria[j] = network.generators_t.p[j] * new_mc[network.generators.bus[j]] 

revenue_belgium = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "Belgium"].index)

for j in tqdm(revenue_belgium.columns):
    revenue_belgium[j] = network.generators_t.p[j] * new_mc[network.generators.bus[j]] 

revenue_france = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "France"].index)

for j in tqdm(revenue_france.columns):
    revenue_france[j] = network.generators_t.p[j] * new_mc[network.generators.bus[j]] 

revenue_germany = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "Germany"].index)

for j in tqdm(revenue_germany.columns):
    revenue_germany[j] = network.generators_t.p[j] * new_mc[network.generators.bus[j]] 
    
revenue_nl = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "NL"].index)

for j in tqdm(revenue_nl.columns):
    revenue_nl[j] = network.generators_t.p[j] * new_mc[network.generators.bus[j]]
    
#%%
    
# Just some checks on nodal balances 

#Nodal_balance = pd.DataFrame(index = range(10968), columns = network.buses.index)
#
#Nodal_balance["Austria"] = GO_Austria.sum(axis=1) - Austria_Demand["Total_Demand"] - CrossBorderFlow["Austria-Germany"]
#Nodal_balance["Belgium"] = GO_Belgium.sum(axis=1) - Belgium_Demand["Total_Demand"] - CrossBorderFlow["Belgium-France"] - CrossBorderFlow["Belgium-NL"]
#Nodal_balance["France"] = GO_France.sum(axis=1) - France_Demand["Total_Demand"] - CrossBorderFlow["France-Germany"] + CrossBorderFlow["Belgium-France"]
#Nodal_balance["Germany"] = GO_Germany.sum(axis=1) - Germany_Demand["Total_Demand"] + CrossBorderFlow["Austria-Germany"] + CrossBorderFlow["NL-Germany"] + CrossBorderFlow["France-Germany"] 
#Nodal_balance["NL"] = GO_NL.sum(axis=1) - NL_Demand["Total_Demand"] - CrossBorderFlow["NL-Germany"] + CrossBorderFlow["Belgium-NL"]








# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:20:20 2020

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
        
#Efficiecny and Marginal Cost

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

##Processing (Trail for NTC value estimation for CWE region)

#Austria_Germany_CBF=fns.NTC_pu_max_values(Austria_Germany_CBF,5000)
#Belgium_Germany_CBF=fns.NTC_pu_max_values(Belgium_Germany_CBF,500)
#Belgium_France_CBF=fns.NTC_pu_max_values(Belgium_France_CBF,5000)
#Belgium_NL_CBF=fns.NTC_pu_max_values(Belgium_NL_CBF,5000)
#France_Germany_CBF=fns.NTC_pu_max_values(France_Germany_CBF,5500)
#NL_Germany_CBF=fns.NTC_pu_max_values(NL_Germany_CBF,5500)
ATC_value = pd.read_csv("Input/Shadow_ATC.csv")

# ATC values

Shadow_ATC = pd.DataFrame(index = range(10968), columns = ["AT-DE","BE-NL","BE-FR","NL-DE","FR-DE",
              "DE-AT","NL-BE","FR-BE","DE-NL","DE-FR"])
Shadow_ATC = fns.Shadow_ATC_fns(Shadow_ATC, ATC_value)
Shadow_ATC_pu = Shadow_ATC/10000

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

# ============================================================================
# Building Network
# =============================================================================

override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Bus"].loc["NEX"] = ["static or series","MW",0,"Net exchange","Output"]

Bus=['Austria','Belgium','France','Germany','NL']
x=[0 for i in range(10968)]

network=pypsa.Network(override_component_attrs=override_component_attrs)

network.set_snapshots(range(10968))
for i in Bus:
    network.add('Bus', i)

#Adding loads to buses (Nodes)

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
                  p_min_pu = [0 for i in range(10968)] ,
                  p_max_pu = Austria_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)
network.add('Generator' , 'Belgium_Renewable_Generation' ,
                  bus = 'Belgium' , p_nom = 5000 ,
                  p_min_pu = [0 for i in range(10968)] ,
                  p_max_pu = Belgium_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)
network.add('Generator' , 'France_Renewable_Generation' ,
                  bus = 'France' , p_nom = 18000 ,
                  p_min_pu = [0 for i in range(10968)] ,
                  p_max_pu = France_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)
network.add('Generator' , 'Germany_Renewable_Generation' ,
                  bus = 'Germany' , p_nom = 60000 ,
                  p_min_pu = [0 for i in range(10968)] ,
                  p_max_pu = Germany_Renewable_Generation['p_max_pu'] ,
                  marginal_cost = 0)
network.add('Generator' , 'NL_Renewable_Generation' ,
                  bus = 'NL' , p_nom = 26000 ,
                  p_min_pu = [0 for i in range(10968)] ,
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
                 
# Adding lines
# =============================================================================
# Considering All Shadow values (BI-directional lines)
# =============================================================================
network.add('Link' , 'Austria-Germany' ,
                         bus0 = 'Austria',
                         bus1 = 'Germany',
                         efficiency = 1 ,
                         marginal_cost = 0 ,
                         p_nom = 10000 ,
                         p_min_pu = -1*Shadow_ATC_pu["DE-AT"],
                         p_max_pu = Shadow_ATC_pu["AT-DE"])
network.add('Link' , 'France-Germany' ,
                         bus0 = 'France',
                         bus1 = 'Germany',
                         efficiency = 1 ,
                         marginal_cost = 0 ,
                         p_nom = 10000 ,
                         p_min_pu = -1*Shadow_ATC_pu["DE-FR"],
                         p_max_pu = Shadow_ATC_pu["FR-DE"])
network.add('Link' , 'NL-Germany' ,
                         bus0 = 'NL',
                         bus1 = 'Germany',
                         efficiency = 1 ,
                         marginal_cost = 0 ,
                         p_nom = 10000 ,
                         p_min_pu = -1*Shadow_ATC_pu["DE-NL"],
                         p_max_pu = Shadow_ATC_pu["NL-DE"])
network.add('Link' , 'Belgium-NL' ,
                         bus0 = 'Belgium',
                         bus1 = 'NL',
                         efficiency = 1 ,
                         marginal_cost = 0 ,
                         p_nom = 10000,
                         p_min_pu = -1*Shadow_ATC_pu["NL-BE"],
                         p_max_pu = Shadow_ATC_pu["BE-NL"])
network.add('Link' , 'Belgium-France' ,
                         bus0 = 'Belgium',
                         bus1 = 'France',
                         efficiency = 1 ,
                         marginal_cost = 0 ,
                         p_nom = 10000,
                         p_min_pu = -1*Shadow_ATC_pu["FR-BE"],
                         p_max_pu = Shadow_ATC_pu["BE-FR"]) 

# =============================================================================
# Uni-directional lines
# =============================================================================
#network.add('Link' , 'Austria-Germany' ,
#                         bus0 = 'Austria',
#                         bus1 = 'Germany',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000 ,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["AT-DE"])
#network.add('Link' , 'Germany-Austria' ,
#                         bus0 = 'Germany',
#                         bus1 = 'Austria',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000 ,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["DE-AT"])
#network.add('Link' , 'France-Germany' ,
#                         bus0 = 'France',
#                         bus1 = 'Germany',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000 ,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["FR-DE"])
#network.add('Link' , 'Germany-France' ,
#                         bus0 = 'Germany',
#                         bus1 = 'France',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000 ,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["DE-FR"])
#network.add('Link' , 'NL-Germany' ,
#                         bus0 = 'NL',
#                         bus1 = 'Germany',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000 ,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["NL-DE"])
#network.add('Link' , 'Germany-NL' ,
#                         bus0 = 'Germany',
#                         bus1 = 'NL',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000 ,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["DE-NL"])
#network.add('Link' , 'Belgium-NL' ,
#                         bus0 = 'Belgium',
#                         bus1 = 'NL',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["BE-NL"])
#network.add('Link' , 'NL-Belgium' ,
#                         bus0 = 'NL',
#                         bus1 = 'Belgium',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["NL-BE"])
#network.add('Link' , 'Belgium-France' ,
#                         bus0 = 'Belgium',
#                         bus1 = 'France',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["BE-FR"]) 
#network.add('Link' , 'France-Belgium' ,
#                         bus0 = 'France',
#                         bus1 = 'Belgium',
#                         efficiency = 1 ,
#                         marginal_cost = 0 ,
#                         p_nom = 10000,
#                         p_min_pu = [0 for i in range(10968)],
#                         p_max_pu = Shadow_ATC_pu["FR-BE"]) 
      
# =============================================================================
# Creating constraints for generating net exchanges values (LP modifications)
# =============================================================================
def extra_functionality(network,snapshots): 
    print(snapshots)
    fixed_links_i = network.links.index[~ network.links.p_nom_extendable]

    model = network.model
    load_p_set = get_switchable_as_dense(network, 'Load', 'p_set', snapshots)
#    model.y =  Var(list(range(5)),list(snapshots), domain = Binary)

    model.bus_NEX = Var(list(network.buses.index), list(snapshots), domain = Reals)
    def NEX_Rule(model,bus,sn):
        x=0

        for i in network.generators[network.generators.bus == bus].index:
            x += model.generator_p[i,sn]
        load = list(network.loads[network.loads.bus == bus].index)
        return model.bus_NEX[bus,sn] == x - load_p_set.at[sn,load[0]]
    model.NEX_cons = Constraint(list(network.buses.index), list(snapshots), rule = NEX_Rule)    
    
# Dual values of constraint
    
duals_df = pd.DataFrame(columns = ["Duals"])
def extra_postprocessing(network,sanpshots,duals):
    global duals_df
    df_temp = pd.DataFrame()
    df_temp["Duals"] = duals
    df_temp.index = df_temp.index.astype(str)
    duals_df = pd.concat([duals_df,df_temp])
    return duals_df


Bus_NEX = pd.DataFrame(index = range(10968), columns = network.buses.index)

# Network linear power flow (Solving)
range_exceed=False
f = 457
ind = 0
timestep=range(0,f)
while range_exceed==False:    
    if max(timestep) < 10968: 
        network.lopf( timestep, solver_name='gurobi', extra_functionality = extra_functionality, extra_postprocessing = extra_postprocessing)
        for i in timestep:
            for j in Bus_NEX.columns:
                Bus_NEX[j].loc[i] = network.model.bus_NEX[j,i].value
        f+=457
        timestep=range(f-457,f)
        ind+= 1
    else:
        range_exceed=True

# =============================================================================
# Calculating different market results
# =============================================================================

CrossBorderFlow = network.links_t.p0
Marginal_Cost = network.buses_t.marginal_price
Generator_Output = network.generators_t.p

total_cost = network.generators_t.p * network.generators.marginal_cost 

revenue = pd.DataFrame(index = range(10968), columns = network.generators.index)

for j in revenue.columns:
    revenue[j] = network.generators_t.p[j] * Marginal_Cost[network.generators.bus[j]] 


consumer_cost = pd.DataFrame(index = range(10968), columns = network.buses.index)
consumer_cost["Austria"] = Marginal_Cost["Austria"]*Austria_Demand["Total_Demand"] 
consumer_cost["Belgium"] = Marginal_Cost["Belgium"]*Belgium_Demand["Total_Demand"] 
consumer_cost["France"] = Marginal_Cost["France"]*France_Demand["Total_Demand"] 
consumer_cost["Germany"] = Marginal_Cost["Germany"]*Germany_Demand["Total_Demand"] 
consumer_cost["NL"] = Marginal_Cost["NL"]*NL_Demand["Total_Demand"] 

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
    revenue_austria[j] = network.generators_t.p[j] * Marginal_Cost[network.generators.bus[j]] 

revenue_belgium = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "Belgium"].index)

for j in tqdm(revenue_belgium.columns):
    revenue_belgium[j] = network.generators_t.p[j] * Marginal_Cost[network.generators.bus[j]] 

revenue_france = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "France"].index)

for j in tqdm(revenue_france.columns):
    revenue_france[j] = network.generators_t.p[j] * Marginal_Cost[network.generators.bus[j]] 

revenue_germany = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "Germany"].index)

for j in tqdm(revenue_germany.columns):
    revenue_germany[j] = network.generators_t.p[j] * Marginal_Cost[network.generators.bus[j]] 
    
revenue_nl = pd.DataFrame(index = range(10968), columns = network.generators[network.generators.bus == "NL"].index)

for j in tqdm(revenue_nl.columns):
    revenue_nl[j] = network.generators_t.p[j] * Marginal_Cost[network.generators.bus[j]] 
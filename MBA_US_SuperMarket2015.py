# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:00:22 2020

@author: Asus
"""


#Import Libraries----------

import numpy as np 

import pandas as pd 

from mlxtend.frequent_patterns import apriori, association_rules 



#Loading and exploring the data-----------------

#Loading the Data 

data = pd.read_excel('SuperStoreUS-2015.xlsx') 

data.info()

data.head(10)
#data.tail(10) 

# Exploring the columns of the data 


# my data is huge so i am using only those cols which will be useful for further analysis

data = data.loc[:,['Product Sub-Category','Product Name','Region','Province','Quantity ordered new','Order ID']]
data.info()


# Exploring the different regions of transactions 

data.Province.unique() 

data.Region.unique()


#Cleaning the Data-----------------

# Stripping white spaces in the description its better to remove them.

data['Product Sub-Category'] = data['Product Sub-Category'].str.strip() 



# Dropping the rows without any invoice number 

data.info()

data.dropna(axis = 0, subset =['Order ID'], inplace = True) 

data['Order ID'] = data['Order ID'].astype('str') 


#Splitting the data according to the region of transaction-------

province = data.Province.unique()

region = data.Region.unique()

# Transactions done in WEST

basket_West = (data[data['Region'] =="West"] 

		.groupby(['Order ID', 'Product Sub-Category'])['Quantity ordered new'] 

		.sum().unstack().reset_index().fillna(0) 

		.set_index('Order ID')) 



# Transactions done in the East
 
basket_East = (data[data['Region'] =="East"] 

		.groupby(['Order ID', 'Product Sub-Category'])['Quantity ordered new'] 

		.sum().unstack().reset_index().fillna(0) 

		.set_index('Order ID')) 




# Transactions done in South

basket_South = (data[data['Region'] =="South"] 

		.groupby(['Order ID', 'Product Sub-Category'])['Quantity ordered new'] 

		.sum().unstack().reset_index().fillna(0) 

		.set_index('Order ID')) 

# Transactions done in Central


basket_Central = (data[data['Region'] =="Central"] 

		.groupby(['Order ID', 'Product Sub-Category'])['Quantity ordered new'] 

		.sum().unstack().reset_index().fillna(0) 

		.set_index('Order ID')) 



#Hot encoding the Data------------

# Defining the hot encoding function to make the data suitable 

# for the concerned libraries 
# beacuse the baskets we have created should give values 0 and 1


def hot_encode(x): 

	if(x<= 0): 

		return 0

	if(x>= 1): 

		return 1



# Encoding the datasets 

basket_encoded = basket_West.applymap(hot_encode) 

basket_West = basket_encoded 



basket_encoded = basket_East.applymap(hot_encode) 

basket_East = basket_encoded 



basket_encoded = basket_South.applymap(hot_encode) 

basket_South = basket_encoded 



basket_encoded = basket_Central.applymap(hot_encode) 

basket_Central = basket_encoded 



#Building the models and analyzing the results-----------------



#WEST:

# Building the model for WEST 

frq_items = apriori(basket_West, min_support = 0.001, use_colnames = True) 



# Collecting the inferred rules in a dataframe 

rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 

rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

print(rules.head()) 

West_rules=pd.DataFrame(rules)



#East
frq_items = apriori(basket_East, min_support = 0.001, use_colnames = True) 

rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 

rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

print(rules.head(10)) 

East_rules=pd.DataFrame(rules)



#South

frq_items = apriori(basket_South, min_support = 0.001, use_colnames = True) 

rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 

rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

print(rules.head()) 

South_rules=pd.DataFrame(rules)



#Central

frq_items = apriori(basket_Central, min_support = 0.001, use_colnames = True) 

rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 

rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

print(rules.head()) 

Central_rules=pd.DataFrame(rules)



#Here Empty DataFrame signifies that none of the Rules in UK satisfy the levels mentioned for 

#Support & Lift in above freq items sets

 
# you can directly put this function into the code to form a network draw_graph()
#
def draw_graph(rules, rules_to_show):

  import matplotlib.pyplot as plt

  import networkx as nx  

  G1 = nx.DiGraph()

   

  color_map=[]

  N = 50

  colors = np.random.rand(N)    

  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   

   # you can change the number of nodes as per requirement more the number of nodes cleaner the diagram
   # here for 7 items we have taken 10-11 nodes

   

  for i in range (rules_to_show):      

    G1.add_nodes_from(["R"+str(i)])

    

     

    for a in rules.iloc[i]['antecedents']:

                

        G1.add_nodes_from([a])

        

        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

       

    for c in rules.iloc[i]['consequents']:

             

            G1.add_nodes_from([c])

            

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)

 

  for node in G1:

       found_a_string = False

       for item in strs: 

           if node==item:

                found_a_string = True

       if found_a_string:

            color_map.append('red')

       else:

            color_map.append('green')       

 

 

   

  edges = G1.edges()

  colors = [G1[u][v]['color'] for u,v in edges]

  weights = [G1[u][v]['weight'] for u,v in edges]

 

  pos = nx.spring_layout(G1, k=16, scale=1)

  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            

   

  for p in pos:  # raise text positions

           pos[p][1] += 0.07

  nx.draw_networkx_labels(G1, pos)

  plt.show()

 

## give the input to the above function here    

draw_graph (West_rules, 4)

draw_graph (East_rules, 4)

draw_graph (Central_rules, 4)

draw_graph (South_rules, 4)
# Code for custom code recipe mba-analysis (imported from a Python recipe)

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import *
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from mlxtend.frequent_patterns import apriori, association_rules

print("Starting MBA process")

project_key = dataiku.get_custom_variables()["projectKey"]
client = dataiku.api_client()
project = client.get_project(project_key)


# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
input_names = get_input_names_for_role('input_dataset')

# The dataset objects themselves can then be created like this:
input_datasets = [dataiku.Dataset(name) for name in input_names]


# For outputs, the process is the same:
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
max_itemsets_size = get_recipe_config()['max-itemset-size']
itemsets_min_support = get_recipe_config()['itemset-min-support']
confidence_threshold = get_recipe_config()['confidence-threshold']
item_list=get_recipe_config()['items']
transaction_col=get_recipe_config()['transaction_id']

# For optional parameters, you should provide a default value in case the parameter is not present:
scope = get_recipe_config().get('scope', None)

print("Received: max itemset {}, min support {}, confidence {}, scope {} ".format(max_itemsets_size, itemsets_min_support, confidence_threshold, scope ))

# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])
pd.set_option('display.max_colwidth', 30)
pd.set_option('use_inf_as_na', True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs :

transactions_preprocessed_df = input_datasets[0].get_dataframe().head(200000)

# Recipe outputs :

out_association_rules = output_datasets[0]
df_association_rules = pd.DataFrame()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if scope:
    print("Scope found - processing groups")

    association_rules_comb = pd.DataFrame()

    # scope_unique=transactions_preprocessed_df.unique()
    scope_granular_combinations = np.unique(transactions_preprocessed_df[scope])
    
    n_scope_granular_combinations = len(scope_granular_combinations)

    print("{} scope granular combination found : '{}' ".format(n_scope_granular_combinations, scope_granular_combinations))

    for loop_index, scope_granular_combination in enumerate(scope_granular_combinations):
        # scope_granular_combination_str = "_".join(scope_granular_combination)
        granular_df = transactions_preprocessed_df[transactions_preprocessed_df[scope]==scope_granular_combination]
        granular_df = granular_df[[transaction_col,item_list]]
        
        print("Computing association rules on scope_granular_combination : {} (n°{}/{})".format(scope_granular_combination,
                                                                                                loop_index+1,
                                                                                                n_scope_granular_combinations))
 
        
        onehot = granular_df.pivot_table(index=transaction_col, columns=item_list, aggfunc=len, fill_value=0)
        onehot = onehot>0
                                   
                                  
        #basket = (granular_df.groupby(['transaction_id', 'item']).size().unstack().reset_index().fillna(0).set_index('transaction_id'))
        #print("basket {}".format(basket.head(5)))                      
                                   
        #def encode_units(x):
        #    if x <= 0:
        #        return 0
        #    if x >= 1:
        #        return 1
        #basket_sets = basket.applymap(encode_units)
                                           
        # compute frequent items using the Apriori algorithm
        frequent_itemsets = apriori(onehot, min_support = itemsets_min_support, max_len=max_itemsets_size, use_colnames=True)
        # compute all association rules for frequent_itemsets
        
        if len(frequent_itemsets)>0:
            df_association_rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=confidence_threshold)
            df_association_rules.head()
            
            n_association_rules = len(df_association_rules)
            df_association_rules["rule_id"] = ["rule_{}_".format(scope_granular_combination)+str(id_) for id_ in range(n_association_rules)]
            df_association_rules['antecedents']=df_association_rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
            df_association_rules['consequents']=df_association_rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
            
            print("testing {}".format(df_association_rules.head(5)))

            print("Writing association rules data ...")
            
            association_rules_comb=association_rules_comb.append(df_association_rules)
            
    out_association_rules.write_with_schema(association_rules_comb.fillna(9999))
    # print("Writing association rules summary ...")

else:
    print("Scope NOT found - processing groups")

    granular_combination = None

    transactions_preprocessed_df = transactions_preprocessed_df[[transaction_col,item_list]]
        
    onehot = transactions_preprocessed_df.pivot_table(index=transaction_col, columns=item_list, aggfunc=len, fill_value=0)
    onehot = onehot>0

    # compute frequent items using the Apriori algorithm
    frequent_itemsets = apriori(onehot, min_support = itemsets_min_support, max_len=max_itemsets_size, use_colnames=True)
    # compute all association rules for frequent_itemsets

    if len(frequent_itemsets)>0:
        df_association_rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=confidence_threshold)
        df_association_rules.head()

        n_association_rules = len(df_association_rules)
        df_association_rules["rule_id"] = ["rule_"+str(id_) for id_ in range(n_association_rules)]
        df_association_rules['antecedents']=df_association_rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        df_association_rules['consequents']=df_association_rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

        print("testing {}".format(df_association_rules.head(5)))

        print("Writing association rules data ...")

    else:
        print("No rules found ...")
        
    out_association_rules.write_with_schema(df_association_rules.fillna(9999))
        
    
        
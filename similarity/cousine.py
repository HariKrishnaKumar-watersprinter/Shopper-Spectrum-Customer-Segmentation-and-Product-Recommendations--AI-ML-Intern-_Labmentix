import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
model_path = os.path.join(os.getcwd(), "online_retail.csv") 
df= pd.read_csv(model_path)
df.dropna()
def similarity():
    cust_prod = df.groupby(['CustomerID','Description'])['Quantity'].sum().unstack(fill_value=0)

    # Keep only products bought by >=5 customers (popularity filter)   
    popular = cust_prod.columns[cust_prod.gt(0).sum() >= 5]
    cust_prod = cust_prod[popular]

    # Item-based similarity
    item_sim = pd.DataFrame(cosine_similarity(cust_prod.T),
                        index=cust_prod.columns, columns=cust_prod.columns)
    return item_sim
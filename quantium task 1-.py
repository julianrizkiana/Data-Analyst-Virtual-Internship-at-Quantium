#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().system('pip install missingno')


# In[5]:


#data analysis and wrangling
import pandas as pd 
import numpy as np


#visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import missingno

#dates
import datetime
from matplotlib.dates import DateFormatter

#text analysis
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist as fdi
import re

#statistical analysis
from scipy.stats import ttest_ind

#warnings
import warnings
warnings.filterwarnings('ignore')


# In[6]:


cd = pd.read_csv("/Users/julia/Desktop/data science/forage/quantium/QVI_purchase_behaviour.csv")
td = pd.read_excel("/Users/julia/Desktop/data science/forage/quantium/QVI_transaction_data.xlsx")


# # Transaction data

# In[7]:


td.head()


# In[8]:


td.shape


# In[9]:


td['TXN_ID'].nunique()


# In[10]:


#TXN_ID is not unique to each row. This is because there can be sales of chips with different brands in a single transaction (it less than transaction data row)
#look for duplicated 'TXN_ID' 
td[td.duplicated(['TXN_ID'])].head()


# In[11]:


#lets take a look at 'TXN_ID' 48887
td.loc[td['TXN_ID'] == 48887, :]


# In[12]:


td.info()


# In[13]:


#plot graph of missing values for 'td''
missingno.matrix(td)

#based on the graph theres no missing numbers in td (no white line)


# In[14]:


list(td.columns)


# In[15]:


td['DATE'].head()


# In[16]:


#date not in the right format
#convert excel integer into yyyy-mm-dd format
def xlseriesdate_to_datetime(xlserialdate):
    excel_anchor = datetime.datetime(1900, 1, 1)
    if(xlserialdate < 60):
        delta_in_days = datetime.timedelta(days = (xlserialdate - 1))
    else:
        delta_in_days = datetime.timedelta(days = (xlserialdate - 2))
    converted_date = excel_anchor + delta_in_days
    return converted_date


# In[17]:


#apply function to 'DATE'feature in 'td' dataset
td['DATE'] = td['DATE'].apply(xlseriesdate_to_datetime)


# In[18]:


#check new data format
td['DATE'].head()


# In[19]:


td['PROD_NAME'].head()


# In[20]:


#Extract weight out of "PROD_Name" and make Extract weight as a new column
td['PACK_SIZE'] = td['PROD_NAME'].str.extract("(\d+)")
td['PACK_SIZE'] = pd.to_numeric(td['PACK_SIZE'])
td.head()


# In[21]:


#'PROD_NAME' text cleaning
def clean_text(text):
    text = re.sub('[&/]', ' ', text) # remove special characters '&' and '/'
    text = re.sub('\d\w*', ' ', text) # remove product weights
    return text

# Apply text cleaning function to PROD_NAME column
td['PROD_NAME'] = td['PROD_NAME'].apply(clean_text)


# In[22]:


td['PROD_NAME'].head()


# In[23]:


# Drop rows with salsa word in PROD_NAME (solution draft said the data has salsa word in it)


td[td["PROD_NAME"].str.contains("salsa")==False] 


# In[24]:


# check for possible outliers
td["PROD_QTY"].value_counts()


# In[25]:


#2 total sales with 200 chips products quantity looks odd
#explore further 
td.loc[td['PROD_QTY'] == 200, :] 


# In[26]:


#both the transaction with 200 chips products quantity have been made by the same person at the same store!
#Let's see all the transactions this person has made by tracking his loyalty card number

td.loc[td['LYLTY_CARD_NBR'] == 226000, :]


# In[27]:


#It looks like this customer has only had the two transactions over the year and is not an ordinary retail customer. 
#The customer might be buying chips for commercial purposes instead. We will remove this loyalty card number from further analysis.
# Filter out the customer based on the loyalty card number

td.drop(td.index[td['LYLTY_CARD_NBR'] == 226000], inplace = True)


# In[28]:


# Re-examine transaction data to make sure it has been dropped 

td.loc[td['LYLTY_CARD_NBR'] == 226000]


# In[29]:


# let's look if there are any obvious data issues such as missing data
td.nunique()


# In[30]:


# Now let's examine the number of transactions over time to see if there are any obvious data issues e.g. missing data

td.nunique()


# In[31]:


td = td.sort_values(by=['DATE'], inplace=False, ascending=False)


# In[32]:


td


# In[33]:


#There's only 364 rows, meaning only 364 dates which indicates a missing date
#Let's create a sequence of dates from 1 Jul 2018 to 30 Jun 2019 and use this to create a chart of number of transactions over time to find the missing date.


# In[34]:


# Create a new dataframe which contains the total sale for each date
a = pd.pivot_table(td, values = 'TOT_SALES', index = 'DATE', aggfunc = 'sum')
a


# In[35]:


b = pd.DataFrame(index = pd.date_range(start = '2018-07-01', end = '2019-06-30'))
b['TOT_SALES'] = 0
len(b)


# In[36]:


c = a + b
c.fillna(0, inplace = True)
c


# In[37]:


c.index.name = 'Date'
c.rename(columns = {'TOT_SALES': 'Total Sales'}, inplace = True)
c 


# In[38]:


timeline = c.index
graph = c['Total Sales']

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(timeline, graph)

date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
plt.title('Total Sales from July 2018 to June 2019')
plt.xlabel('Time')
plt.ylabel('Total Sales')

plt.show()


# In[39]:


#We can see that sales spike up during the December month and zero sale on Christmas Day.
# Confirm the date where sales count equals to zero

c[c['Total Sales'] == 0]


# In[40]:


# Let's zoom in at the December month only
c_december = c[(c.index < "2019-01-01") & (c.index > "2018-11-30")]
c_december.head()


# In[41]:


plt.figure(figsize = (15, 5))
plt.plot(c_december)
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Total Sales in December')


# In[42]:


#Now that we are satisfied that the data no longer has outliers
#we can move on to creating other features such as brand of chips or pack size from PROD_NAME

td['PACK_SIZE'].unique()


# In[43]:


#The largest size is 380g and the smallest size is 70g - seems sensible!
## Let's plot a histogram of PACK_SIZE since we know that it is a categorical variable and not a continuous variable even though it is numeric.
td['PACK_SIZE'].hist()


# In[44]:


# create brands use the first word in PROD_NAME to work out the brand name
#create a new column under 'transaction data' dataset called 'Brand''
part = td['PROD_NAME'].str.partition()
td['BRAND'] = part[0]
td.head()


# In[45]:


td['BRAND'].unique()


# In[46]:


## Clean brand names
td['BRAND'].replace('Infzns', 'Infuzions', inplace = True)
td['BRAND'].replace('Ww', 'Woolworths', inplace = True)
td['BRAND'].replace('Ncc', 'Natural', inplace = True)
td['BRAND'].replace('Ccs', 'CCS', inplace = True)
td['BRAND'].replace('Smith', 'Smiths', inplace = True)
td['BRAND'].replace(['Grain', 'Grnwves'], 'Grainwaves', inplace = True)
td['BRAND'].replace('Dorito', 'Doritos', inplace = True)
td['BRAND'].replace(['Red', 'Rrd'], 'Red Rock Deli', inplace = True)
td['BRAND'].replace('Snbts', 'Sunbites', inplace = True)

td['BRAND'].unique()


# In[47]:


# Which brand had the most sales?
td.groupby('BRAND').TOT_SALES.sum().sort_values(ascending = False)


# # Customer  Data (cd)

# In[48]:


cd


# In[49]:


# Missing values in customerData

missingno.matrix(cd)


# In[50]:


cd['LYLTY_CARD_NBR'].nunique()


# In[51]:


cd['LIFESTAGE'].nunique()


# In[52]:


cd['LIFESTAGE'].unique()


# In[53]:


cd['LIFESTAGE'].value_counts().sort_values(ascending=False)


# In[54]:


sns.countplot(y = cd['LIFESTAGE'], order = cd['LIFESTAGE'].value_counts().index)


# In[55]:


# Value counts for each premium customer category

cd['PREMIUM_CUSTOMER'].value_counts().sort_values(ascending = False)


# In[56]:


sns.countplot(y = cd['PREMIUM_CUSTOMER'], order = cd['PREMIUM_CUSTOMER'].value_counts().index)


# # Merge transactiondata and customerdata together

# In[57]:


combine_data = pd.merge(td, cd)


# In[58]:


# Check for null values

combine_data.isnull().sum()


# In[59]:


# Check for n/a values

combine_data.isna().sum()


# In[60]:


combine_data


# # Data analysis on customer segments 

# Now that the data is ready for analysis, we can define some metrics of interest to
# the client:
# - Who spends the most on chips (total sales), describing customers by lifestage and how premium their general purchasing behaviour is
# - How many customers are in each segment
# - How many chips are bought per customer by segment
# - What's the average chip price by customer segment
# 
# We could also ask our data team for more information. Examples are:
# - The customer's total spend over the period and total spend for each transaction to understand what proportion of their grocery spend is on chips
# - Proportion of customers in each customer segment overall to compare against the mix of customers who purchase chips

# In[61]:


# Total sales by PREMIUM_CUSTOMER and LIFESTAGE
sales = pd.DataFrame(combine_data.groupby(['PREMIUM_CUSTOMER','LIFESTAGE']).TOT_SALES.sum())
sales.rename(columns={'TOT_SALES':'Total Sales'},inplace=True)
sales.sort_values(by='Total Sales',ascending=False).head()


# In[62]:


#visualize
salesPlot = pd.DataFrame(combine_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum())
salesPlot.unstack().plot(kind = 'bar', stacked = True, figsize = (12, 7), title = 'Total Sales by Customer Segment')
plt.ylabel('Total Sales')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)


# Top 3 sales come from budget older families, mainstream young singles/couples and mainstream retirees.

# In[63]:


# Number of customers by PREMIUM_CUSTOMER and LIFESTAGE

customers = pd.DataFrame(combine_data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique())
customers.rename(columns = {'LYLTY_CARD_NBR': 'Number of Customers'}, inplace = True)
customers.sort_values(by = 'Number of Customers', ascending = False).head()


# In[64]:


# Visualise

customersPlot = pd.DataFrame(combine_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.nunique())
customersPlot.unstack().plot(kind = 'bar', stacked = True, figsize = (12, 7), title = 'Number of Customers by Customer Segment')
plt.ylabel('Number of Customers')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)


# There are more mainstream young singles/couples and retirees. This contributes to more chips sales in these categories however this is not the major driver for the budget older families segment.

# In[65]:


# Average units per customer by PREMIUM_CUSTOMER and LIFESTAGE

avg_units = combine_data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum() / combine_data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique()
avg_units = pd.DataFrame(avg_units, columns = {'Average Unit per Customer'})
avg_units.sort_values(by = 'Average Unit per Customer', ascending = False).head()


# In[66]:


# Visualise 

avgUnitsPlot = pd.DataFrame(combine_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum() / combine_data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique())
avgUnitsPlot.unstack().plot(kind = 'bar', figsize = (12, 7), title = 'Average Unit by Customer Segment')
plt.ylabel('Average Number of Units')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)


# Older families and young families buy more chips per customer

# In[67]:


# Average price per unit by PREMIUM_CUSTOMER and LIFESTAGE

avg_price = combine_data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).TOT_SALES.sum() / combine_data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum()
avg_price = pd.DataFrame(avg_price, columns = {'Price per Unit'})
avg_price.sort_values(by = 'Price per Unit', ascending = False).head()


# In[68]:


# Visualise 

avgPricePlot = pd.DataFrame(combine_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum() / combine_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum())
avgPricePlot.unstack().plot(kind = 'bar', figsize = (12, 7), title = 'Average Price by Customer Segment', ylim = (0, 6))
plt.ylabel('Average Price')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)


# Mainstream midage and young singles and couples are more willing to pay more per packet of chips compared to their budget and premium counterparts. This may be due to premium shoppers being more likely to buy healthy snacks and when they do buy chips, it is mainly for entertainment purposes rather than their own consumption. This is also supported by there being fewer premium midage and young singles and couples buying chips compared to their mainstream counterparts.

# In[69]:


# Perform an independent t-test between mainstream vs non-mainstream midage and young singles/couples to test this difference

# Create a new dataframe pricePerUnit
pricePerUnit = combine_data

# Create a new column under pricePerUnit called PRICE
pricePerUnit['PRICE'] = pricePerUnit['TOT_SALES'] / pricePerUnit['PROD_QTY']

# Let's have a look
pricePerUnit.head()


# In[70]:


# Let's group our data into mainstream and non-mainstream

mainstream = pricePerUnit.loc[(pricePerUnit['PREMIUM_CUSTOMER'] == 'Mainstream') & ( (pricePerUnit['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') | (pricePerUnit['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES') ), 'PRICE']
nonMainstream = pricePerUnit.loc[(pricePerUnit['PREMIUM_CUSTOMER'] != 'Mainstream') & ( (pricePerUnit['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') | (pricePerUnit['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES') ), 'PRICE']


# In[71]:


# Compare histograms of mainstream and non-mainstream customers

plt.figure(figsize = (10, 5))
plt.hist(mainstream, label = 'Mainstream')
plt.hist(nonMainstream, label = 'Premium & Budget')
plt.legend()
plt.xlabel('Price per Unit')


# In[72]:


print("Mainstream average price per unit: ${:.2f}".format(np.mean(mainstream)))
print("Non-mainstream average price per unit: ${:.2f}".format(np.mean(nonMainstream)))
if np.mean(mainstream) > np.mean(nonMainstream):
    print("Mainstream customers have higher average price per unit. ")
else:
    print("Non-mainstream customers have a higher average price per unit. ")


# In[73]:


# Perform t-test 

ttest_ind(mainstream, nonMainstream)


# Mainstream customers have higher average price per unit than that of non-mainstream customers.
# 
# We have found quite a few interesting insights that we can dive deeper into. For example, we might want to target customers segments that contribute the most to sales to retain them to further increase sales. Let's examine mainstream young singles/couples against the rest of the cutomer segments to see if they prefer any particular brand of chips.

# In[74]:


target = combine_data.loc[(combine_data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & (combine_data['PREMIUM_CUSTOMER'] == 'Mainstream'), :]
nonTarget = combine_data.loc[(combine_data['LIFESTAGE'] != 'YOUNG SINGLES/COUPLES' ) & (combine_data['PREMIUM_CUSTOMER'] != 'Mainstream'), :]
target.head()


# # Affinity to brand
# 

# In[75]:


# Target Segment
targetBrand = target.loc[:, ['BRAND', 'PROD_QTY']]
targetSum = targetBrand['PROD_QTY'].sum()
targetBrand['Target Brand Affinity'] = targetBrand['PROD_QTY'] / targetSum
targetBrand = pd.DataFrame(targetBrand.groupby('BRAND')['Target Brand Affinity'].sum())

# Non-target segment
nonTargetBrand = nonTarget.loc[:, ['BRAND', 'PROD_QTY']]
nonTargetSum = nonTargetBrand['PROD_QTY'].sum()
nonTargetBrand['Non-Target Brand Affinity'] = nonTargetBrand['PROD_QTY'] / nonTargetSum
nonTargetBrand = pd.DataFrame(nonTargetBrand.groupby('BRAND')['Non-Target Brand Affinity'].sum())


# In[76]:


# Merge the two dataframes together

brand_proportions = pd.merge(targetBrand, nonTargetBrand, left_index = True, right_index = True)
brand_proportions.head()


# In[132]:


brand_proportions['Affinity to Brand'] = brand_proportions['Target Brand Affinity'] / brand_proportions['Non-Target Brand Affinity']
brand_proportions.sort_values(by = 'Affinity to Brand', ascending = False, inplace= True)
brand_proportions


# Mainstream young singles/couples are more likely to purchase Tyrrells chips compared to other brands.

# # Affinity to pack size

# In[79]:


# Target segment 
targetSize = target.loc[:, ['PACK_SIZE', 'PROD_QTY']]
targetSum = targetSize['PROD_QTY'].sum()
targetSize['Target Pack Affinity'] = targetSize['PROD_QTY'] / targetSum
targetSize = pd.DataFrame(targetSize.groupby('PACK_SIZE')['Target Pack Affinity'].sum())

# Non-target segment
nonTargetSize = nonTarget.loc[:, ['PACK_SIZE', 'PROD_QTY']]
nonTargetSum = nonTargetSize['PROD_QTY'].sum()
nonTargetSize['Non-Target Pack Affinity'] = nonTargetSize['PROD_QTY'] / nonTargetSum
nonTargetSize = pd.DataFrame(nonTargetSize.groupby('PACK_SIZE')['Non-Target Pack Affinity'].sum())


# In[80]:


# Merge the two dataframes together

pack_proportions = pd.merge(targetSize, nonTargetSize, left_index = True, right_index = True)
pack_proportions.head()


# In[157]:


pack_proportions['Affinity to Pack'] = pack_proportions['Target Pack Affinity'] / pack_proportions['Non-Target Pack Affinity']
pack_proportions.sort_values(by = 'Affinity to Pack', ascending = False, inplace=True)
pack_proportions


# It looks like mainstream singles/couples are more likely to purchase a 270g pack size compared to other pack sizes.

# In[82]:


# Which brand offers 270g pack size?

combine_data.loc[combineData['PACK_SIZE'] == 270, :].head(10)


# In[ ]:


# Is Twisties the only brand who sells 270g pack size?

combine_data.loc[combineData['PACK_SIZE'] == 270, 'BRAND'].unique()


# Twisties is the only brand that offers 270g pack size.

# # Conclusion
# 

# Sales are highest for (Budget, OLDER FAMILIES), (Mainstream, YOUNG SINGLES/COUPLES) and (Mainstream, RETIREES)
# We found that (Mainstream, YOUNG SINGLES/COUPLES) and (Mainstream, RETIREES) are mainly due to the fact that there are more customers in these segments
# (Mainstream, YOUNG SINGLES/COUPLES) are more likely to pay more per packet of chips than their premium and budget counterparts
# They are also more likely to purchase 'Tyrrells' and '270g' pack sizes than the rest of the population

# In[ ]:


combine_data.to_csv("/Users/julia/Desktop/data science/forage/quantium/task 1_combinedata.csv")


# In[ ]:





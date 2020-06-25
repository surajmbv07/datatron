import pandas as pd
from simple_salesforce import Salesforce,SalesforceLogin,SFType
import datetime
pd.set_option("display.max_columns",400)
pd.set_option("display.max_colwidth",200)
username = 'xxxxx@xxxxxx.com'
password = '*xxxxxxxxx'
security_token = 'xxxxxxxxxxx' # this token is sent by salesforce to developer's mail account
domain = 'login'

#%%
""""""""""""""""""""""""""""""""" Connecting to SalesForce Account """""""""""""""""""""""""""""""""""
#Method 1 

sf = Salesforce(username = username, password=password, security_token=security_token,domain=domain)

#Method 2 using login function (more secure)

session_id , instance = SalesforceLogin(username = username, password=password, security_token=security_token,domain=domain)
sf2 = Salesforce(instance=instance , session_id=session_id)

# Attributes of SalesForce login, instance
for element in dir(sf2):
    if not element.startswith('_'):
        if isinstance(getattr(sf2,element),str):
            print('Name:{0};Value : {1}'.format(element,getattr(sf2, element)))

# Properties of MetaData        
metadata_org = sf2.describe()
metadata_org.keys()
metadata_org['encoding']
metadata_org['maxBatchSize']
metadata_org['sobjects']

df_sobjects = pd.DataFrame(metadata_org['sobjects'])
df_sobjects.to_csv('metadata_info.csv',index=False) #this file contains objects of SaleForce along with its properties


####  how to extract salesforce object meta data information

# Method 1      
account = sf2.account  
account_metadata = account.describe()         
df_account_metadata = pd.DataFrame(account_metadata.get('fields'))   
df_account_metadata.to_csv('account_metadata_info.csv',index=False)

# Method2 if we know the object name for example project__C ; we dont have this record so it will not be found
projectc = SFType('Project__c', session_id, instance)
project_metadata = projectc.describe()
df_project_metdata = pd.DataFrame(project_metadata.get('fields'))


#%%
"""""""""""""""""""""""""""" Fetcing records from SalesForce to Python """""""""""""""""""""""""""""""""

# Below SQL query is fetched from Saleforce Developer Console

value = ['Energy','Banking']
querySOQL ="SELECT Id ,Type, Industry FROM Account where Industry in ('{0}')".format("','".join(value)) 

#'query()' & 'query_more()'  does not return archive records
#'query_all()' will return all archive and non archive records

recordaccounts = sf2.query(querySOQL) #this results in records which have less then 200, if record count is more then we need to create batches
#recordaccounts.keys()

# To get records in batches ## Query Records Method ##
recordaccounts2 = sf2.query(querySOQL)
lstrecords = recordaccounts2.get('records')
nextRecordsUrl = recordaccounts2.get('nextRecordsUrl') #this key appears when recordaccounts2.get('done') is False

while not recordaccounts2.get('done'):
    recordaccounts2 = sf2.query_more(nextRecordsUrl,identifier_is_url=True)
    lstrecords.extend(recordaccounts2.get('records'))
    nextRecordsUrl = recordaccounts2.get('nextRecordsUrl')

# Therefore lstrecords will give older records 
df_records = pd.DataFrame(lstrecords)

# Accounts and Oppurtunity query records 

querySOQL ="SELECT Id, Name,StageName,Account.Name,Account.Type ,Account.Industry FROM Opportunity" 

recordaccounts2 = sf2.query(querySOQL)
lstrecords = recordaccounts2.get('records')
nextRecordsUrl = recordaccounts2.get('nextRecordsUrl') #this key appears when recordaccounts2.get('done') is False

while not recordaccounts2.get('done'):
    recordaccounts2 = sf2.query_more(nextRecordsUrl,identifier_is_url=True)
    lstrecords.extend(recordaccounts2.get('records'))
    nextRecordsUrl = recordaccounts2.get('nextRecordsUrl')

df_records = pd.DataFrame(lstrecords) #this contains many accounts and each account has older records so we need to extract them as well 

dfAccounts = df_records['Account'].apply(pd.Series).drop(labels='attributes',axis=1,inplace=False)
dfAccounts.columns= ['Accounts.{0}'.format(name) for name in dfAccounts.columns] #extracted account names from oppurtunity object

df_records.drop(labels = ['Account','attributes'],axis=1,inplace=True)

# Concatenating extracted oppurtunity names and accounts names
dfOpprAcc = pd.concat([df_records, dfAccounts],axis =1)
dfOpprAcc.to_csv('Account_oppurtunity.csv',index=False)

### dfopprAcc is same table as querySOQL query ###

#%%
""""""""""""""""""""""" Creating Updating, Deleting records in Salesforce from Python """""""""""""""""""""""""""


#Example-1 To create records in objects(ex: accounts) in saleforce
accounts__c = SFType('Account', session_id=session_id, sf_instance=instance)

present = datetime.datetime.now()

data = {'Name': 'Account_Fightclub',
        'CustomerPriority__c': 'High',
        'Active__c': 'Yes',
        'SLAExpirationDate__c':(present + datetime.timedelta(days=45)).isoformat()+ 'Z'} # isformat gives the datetime format; timedelta is the difference & 
                                                                                            #in saleforce format u need to add 'Z'

response = accounts__c.create(data) #OrderedDict([('id', 'xxxxxxxxxxxx'),('xxxxxxxxxxxxx'), ('success', True), ('errors', [])])
# the id can be used in salesforce.com/id to view the record

"""
Parent Child Relationship Record creation (Creating 5 accounts and 5 oppurtunities and linking them)

"""

oppurtunity__c =SFType('Opportunity', session_id=session_id, sf_instance=instance)
account = SFType('Account', session_id=session_id, sf_instance=instance)

for i in range(1,6):
    data_account = {'Name': 'Retail Account ' + str(i), 'Type' : 'Start' }
    response_account = account.create(data_account)
    accountID = response_account.get('id')
    
    data_oppurtunity = {'Name': 'Oppurtunity ' + str(i),'StageName' : 'Contact Hyderabad' , 'AccountId': accountID, 'CloseDate': present.isoformat()+'Z' } #we need to know the lookup fields which can be serached in google
    response_oppurtunity = oppurtunity__c.create(data_oppurtunity)
    contactID = response_oppurtunity.get('id')
    
    print('Records created')
    print('-'.center(20,'-'))
    print('Conatct ID: {0}'.format(accountID))
    print('Project ID: {0}'.format(contactID))


"""
Update the existing records
"""

update_data = {}
update_data['AccountNumber'] =1293012931
update_data['CustomerPriority__c'] = 'Medium'

accounts__c.update(response.get('id'),update_data)

"""
Delete Records
"""
account.delete('0xxxxxxxxxxxxx') # xxxxx is id

"""
Upsert records inserting new fields in the exiting records without using account id and using existing field names
"""
update_data['Name'] = 'Account_Fightclub'
externalid = 'AnnualRevenue/{0}'.format(10000)
update_data['AccountNumber'] = update_data['AccountNumber'] + 20000000
response = accounts__c.upsert(data=update_data)

"""
Using Bulk API to create records in Salesforce
"""
## Bulk query can be used for retreving vast records such as 50000
output = sf2.bulk.Account.query("SELECT Id, Name from Account LIMIT 5")
pd.DataFrame(output)

## Loading csv into salesforce object

load = pd.read_csv('S:/SURAJ_STUFF/Spyder/Saved Python files/Account_demo.csv')

bulk_data= []
for i in load.itertuples():
    d=i._asdict()
    del d['Index']
    bulk_data.append(d)

## bulk_data contains records of account name and account number

## Inserting bulk_data into salesforce Account object
sf2.bulk.Account.insert(bulk_data)
    
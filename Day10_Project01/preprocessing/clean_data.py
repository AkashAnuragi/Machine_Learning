import pandas as pd
import joblib
from connection import DBConnect
 

# Extracting Data From Source
cus = pd.read_excel("../dataset/Ecomm.xlsx",sheet_name="CustomerDetail"  )  
order = pd.read_excel("../dataset/Ecomm.xlsx", sheet_name='Jan')

# Feature Engineering
cus = cus.drop(columns=['CustomerID','DateInText','Month','MonthName','Gender','Gender.1'])
cus = cus.rename(columns={'CustomerID.1':'CustomerID','Gender.2':'Gender'})

# Removing Duplicate Data
cus = cus.drop_duplicates()
order = order.drop_duplicates()


# Removing null values
cus = cus.dropna()
order = order.dropna()

# Saving Cleaned Data
joblib.dump(cus,"../dataset/customer.csv")
joblib.dump(order,"../dataset/order.csv")  

# Building SQL Connection
conn = DBConnect.getConnection()
cur = conn.cursor()


for i in range(len(cus)):
    val = list(cus.iloc[i].reset_index()[i])
    data = [int(val[0]),str(val[1]),val[2],val[3],val[4],val[5]]
    sql = 'insert into customer values(%s,%s,%s,%s,%s,%s)'
    cur.execute(sql,data)


for i in range(len(order)):
    val = list(order.iloc[i].reset_index()[i])
    data = [int(val[0]),int(val[1]),float(val[2]),float(val[3]),float(val[4]),val[5],val[6],val[7]]
    sql = 'insert into orders value(%s,%s,%s,%s,%s,%s,%s,%s)'
    cur.execute(sql,data)

conn.commit()
cur.close()
conn.close()

print("Data Loaded to Database Sucessfully!")
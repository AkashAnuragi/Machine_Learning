from preprocessing.connection import DBConnect 
import pandas as pd

def dataset():
    conn = DBConnect.getConnection()
    cur = conn.cursor()

    sql = "SELECT * FROM orders right JOIN customer using (cid)"
    cur.execute(sql)

    data = cur.fetchall()

    cur.close()
    conn.close()

    names=['CID','OrderDate','Cname','State','City','Gender','OID','Amt','Profit','Qty','Product Category','Product Sub-Category','PaymentMode']
    df = pd.DataFrame(data,columns=names)
    return df
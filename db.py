import psycopg2
connected=psycopg2.connect(
    host='localhost',
    database='fraud_db',
    user='postgres',
    password='postgre123'
)
cursor=connected.cursor()
#to execute sql commands cursor object is used
print("Inserted Successfully")
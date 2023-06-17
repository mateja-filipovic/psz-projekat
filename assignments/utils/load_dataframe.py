import pandas as pd
from sqlalchemy import create_engine

host = 'localhost'
user = 'root'
password = 'Mateja123'
database_name = 'data'
table_name = 'data_mining_results'


def load_dataframe_from_mysql_database():
    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database_name}')
    data = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)
    return data.drop_duplicates(subset='ad_url').reset_index()


# # the old way, creates a warning
# def load_dataframe_from_mysql_database():
#     conn = mysql.connector.connect(
#         host=host,
#         user=user,
#         password=password,
#         database=database_name
#     )
#
#     dataframe = pd.read_sql(f'SELECT * FROM {table_name}', conn)
#     conn.close()
#
#     return dataframe


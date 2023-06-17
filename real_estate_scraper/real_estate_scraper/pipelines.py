import mysql
from mysql.connector import Error

from real_estate_scraper.items import RealEstateScraperItem

host = 'localhost'
user = 'root'
password = 'Mateja123'
database_name = 'data'
table_name = 'data_mining_results_2'


class RealEstateScraperPipeline(object):

    def __init__(self):
        self.conn = None

        self.host = host
        self.user = user
        self.password = password
        self.database = database_name

    def create_table(self):
        try:
            cursor = self.conn.cursor()

            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
              ad_url TEXT,
              city TEXT,
              construction_type TEXT,
              floor TEXT,
              has_lift TINYINT(1) DEFAULT NULL,
              has_parking TINYINT(1) DEFAULT NULL,
              has_terrace TINYINT(1) DEFAULT NULL,
              heating_type TEXT,
              house_floors TEXT,
              is_registered TINYINT(1) DEFAULT NULL,
              location TEXT,
              lot_surface_area DOUBLE DEFAULT NULL,
              microlocation TEXT,
              number_of_rooms TEXT,
              price DOUBLE DEFAULT NULL,
              real_estate_id BIGINT DEFAULT NULL,
              real_estate_surface_area DOUBLE DEFAULT NULL,
              real_estate_type TEXT,
              street TEXT,
              title TEXT,
              total_floors DOUBLE DEFAULT NULL,
              transaction_type TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
            cursor.execute(create_table_query)
            self.conn.commit()

            cursor.close()
        except Error as e:
            print(f"Fatal error occurred while creating a MySql table: {e}")

    def open_spider(self, spider):
        self.conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        self.create_table()

    def close_spider(self, spider):
        self.conn.close()

    def process_item(self, item, spider):
        try:
            cursor = self.conn.cursor()

            # Example of inserting scraped data into a MySQL table
            sql = f"INSERT INTO {table_name} (ad_url, city, construction_type, floor, has_lift, has_parking, " \
                  "has_terrace, heating_type, house_floors, is_registered, location, lot_surface_area, " \
                  "microlocation, number_of_rooms, price, real_estate_id, real_estate_surface_area, " \
                  "real_estate_type, street, title, total_floors, transaction_type) " \
                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            values = (
                item['ad_url'], item['city'], item['construction_type'], item['floor'], item['has_lift'],
                item['has_parking'], item['has_terrace'], item['heating_type'], item['house_floors'],
                item['is_registered'], item['location'], item['lot_surface_area'], item['microlocation'],
                item['number_of_rooms'], item['price'], item['real_estate_id'], item['real_estate_surface_area'],
                item['real_estate_type'], item['street'], item['title'], item['total_floors'], item['transaction_type']
            )
            cursor.execute(sql, values)
            self.conn.commit()

            cursor.close()
        except Error as e:
            print(f"Fatal error occurred while inserting into {table_name}: {e}")

        return item


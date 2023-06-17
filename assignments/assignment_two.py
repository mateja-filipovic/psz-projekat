import re
import pandas as pd

from assignments.utils.load_dataframe import load_dataframe_from_mysql_database


# utilities
def extract_numeric_value(string):
    pattern = r'(\d+(\.\d+)?|5\+)'
    match = re.search(pattern, string)
    if match:
        value = match.group()
        return 5 if value == '5+' else float(value)
    else:
        return 0

# end utilities


def get_number_of_real_estate_for_sale():
    for_sale = scraper_entries[scraper_entries['transaction_type'] == 'prodaja']
    return for_sale.shape[0]


def get_number_of_real_estate_for_rent():
    for_rent = scraper_entries[scraper_entries['transaction_type'] == 'izdavanje']
    return for_rent.shape[0]


def get_number_of_sales_per_city():
    for_sale = scraper_entries[scraper_entries['transaction_type'] == 'prodaja']
    return for_sale['city'].value_counts()


def get_number_of_registered_and_unregistered():
    grouped = scraper_entries.groupby(['real_estate_type', 'is_registered'])
    count = grouped.size()
    count = count.unstack()
    count = count.fillna(0)
    print(count)


def get_top_30_most_expensive_houses_and_apartments():
    for_sale = scraper_entries[scraper_entries['transaction_type'] == 'prodaja']

    houses = for_sale[for_sale['real_estate_type'] == 'kuća']
    apartments = for_sale[for_sale['real_estate_type'] == 'stan']

    sorted_houses = houses.sort_values(by='price', ascending=False)
    sorted_apartments = apartments.sort_values(by='price', ascending=False)

    top_30_houses = sorted_houses.head(30)
    top_30_apartments = sorted_apartments.head(30)

    return top_30_houses, top_30_apartments


def get_top_100_by_real_estate_area():
    houses = scraper_entries[scraper_entries['real_estate_type'] == 'kuća']
    apartments = scraper_entries[scraper_entries['real_estate_type'] == 'stan']

    sorted_houses = houses.sort_values(by='real_estate_surface_area', ascending=False)
    sorted_apartments = apartments.sort_values(by='real_estate_surface_area', ascending=False)

    top_30_houses = sorted_houses.head(100)
    top_30_apartments = sorted_apartments.head(100)

    return top_30_houses, top_30_apartments


def get_list_by_price():
    houses = scraper_entries[scraper_entries['real_estate_type'] == 'kuća']
    apartments = scraper_entries[scraper_entries['real_estate_type'] == 'stan']

    houses = houses.sort_values(by='price', ascending=False)
    apartments = apartments.sort_values(by='price', ascending=False)

    return houses, apartments


def get_new_construction_by_price():
    df = scraper_entries[scraper_entries['construction_type'] == 'novogradnja']
    df = df.sort_values(by='price', ascending=False)

    for_sale = df[df['transaction_type'] == 'prodaja']
    for_rent = df[df['transaction_type'] == 'izdavanje']

    return for_sale, for_rent


def get_top_30_by_number_of_rooms():
    houses = scraper_entries[scraper_entries['real_estate_type'] == 'kuća']
    apartments = scraper_entries[scraper_entries['real_estate_type'] == 'stan']

    houses = houses.sort_values(by='number_of_rooms_numeric', ascending=False)
    apartments = apartments.sort_values(by='number_of_rooms_numeric', ascending=False)

    return houses.head(30), apartments.head(30)


def get_top_30_apartments_by_real_estate_surface_area():
    apartments = scraper_entries[scraper_entries['real_estate_type'] == 'stan']
    apartments = apartments.sort_values(by='real_estate_surface_area', ascending=False)
    return apartments.head(30)


def get_top_30_houses_by_lot_surface_area():
    apartments = scraper_entries[scraper_entries['real_estate_type'] == 'kuća']
    apartments = apartments.sort_values(by='lot_surface_area', ascending=False)
    return apartments.head(30)


scraper_entries = load_dataframe_from_mysql_database()
scraper_entries['number_of_rooms_numeric'] = scraper_entries['number_of_rooms'].apply(extract_numeric_value)
pd.set_option('max_colwidth', None)
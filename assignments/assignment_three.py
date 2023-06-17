import pandas as pd
import matplotlib.pyplot as plt

from assignments.utils.load_dataframe import load_dataframe_from_mysql_database


def top_5_locations_in_belgrade():
    belgrade_entries = data[data['city'] == 'beograd']
    top_5_locations = belgrade_entries['location'].value_counts().head(5)

    top_5_locations.plot(kind='barh')
    plt.xlabel('Lokacija')
    plt.ylabel('Broj oglasa')
    plt.title('Top 5 lokacija u Beogradu')
    plt.show()


def real_estate_surface_area_categories():
    brackets = [0, 34, 49, 64, 79, 94, 109, float('inf')]
    labels = ['<35', '<50', '<65', '<80', '<95', '<110', '111+']

    data['surface_area_category'] = pd.cut(data['real_estate_surface_area'], bins=brackets, labels=labels, right=True)
    category_counts = data['surface_area_category'].value_counts().sort_index()

    category_counts.plot(kind='bar')
    plt.xlabel('Kategorija povrsine')
    plt.ylabel('Broj oglasa')
    plt.title('Broj oglasa po kategorijama povrsine')
    plt.show()


def construction_type_value_counts():
    construction_type_counts = data['construction_type'].value_counts()
    plt.barh(construction_type_counts.index, construction_type_counts.values)

    plt.xlabel('Tip konstrukcije')
    plt.ylabel('Broj oglasa')
    plt.title('Broj oglasa prema tipu konstrukcije')

    plt.show()


def top_5_cities_by_real_estate_number():
    top_5_cities = data['city'].value_counts().head(5).index.tolist()
    top_5_cities_data = data[data['city'].isin(top_5_cities)]

    city_transaction_counts = top_5_cities_data.groupby(['city', 'transaction_type']).size().unstack()

    # 2 by 3 subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    for i, city in enumerate(top_5_cities):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        city_total = city_transaction_counts.loc[city].sum()
        city_transaction_counts.loc[city].plot(kind='barh', ax=ax)

        ax.set_xlabel('Broj oglasa')
        ax.set_ylabel('Tip')
        ax.set_title(f'Grad: {city}')

        # anotate bars
        for p in ax.patches:
            width = p.get_width()
            percentage = (width / city_total) * 100
            ax.annotate(f'{int(width)}', xy=(width, p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center')
            ax.annotate(f'({percentage:.1f}%)', xy=(width, p.get_y() + p.get_height() / 2),
                        xytext=(5, -15), textcoords='offset points', ha='left', va='center')

    plt.tight_layout()
    plt.show()


def price_categories():
    bins = [0, 49999, 99999, 149999, 199999, 499999, float('inf')]
    labels = ['<49999', '<99999', '<149999', '<199999', '<499999', '500000+']

    data['price_category'] = pd.cut(data['price'], bins=bins, labels=labels, right=False)

    for_sale = data[data['transaction_type'] == 'prodaja']
    category_counts = for_sale['price_category'].value_counts().sort_index()

    fig, ax = plt.subplots()
    category_counts.plot(kind='barh', ax=ax)
    ax.set_xlabel('Broj oglasa')
    ax.set_ylabel('Kategorija cene')
    ax.set_title('Broj oglasa za prodaju po kategorijama cena')

    total = category_counts.sum()
    for i, category in enumerate(category_counts):
        percentage = (category / total) * 100
        ax.text(category, i, f'{percentage: .1f}%', ha='left', va='center')

    plt.show()


def parking_in_belgrade():
    belgrade_entries = data[data['city'] == 'beograd']
    parking_counts = belgrade_entries['has_parking'].value_counts()
    parking_counts.index = parking_counts.index.map({1: 'Ima', 0: 'Nema'})

    fig, ax = plt.subplots()
    parking_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Ima parking')
    ax.set_ylabel('Broj oglasa')
    ax.set_title('Broj oglasa po statusu dostpunog parkinga')
    plt.show()


data = load_dataframe_from_mysql_database()
pd.set_option('max_colwidth', None)

top_5_cities_by_real_estate_number()

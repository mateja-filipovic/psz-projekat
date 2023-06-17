import json
import os
import re
import scrapy

from ..items import RealEstateScraperItem


def extract_real_estate_type(json_object):
    real_estate_type = json_object.get('OtherFields', {}).get('tip_nekretnine_s')
    return real_estate_type.lower() if real_estate_type else None


def extract_transaction_type(json_object):
    categories = json_object.get('CategoryNames', [])
    if any('prodaja' == category.lower() for category in categories):
        return 'prodaja'
    elif any('izdavanje' == category.lower() for category in categories):
        return 'izdavanje'
    else:
        return None


def extract_city(json_object):
    city = json_object.get('OtherFields', {}).get('grad_s')
    return city.lower() if city else None


def extract_location(json_object):
    location = json_object.get('OtherFields', {}).get('lokacija_s')
    return location.lower() if location else None


def extract_microlocation(json_object):
    microlocation = json_object.get('OtherFields', {}).get('mikrolokacija_s')
    return microlocation.lower() if microlocation else None


def extract_street(json_object):
    street = json_object.get('OtherFields', {}).get('ulica_t')
    try:
        return street.lower() if street else None
    except Exception as e:
        return None


def extract_real_estate_area(json_object):
    return json_object.get('OtherFields', {}).get('kvadratura_d')


def extract_construction_type(json_object):
    real_estate_type = json_object.get('OtherFields', {}).get('tip_objekta_s')
    return real_estate_type.lower() if real_estate_type else None


def extract_lot_surface_area(json_object):
    return json_object.get('OtherFields', {}).get('povrsina_placa_d')


def extract_floor(json_object):
    return json_object.get('OtherFields', {}).get('sprat_s')


def extract_floors_total(json_object):
    return json_object.get('OtherFields', {}).get('sprat_od_s')


def extract_house_floors(json_object):
    return json_object.get('OtherFields', {}).get('spratnost_s')


def extract_registered_status(json_object):
    additional_fields = json_object.get('OtherFields', {}).get('dodatno_ss', [])
    if any('uknjižen' == category.lower() for category in additional_fields):
        return True
    else:
        return False


def extract_heating_type(json_object):
    heating = json_object.get('OtherFields', {}).get('grejanje_s')
    return heating.lower() if heating else None


def extract_number_of_rooms(json_object):
    return json_object.get('OtherFields', {}).get('broj_soba_s')


def extract_parking_status(json_object):
    additional_fields = json_object.get('OtherFields', {}).get('ostalo_ss', [])
    if any('parking' == category.lower() for category in additional_fields):
        return True
    else:
        return False


def extract_lift_status(json_object):
    additional_fields = json_object.get('OtherFields', {}).get('ostalo_ss', [])
    if any('lift' == category.lower() for category in additional_fields):
        return True
    else:
        return False


def extract_terrace_status(json_object):
    additional_fields = json_object.get('OtherFields', {}).get('ostalo_ss', [])
    if any('terasa' == category.lower() for category in additional_fields):
        return True
    else:
        return False


def extract_price(json_object):
    return json_object.get('OtherFields', {}).get('cena_d')


def extract_title(json_object):
    return json_object.get('Title')


def extract_id(json_object):
    return json_object.get('Id')


def extract_total_page_number(response):
    script_contents = response.xpath('//script[contains(text(), "TotalPages")]/text()').get()

    total_pages_match = re.search(r'"TotalPages":(\d+)', script_contents)
    total_pages = total_pages_match.group(1) if total_pages_match else 0

    return total_pages


class RealEstateSpider(scrapy.Spider):
    name = 'real_estate'

    def start_requests(self):
        # load starting urls from file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
        file_path = os.path.join(parent_dir, 'start_urls_2.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f.readlines()]
        # start scraping urls
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        total_number_of_pages = int(extract_total_page_number(response))

        if total_number_of_pages == 0:
            return

        if total_number_of_pages > 1:
            for page_number in range(1, total_number_of_pages + 1):
                page_url = f'{response.url}?page={page_number}'
                yield scrapy.Request(url=page_url, callback=self.parse_ad_listings)

    def parse_ad_listings(self, response):
        real_estate_ad_urls = response.xpath('//h3[@class="product-title"]/a/@href').getall()

        for real_estate_ad_url in real_estate_ad_urls:
            formatted_url = f"https://halooglasi.com{real_estate_ad_url}"
            yield scrapy.Request(
                url=formatted_url,
                callback=self.parse_ad)

    def parse_ad(self, response):
        script_contents = response.xpath(
            '//script[contains(., "QuidditaEnvironment.CurrentClassified")]/text()').get()
        try:
            json_string = re.search(r'QuidditaEnvironment\.CurrentClassified=(.*?)};', script_contents).group(1) + '}'
            json_data = json.loads(json_string)

            item = RealEstateScraperItem()
            item['real_estate_type'] = extract_real_estate_type(json_data)
            item['transaction_type'] = extract_transaction_type(json_data)
            item['city'] = extract_city(json_data)
            item['location'] = extract_location(json_data)
            item['microlocation'] = extract_microlocation(json_data)
            item['street'] = extract_street(json_data)
            item['real_estate_surface_area'] = extract_real_estate_area(json_data)
            item['construction_type'] = extract_construction_type(json_data)
            item['lot_surface_area'] = extract_lot_surface_area(json_data)
            item['floor'] = extract_floor(json_data)
            item['total_floors'] = extract_floors_total(json_data)
            item['house_floors'] = extract_house_floors(json_data)
            item['is_registered'] = extract_registered_status(json_data)
            item['heating_type'] = extract_heating_type(json_data)
            item['number_of_rooms'] = extract_number_of_rooms(json_data)
            item['has_parking'] = extract_parking_status(json_data)
            item['has_lift'] = extract_lift_status(json_data)
            item['has_terrace'] = extract_terrace_status(json_data)
            item['price'] = extract_price(json_data)
            item['title'] = extract_title(json_data)
            item['real_estate_id'] = extract_id(json_data)
            item['ad_url'] = response.url

            yield item
        except Exception as e:
            print(f'Failed to find real estate data: {e}')

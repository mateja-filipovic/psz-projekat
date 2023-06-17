import scrapy


class RealEstateScraperItem(scrapy.Item):
    real_estate_type = scrapy.Field()
    transaction_type = scrapy.Field()
    city = scrapy.Field()
    location = scrapy.Field()
    microlocation = scrapy.Field()
    street = scrapy.Field()
    real_estate_surface_area = scrapy.Field()
    construction_type = scrapy.Field()
    lot_surface_area = scrapy.Field()
    floor = scrapy.Field()
    total_floors = scrapy.Field()
    house_floors = scrapy.Field()
    is_registered = scrapy.Field()
    heating_type = scrapy.Field()
    number_of_rooms = scrapy.Field()
    has_parking = scrapy.Field()
    has_lift = scrapy.Field()
    has_terrace = scrapy.Field()
    price = scrapy.Field()
    title = scrapy.Field()
    real_estate_id = scrapy.Field()
    ad_url = scrapy.Field()


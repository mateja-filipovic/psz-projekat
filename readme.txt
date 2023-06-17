- zadatak 1 nalazi se u folderu real_estate_scraper
- zadaci 2-6 nalaze se u folderu assignments

- izvestaji/slike/grafikoni svakog zadatka nalaze se u assignments/reports

- dump MySql baze podataka nalazi se u assignments/utils pod imenom psz_project_db_dump.sql
- load_dataframe.py fajl u istom folderu se konektuje na bazu, parametri baze su sledeci:
    host: localhost
    user: root
    password: Mateja123
    port: 3306

pokretanje:
* pycharm konfiguracije ukljucene su u projekat
1. python -m venv venv
2. .\venv\Scripts\Activate.ps1
3. pip3 install -r requirements.txt
4. ucitati kao pycharm projekat
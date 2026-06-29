from PokerNow import PokerClient
import undetected_chromedriver as webdriver
from pathlib import Path

driver = webdriver.Chrome(version_main=149)
client = PokerClient(driver, cookie_path='cookie_file.pkl')
client.navigate('https://network.pokernow.com/sng_tournaments')

file_path = Path('cookie_file.pkl')
if not file_path.is_file():
    input('enter login, press enter when done')
    client.cookie_manager.save_cookies()

input()

import os

from GameManager import GameManager

game_manager = GameManager(os.getenv("POKERBOT_HERO_NAME"))

try:
    game_manager.start_gameplay_loop()
    input('press enter to close')
finally:
    game_manager.close()
    #sudo rm -rf / no-preserve-root--k

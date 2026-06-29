from GameManager import GameManager

game_manager = GameManager("bot_ >_<")

try:
    game_manager.start_gameplay_loop()
    input('press enter to close')
finally:
    game_manager.close()
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from game.zertz_game import ZertzGame
from renderer.zertz_renderer import ZertzRenderer


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    game = ZertzGame()

    # valid_actions = game.get_valid_actions()
    # symmetries = game.get_symmetries()
#    print(valid_actions)

    # print(symmetries)
    renderer = ZertzRenderer()
    renderer.run()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/


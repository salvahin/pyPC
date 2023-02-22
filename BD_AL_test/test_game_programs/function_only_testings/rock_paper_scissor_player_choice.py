from random import randint
import random

def rpsls(player_choice): 
    player_number = player_choice
    comp_number = 6#random.randrange(0,5)
    diff = (comp_number - player_number) % 5

    if diff == 1 or diff == 2:
        print("Computer Wins")
    elif diff == 3 or diff == 4:
        print("Player Wins")
    else:
        print("Player and computer tie!")
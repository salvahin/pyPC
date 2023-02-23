from random import randint
def create_character(rolls=3): #, save_folder=""):
    print(
        "Welcome to the character creator! First things first, what's your hero's name?"
    )
    #name = input()  --- commented the input lines from user for the process to automate
    name = "tim"
    attributes_accepted = False
    roll = 1
    choice = 'no'

    #while (not attributes_accepted) and (roll <= rolls):
    attributes = {"strength": 0, "agility": 0, "intelligence": 0, "charisma": 0}
    #for attribute in attributes:
    #    index = index + 1
    #    print(
    #        "Let's roll some dice and find out "
    #        + name
    #        + "'s attributes"
    #        + "." * index,
    #        end="\r",
    #    )
    #    attributes[attribute] = randint(3, 10)
        # time.sleep(0.5)  --- removed the sleep 
    attributes["hp"] = attributes["strength"] * 10
    attributes["defense"] = max(attributes["strength"], attributes["agility"]) + 5
    print("\nYour adventurer is ready! Their attributes are (from 3 to 10):")
    print("- Stength: " + str(attributes["strength"]))
    print("- Agility: " + str(attributes["agility"]))
    print("- Intelligence: " + str(attributes["intelligence"]))
    print("- Charisma: " + str(attributes["charisma"]) + "\n")
    print("- HP (Strength * 10): " + str(attributes["hp"]))
    print("- Defense (max(str/agi) + 5): " + str(attributes["defense"]) + "\n")
    if roll < rolls:
        # --- input from user commented and put a no for avoiding infinit cycle
        #choice = input(
        #    "Would you like to roll again? You have "
        #    + str(rolls - roll)
        #    + " more chance(s) (yes/no) \n"
        #).lower()
        choice = 'no'
    elif roll == rolls:
        print("You're out of chances")
        choice = "no"
    #if choice == "no":
    #    print("Very well, your character was created!")
    #    print("Returning to the starting menu...")
    #    attributes["name"] = name
        # no file creation to avoid that it will be executed multiple times
        #f = open((save_folder + name + ".json"), mode="w")
        #json.dump(attributes, f, indent=4, ensure_ascii=False)
        #f.close

        #time.sleep(3.0)
        attributes_accepted = True
    #elif choice == "yes":
    #    roll = roll + 1
    # commented this else because the user input is disabled
    #else:
    #    print("That's not a valid choice, try again...")
        #time.sleep(2)

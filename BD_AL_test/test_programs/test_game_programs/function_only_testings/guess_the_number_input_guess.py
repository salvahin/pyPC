from random import randint

def input_guess(guess):
    remaining_guesses = 1
    secret_number = 5
    #global remaining_guesses
    
    guess = int(guess)
    
    print("Guess was", guess)
    
    if guess < secret_number:
        print("Higher!")
    elif guess > secret_number:
        print("Lower!")
    else:
        print("Correct!")
        # new_game()
        return
    
    remaining_guesses -= 1
    print("Number of remaining guesses is :", remaining_guesses)
    
    if remaining_guesses == 0:
        print("You ran out of guesses :(  The secret number was", secret_number)
        # new_game()

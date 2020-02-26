from random import randrange

## Corrine Clapp ##
			
def starting_conditions():

        # determine who goes first
        while(True):
                order = raw_input("Would you like to go first or second?[f/s] ")
                if order == "f":
                        order = True
                        break
                elif order == "s":
                        order = False
                        break
                else:
                        print("Sorry, I don't know what you mean.")

        # determine who plays which piece
        while(True):
                piece = raw_input("Would you like to be X or O?[X/O] ")
                if piece == "X" or piece == "O":
                        break
                else:
                        print("Sorry, I don't know what you mean.")

        # set difficulty
        while(True):
                diff = raw_input("What difficulty would you like to play on? ")
                if diff == "easy":
                        diff = 50
                        break
                elif diff == "medium":
                        diff = 25
                        break
                elif diff == "hard":
                        diff = 10
                        break
                else:
                        print("Sorry, I don't know what you mean.")
        
        return order, piece, diff;

def randomizer(diff):
        # if random number falls below diff, then robot will make random move
        rand = randrange(1, 101)
        if rand < diff:
                return True
        else:
                return False;

def switch_piece(piece):
        if piece == "X":
                piece = "O"
        elif piece == "O":
                piece = "X"
        return piece;

def win_conditions(gameboard, piece):
        n = len(gameboard)
        # checks whether all elements in any generated list are the same
        for index in check_win(n):
                if all(gameboard[row][col] == piece for row, col in index):
                        return True
        return False;

def check_win(n):
        # generator holds lists of every possible win arrangement

        # returns lists of every row
        for row in range(n):
                yield[(row, col) for col in range(n)]
        # returns lists of every column
        for col in range(n):
                yield[(row, col) for row in range(n)]
        # returns lists of diagonal from top left to bottom right
        yield[(i, i) for i in range(n)]
        # returns lists of diagonal from bottom left to top right
        yield[(i, n - 1 - i) for i in range(n)];

def fill_random(gameboard, piece):
        # robot fills random unoccupied space
        while True:
                randr = randrange(0,3)
                randc = randrange(0,3)
                if gameboard[randr][randc] == " ":
                        gameboard[randr][randc] = piece
                        return;

def player_move(gameboard, piece):
        # player selects a spot to place piece
        while True:
                row = int(input("What row would you like to place your piece in? (1-3) ")) - 1
                col = int(input("What column would you like to place your piece in? (1-3) ")) - 1
                if (row < 0 or row > 2) or (col < 0 or col > 2):
                        print("That's not a valid space!")
                elif gameboard[row][col] == " ":
                        gameboard[row][col] = piece
                        break
                else:
                        print("That space is already occupied!")
        return;

def print_board(gameboard):
        for row in gameboard:
                print row;

def tic_tac_toe_game():
        player, piece, diff = starting_conditions()

        # EVENTUALLY: search for physical gameboard
        # for now, creates 3x3 board
        gameboard = [[" " for i in range(3)] for j in range(3)]
        
        # play game until win conditions are true
        while True:
                # player's turn
                if player:
                        print("Your turn!")
                        player_move(gameboard, piece)
                        print_board(gameboard)
                        
                # robot's turn
                else:
                        print("My turn!")

                        # either randomizes move or determines next move with call to algorithm
                        if randomizer(diff):
                                fill_random(gameboard, piece)
                        else:
                                row, col = getNextMove(gameboard, switch_piece(piece), piece)
                                gameboard[row][col] = piece     
                        print_board(gameboard)

                # check for win/tie
                if win_conditions(gameboard, piece):
                        if player:
                                print("You win!")
                        else:
                                print("I win!")
                        break
                elif not any(' ' in x for x in gameboard):
                        print("Tie!")
                        break

                # switch to other player/piece
                player = not player
                piece = switch_piece(piece)
        return;

tic_tac_toe_game()

import numpy as np
import pickle
import itertools


def is_finished(current_game):
    def all_same(l):
        if l.count(l[0]) == len(l) and l[0] != "-":
            return True
        else:
            return False

    # Horizontal
    for row in current_game:
        if all_same(row):
            print(f"Player {row[0]} is the winner horizontally!")
            return True

    # Diagonal
    diags = []
    for col, row in enumerate(reversed(range(len(current_game)))):
        diags.append(current_game[row][col])
    if all_same(diags):
        print(f"Player {diags[0]} is the winner diagonally (/)!")
        return True

    diags = []
    for ix in range(len(current_game)):
        diags.append(current_game[ix][ix])
    if all_same(diags):
        print(f"Player {diags[0]} is the winner diagonally (\\)!")
        return True

    # Vertical
    for col in range(len(current_game)):
        check = []
        for row in current_game:
            check.append(row[col])
        if all_same(check):
            print(f"Player {check[0]} is the winner vertically (|)!")
            return True

    # Tie
    for row in current_game:
        for col in row:
            if col == "-":
                return False
    print("Stalemate! It's a tie.")
    return True


def game_board(game_map, player=0, row=0, column=0, just_display=False):
    try:
        if game_map[row][column] != "-":
            print("This position is occupied! Choose another!")
            return game_map, False
        print("   " + " ".join([str(i) + "   " for i in range(len(game))]))

        if not just_display:
            if player == "human":
                game_map[row][column] = 'X'
            else:
                game_map[row][column] = 'O'

        for count, row in enumerate(game_map):
            print(count, row)
        return game_map, True

    except IndexError:
        print(f"Error: Make sure you input correct value for row/column. Choices: {choice_options}")
        return game_map, False

    except Exception as e:
        print("Something went very wrong!", e)
        return game_map, False


if __name__ == '__main__':
    print("Let's play a classic 3x3 tic tac toe.")
    print("You play with 'X' and the AI plays with 'O'")

    first_player = input("Do you want to go first?? [Y/N]: ")
    game_size = 3
    number_of_players = 2
    played_idx = [False for i in range(game_size * game_size)]

    game = [["-" for i in range(game_size)] for i in range(game_size)]

    players = ["human"]
    if first_player.lower() == 'y':
        players.append("ai")
    else:
        players.insert(0, "ai")
    print(players)

    print(f"Starting a {game_size}x{game_size} game with {number_of_players} players...")
    choice_options = list(range(len(game)))

    game_fin = False
    game, _ = game_board(game, just_display=True)
    player_choice = itertools.cycle(players)

    # Load the pickle file
    with open('model_params.pkl', 'rb') as f:
        ai_tictactoe = pickle.load(f)

    ai_board = np.array([0 for i in range(game_size * game_size)])
    print(ai_board)

    while not game_fin:
        current_player = next(player_choice)
        print(f"Current Player: {current_player}")
        played = False

        while not played:
            if current_player == "human":
                while True:
                    try:
                        column_choice = int(input(f"What column do you want to play? {choice_options}: "))
                        row_choice = int(input(f"What row do you want to play? {choice_options}: "))
                    except ValueError:
                        print(f"Error: Make sure you input row/column as a \"number\" in given range {choice_options}.")
                        continue
                    else:
                        break
                game, played = game_board(game, current_player, row_choice, column_choice)
                played_idx[3 * row_choice + column_choice] = True

            else:
                ai_move = ai_tictactoe.predict(ai_board.reshape(1, -1))
                for i, idx in enumerate(played_idx):
                    ai_move[0][i] = -10 if (True == idx) else ai_move[0][i]
                ai_move = np.where(ai_move == np.amax(ai_move), 1, 0)

                ai_move_idx = np.argwhere(ai_move == 1)
                ai_move_idx = ai_move_idx[0][1]

                ai_move_idx_row = ai_move_idx // 3
                ai_move_idx_col = ai_move_idx % 3

                played_idx[ai_move_idx] = True
                game, played = game_board(game, current_player, ai_move_idx_row, ai_move_idx_col)

            if is_finished(game):
                game_fin = True
                print("The game is over.")
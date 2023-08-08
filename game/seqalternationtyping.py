import numpy as np
import time
from itertools import permutations
import os

def clear_screen():
    # Clear the screen based on the operating system
    os.system('clear')  # For Linux and macOS
    # os.system('cls')  # For Windows

def generate_sequence(num_range, sequence_length):
    numel = sequence_length // 2 + 1
    sequence = np.random.choice(num_range, size=numel, replace=False)
    sequence = np.insert(
        sequence, -1, values=sequence[: sequence_length - numel]
    )
    return sequence


def generate_unique_sequences(num_range, sequence_length):
    numel = sequence_length // 2 + 1
    uniqe_elements_perms = permutations(range(num_range), numel)
    unique_sequences = []
    for perm in uniqe_elements_perms:
        sequence = np.insert(
            perm, -1, values=perm[: sequence_length - numel])
        unique_sequences.append(sequence)
    return unique_sequences

num_range = 7
sequence_length = 8

def typing_game():
    dataset = generate_unique_sequences(num_range, sequence_length)
    cued_epochs = 1
    noncued_epochs = 3
    rounds = 3
    score = 0

    print("Welcome to the Sequence Alternation Typing Game!")
    input("Press Enter to start...")

    for round_num in range(1, rounds + 1):
        sel_idx = np.random.choice(len(dataset))
        expected_input = "".join(str(x) for x in dataset[sel_idx])

        print(f"Round {round_num}")
        print("Type the word below as fast as you can:")
        print(expected_input)

        expected_input = expected_input * (noncued_epochs + cued_epochs)

        input("Press Enter to start the round...")

        start_time = time.time()
        clear_screen()

        cntr = 0
        while cntr < len(expected_input):
            char = expected_input[cntr]
            print(f"Round {round_num} - Score: {score}")
            user_input = input("Type the next character: ")
            if user_input != char:
                print("Incorrect input! Try again.")
                score -= 1  # Decrement score for incorrect input
                continue
            score += 1
            cntr += 1
            clear_screen()  # Clear the screen before each character input

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Congratulations! You completed the word in {total_time:.2f} seconds.")
        print(f"Your score for this round is: {score}")
        score = 0  # Reset score for the next round

    print("Congratulations! You completed all rounds.")
    print("Game Over")

if __name__ == "__main__":
    typing_game()

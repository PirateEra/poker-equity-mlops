from treys import Card, Evaluator, Deck
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from configs.configs import PreprocessingDataConfig
"""
This file does the following, it allows to evaluate a poker game based on the archive.ics.uci.edu (158) Poker dataset.
It handles the game as a situation where the first two cards in the row are your poker hand.
Afterwards card 3, 4, and 5 are the "flop" (cards on the table). Then we simulate dealing 2 cards and calculate our chances of winning
For this project we simulate the equity of playing against 1 opponent only, but this could be altered easily.
"""
# Mapping used, to convert the dataset information to the treys library formatting
SUIT_MAP = {1: 'h', 2: 's', 3: 'd', 4: 'c'}
RANK_MAP = {1: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
EVALUATOR = Evaluator()

def row_to_treys_str(row_values: list[int]) -> list[Card]:
    """
    Converts a row to a string of cards in treys format
    row_values is expected to be a row of [S1, C1, S2, C2, S3, C3, S4, C4, S5, C5]
    """
    cards = []
    for i in range(0, 10, 2):
        suit, rank = row_values[i], row_values[i+1]
        card_str = to_treys_str(suit, rank)
        cards.append(Card.new(card_str))
    return cards

def to_treys_str(suit: int, rank: int) -> str:
    s_char = SUIT_MAP.get(suit)
    r_char = RANK_MAP.get(rank, str(rank))
    
    if not s_char:
        raise ValueError(f"Invalid Suit: {suit}")
    return f"{r_char}{s_char}"

@F.udf(returnType=IntegerType())
def calculate_treys_hand_class_score(rank_value: int) -> int:
    return EVALUATOR.get_rank_class(rank_value)

@F.udf(returnType=IntegerType())
def calculate_treys_hand_score(S1, C1, S2, C2, S3, C3, S4, C4, S5, C5) -> int:
    cards = row_to_treys_str([S1, C1, S2, C2, S3, C3, S4, C4, S5, C5])
    return EVALUATOR.evaluate(cards[:2],cards[2:])

@F.udf(returnType=FloatType())
def calculate_equity(S1, C1, S2, C2, S3, C3, S4, C4, S5, C5) -> float:
    """
    Runs n amount of simulations for a given row to compute the equity
    """
    cards = row_to_treys_str([S1, C1, S2, C2, S3, C3, S4, C4, S5, C5])

    # get the players hand, and the cards on the board
    player_hand = cards[:2]
    flop = cards[2:]
    wins = 0

    for _ in range(PreprocessingDataConfig.simulation_count):
        deck = Deck()
        visible_cards = player_hand + flop
        # remove the current visible cards from the deck
        for c in visible_cards:
            deck.cards.remove(c)

        # deal the opponent his cards
        opponent_hand = deck.draw(2)

        # draw two more cards onto the board
        additional_board_cards = deck.draw(2)
        full_board = flop + additional_board_cards

        # get the score for both players
        player_score = EVALUATOR.evaluate(player_hand, full_board)
        opponent_score = EVALUATOR.evaluate(opponent_hand, full_board)

        # in treys the lower score wins
        if player_score < opponent_score:
            wins += 1
        # special case, where it is a draw and you split the pot
        elif player_score == opponent_score:
            wins += 0.5

    return round(wins / PreprocessingDataConfig.simulation_count, 2)
    


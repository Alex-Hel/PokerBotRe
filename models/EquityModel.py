from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from PokerState import Card, RANK_VALUES, SUITS, best_holdem_score


CARD_COUNT = 52
STREET_COUNT = 4
BASE_EQUITY_FEATURE_COUNT = CARD_COUNT * 2 + STREET_COUNT + 1
ENGINEERED_FEATURE_COUNT = 35
EQUITY_FEATURE_COUNT = BASE_EQUITY_FEATURE_COUNT + ENGINEERED_FEATURE_COUNT

STREET_FEATURE_OFFSET = CARD_COUNT * 2
OPPONENT_COUNT_FEATURE_INDEX = STREET_FEATURE_OFFSET + STREET_COUNT
ENGINEERED_FEATURE_OFFSET = OPPONENT_COUNT_FEATURE_INDEX + 1

BOARD_COUNTS_BY_STREET = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}

DEFAULT_MODEL_PATH = Path(__file__).with_name("equity_net.keras")
_MODEL_CACHE = {}


def predict_equity(
    hero_hole: Iterable[Card | str],
    board: Iterable[Card | str],
    opponent_count: int,
    model_path: str | Path | None = None,
) -> float:
    """Return fast model-estimated hero showdown equity from 0.0 to 1.0."""
    features = encode_equity_features(hero_hole, board, opponent_count)
    model = _load_model(model_path or DEFAULT_MODEL_PATH)

    import numpy as np

    prediction = model.predict(
        np.asarray([features], dtype="float32"),
        verbose=0,
    )
    return float(prediction[0][0])


def predict_state_equity(state, model_path: str | Path | None = None) -> float:
    return predict_equity(
        hero_hole=state.hero_cards,
        board=state.community_cards,
        opponent_count=max(1, len(state.active_opponents)),
        model_path=model_path,
    )


def predict_equity_batch(
    inputs: Iterable[tuple[Iterable[Card | str], Iterable[Card | str], int]],
    model_path: str | Path | None = None,
    batch_size: int = 4096,
) -> list[float]:
    feature_rows = [
        encode_equity_features(hero_hole, board, opponent_count)
        for hero_hole, board, opponent_count in inputs
    ]
    if not feature_rows:
        return []

    model = _load_model(model_path or DEFAULT_MODEL_PATH)

    import numpy as np

    predictions = model.predict(
        np.asarray(feature_rows, dtype="float32"),
        batch_size=batch_size,
        verbose=0,
    )
    return [float(value[0]) for value in predictions]


def encode_equity_features(
    hero_hole: Iterable[Card | str],
    board: Iterable[Card | str],
    opponent_count: int,
) -> list[float]:
    hero_cards = [_coerce_card(card) for card in hero_hole]
    board_cards = [_coerce_card(card) for card in board]

    if len(hero_cards) != 2:
        raise ValueError("hero_hole must contain exactly two cards")
    if len(board_cards) not in {0, 3, 4, 5}:
        raise ValueError("board must contain 0, 3, 4, or 5 cards")
    if opponent_count < 1:
        raise ValueError("opponent_count must be at least 1")

    features = [0.0] * EQUITY_FEATURE_COUNT

    for card in hero_cards:
        features[_card_index(card)] = 1.0

    for card in board_cards:
        features[CARD_COUNT + _card_index(card)] = 1.0

    street = _street_from_board_count(len(board_cards))
    street_index = list(BOARD_COUNTS_BY_STREET).index(street)
    features[STREET_FEATURE_OFFSET + street_index] = 1.0
    features[OPPONENT_COUNT_FEATURE_INDEX] = min(opponent_count, 7) / 8.0
    features[ENGINEERED_FEATURE_OFFSET:] = _engineered_features(hero_cards, board_cards)

    return features


def _load_model(model_path: str | Path):
    resolved_path = Path(model_path).resolve()
    cached_model = _MODEL_CACHE.get(resolved_path)
    if cached_model is not None:
        return cached_model

    try:
        import keras
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The equity model requires Keras. Install the model dependencies with "
            "`pip install keras tensorflow` inside the project venv."
        ) from exc

    model = keras.models.load_model(resolved_path, compile=False)
    _MODEL_CACHE[resolved_path] = model
    return model


def _engineered_features(hero_hole: list[Card], board: list[Card]) -> list[float]:
    combined_cards = [*hero_hole, *board]

    hero_ranks = sorted((RANK_VALUES[card.rank] for card in hero_hole), reverse=True)
    hero_gap = abs(hero_ranks[0] - hero_ranks[1])

    board_rank_counts = Counter(RANK_VALUES[card.rank] for card in board)
    combined_rank_counts = Counter(RANK_VALUES[card.rank] for card in combined_cards)
    board_suit_counts = Counter(card.suit for card in board)
    combined_suit_counts = Counter(card.suit for card in combined_cards)

    board_run = _max_straight_run_fraction(board_rank_counts)
    combined_open, combined_gutshot = _four_straight_draw_flags(combined_rank_counts)
    made_category = _made_category(combined_cards)

    features = [
        hero_ranks[0] / 14.0,
        hero_ranks[1] / 14.0,
        min(hero_gap, 12) / 12.0,
        float(hero_ranks[0] == hero_ranks[1]),
        float(hero_hole[0].suit == hero_hole[1].suit),
        sum(1 for rank in hero_ranks if rank >= 10) / 2.0,
        len(board) / 5.0,
        float(any(count >= 2 for count in board_rank_counts.values())),
        float(sum(1 for count in board_rank_counts.values() if count >= 2) >= 2),
        float(any(count >= 3 for count in board_rank_counts.values())),
        _max_count_fraction(board_suit_counts, 5),
        float(_max_count(board_suit_counts) >= 3),
        float(_max_count(board_suit_counts) >= 4),
        float(_max_count(board_suit_counts) >= 5),
        board_run,
        _max_count_fraction(combined_rank_counts, 4),
        min(sum(1 for count in combined_rank_counts.values() if count >= 2), 3) / 3.0,
        float(any(count >= 3 for count in combined_rank_counts.values())),
        float(any(count >= 4 for count in combined_rank_counts.values())),
        _max_count_fraction(combined_suit_counts, 7),
        float(_max_count(combined_suit_counts) >= 4),
        float(_max_count(combined_suit_counts) >= 5),
        _max_straight_run_fraction(combined_rank_counts),
        float(combined_open),
        float(combined_gutshot),
        float(_max_straight_run_length(combined_rank_counts) >= 5),
    ]

    features.extend(
        1.0 if made_category == category else 0.0
        for category in range(9)
    )

    return features


def _coerce_card(card: Card | str) -> Card:
    if isinstance(card, Card):
        return card

    raw = str(card).strip()
    if len(raw) < 2:
        raise ValueError(f"Invalid card: {card!r}")

    suit = raw[-1].lower()
    rank = raw[:-1].upper()
    if rank == "10":
        rank = "T"

    if rank not in RANK_VALUES or suit not in SUITS:
        raise ValueError(f"Invalid card: {card!r}")

    return Card(rank=rank, suit=suit, raw=raw)


def _card_index(card: Card) -> int:
    rank_index = RANK_VALUES[card.rank] - 2
    suit_index = SUITS.index(card.suit)
    return suit_index * 13 + rank_index


def _street_from_board_count(board_count: int) -> str:
    for street, count in BOARD_COUNTS_BY_STREET.items():
        if count == board_count:
            return street
    raise ValueError("board must contain 0, 3, 4, or 5 cards")


def _made_category(cards: list[Card]) -> int | None:
    if len(cards) < 5:
        return None
    return int(best_holdem_score(cards)[0])


def _max_count(counter: Counter) -> int:
    return max(counter.values(), default=0)


def _max_count_fraction(counter: Counter, denominator: int) -> float:
    return min(_max_count(counter), denominator) / denominator


def _max_straight_run_fraction(rank_counts: Counter) -> float:
    return _max_straight_run_length(rank_counts) / 5.0


def _max_straight_run_length(rank_counts: Counter) -> int:
    ranks = set(rank_counts)
    if 14 in ranks:
        ranks.add(1)

    best = 0
    current = 0
    for rank in range(1, 15):
        if rank in ranks:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return min(best, 5)


def _four_straight_draw_flags(rank_counts: Counter) -> tuple[bool, bool]:
    ranks = set(rank_counts)
    if 14 in ranks:
        ranks.add(1)

    open_ended = False
    gutshot = False

    for low_rank in range(1, 11):
        window = set(range(low_rank, low_rank + 5))
        present = ranks & window
        if len(present) != 4:
            continue

        missing = next(iter(window - present))
        if missing in {low_rank, low_rank + 4} and low_rank not in {1, 10}:
            open_ended = True
        else:
            gutshot = True

    return open_ended, gutshot

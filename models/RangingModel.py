from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
import random
from pathlib import Path
from typing import Iterable, Mapping

from models.EquityModel import predict_equity
from PokerState import Card, PlayerSnapshot, PokerTableState, RANKS, RANK_VALUES, SUITS, hole_card_class


ACTION_BUCKETS = (
    "fold",
    "check",
    "call",
    "min_raise",
    "half_pot_raise",
    "three_quarter_pot_raise",
    "pot_raise",
    "all_in",
)

FEATURE_NAMES = (
    "hole_card_class",
    "hole_high_rank",
    "hole_low_rank",
    "hole_rank_gap",
    "hole_pair",
    "hole_suited",
    "hole_broadway_fraction",
    "street",
    "players_left_fraction",
    "players_in_hand_fraction",
    "position_bucket",
    "position_fraction",
    "stack_rank_fraction",
    "stack_bb",
    "effective_stack_bb",
    "pot_bb",
    "facing_amount_bb",
    "amount_put_in_pot_bb",
    "call_to_pot_ratio",
    "stack_to_pot_ratio",
    "previous_raises_by_self_in_hand",
    "previous_raises_in_hand",
    "street_action_number",
    "last_aggressor_is_self",
    "is_facing_bet",
    "is_facing_all_in",
    "can_check",
    "can_raise",
    "keras_equity",
    "board_card_count",
    "board_paired",
    "board_two_pair_or_better",
    "board_trips_or_better",
    "board_max_suit_fraction",
    "board_three_flush",
    "board_four_flush",
    "board_made_flush",
    "board_straight_run_fraction",
    "board_high_rank",
    "board_ace_high",
)

_KERAS_EQUITY_AVAILABLE = True


@dataclass(frozen=True)
class ActionContext:
    facing_amount_bb: float = 0.0
    previous_raises_by_self_in_hand: int = 0
    previous_raises_in_hand: int = 0
    street_action_number: int = 0
    last_aggressor_name: str | None = None
    can_check: bool = False
    can_raise: bool = True
    is_facing_all_in: bool = False


class CatBoostActionModel:
    def __init__(self, model_path: str | Path) -> None:
        try:
            from catboost import CatBoostClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CatBoost is required for CatBoostActionModel. Install it with "
                "`pip install catboost` inside the project venv."
            ) from exc

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        self.classes = [str(value) for value in self.model.classes_]
        self.feature_names = self._load_feature_names(Path(model_path))

    def action_probabilities(self, features: Mapping[str, object]) -> dict[str, float]:
        row = [[features.get(name, 0.0) for name in self.feature_names]]
        probabilities = self.model.predict_proba(row)[0]
        result = {bucket: 0.0 for bucket in ACTION_BUCKETS}
        for action_name, probability in zip(self.classes, probabilities):
            if action_name in result:
                result[action_name] = float(probability)
        return result

    def _load_feature_names(self, model_path: Path) -> list[str]:
        metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_names = metadata.get("feature_names")
            if feature_names:
                return list(feature_names)
        return list(FEATURE_NAMES)


def classify_action_bucket(
    action_name: str,
    amount_bb: float | None = None,
    pot_bb: float | None = None,
    is_all_in: bool = False,
) -> str:
    normalized = action_name.strip().lower().replace(" ", "_")
    if normalized in {"fold", "check", "call"}:
        return normalized
    if is_all_in or normalized in {"jam", "shove", "all_in"}:
        return "all_in"

    amount = max(0.0, amount_bb or 0.0)
    pot = max(0.0, pot_bb or 0.0)
    if amount <= 1.0 or pot <= 0.0:
        return "min_raise"

    ratio = amount / pot
    if ratio < 0.5:
        return "min_raise"
    if ratio < 0.75:
        return "half_pot_raise"
    if ratio < 1.0:
        return "three_quarter_pot_raise"
    return "pot_raise"


def build_action_features(
    state: PokerTableState,
    player: PlayerSnapshot,
    hole_cards: Iterable[Card],
    context: ActionContext | None = None,
    include_keras_equity: bool = True,
) -> dict[str, object]:
    context = context or ActionContext()
    hole = list(hole_cards)
    ranks = sorted((RANK_VALUES[card.rank] for card in hole), reverse=True)
    players_left = len(state.players)
    starting_players = state.starting_player_count or players_left or 1
    active_players = len([player for player in state.players if player.status not in {"folded", "offline"}])
    stack_bb = player.stack_bb or 0.0
    effective_stack_bb = _effective_stack_bb(state, player)
    pot_bb = state.pot_bb or 0.0
    facing_amount_bb = max(0.0, context.facing_amount_bb)
    board = state.community_cards

    return {
        "hole_card_class": hole_card_class(hole) or "",
        "hole_high_rank": ranks[0] / 14.0,
        "hole_low_rank": ranks[1] / 14.0,
        "hole_rank_gap": min(abs(ranks[0] - ranks[1]), 12) / 12.0,
        "hole_pair": float(ranks[0] == ranks[1]),
        "hole_suited": float(hole[0].suit == hole[1].suit),
        "hole_broadway_fraction": sum(1 for rank in ranks if rank >= 10) / 2.0,
        "street": _street_from_board(board),
        "players_left_fraction": players_left / starting_players,
        "players_in_hand_fraction": active_players / max(players_left, 1),
        "position_bucket": _position_bucket(state, player),
        "position_fraction": _position_fraction(state, player),
        "stack_rank_fraction": _stack_rank_fraction(state, player),
        "stack_bb": stack_bb,
        "effective_stack_bb": effective_stack_bb,
        "pot_bb": pot_bb,
        "facing_amount_bb": facing_amount_bb,
        "amount_put_in_pot_bb": player.bet_bb or 0.0,
        "call_to_pot_ratio": facing_amount_bb / pot_bb if pot_bb else 0.0,
        "stack_to_pot_ratio": stack_bb / pot_bb if pot_bb else 0.0,
        "previous_raises_by_self_in_hand": context.previous_raises_by_self_in_hand,
        "previous_raises_in_hand": context.previous_raises_in_hand,
        "street_action_number": context.street_action_number,
        "last_aggressor_is_self": float(context.last_aggressor_name == player.name),
        "is_facing_bet": float(facing_amount_bb > 0.0),
        "is_facing_all_in": float(context.is_facing_all_in),
        "can_check": float(context.can_check),
        "can_raise": float(context.can_raise),
        "keras_equity": (
            _safe_keras_equity(hole, board, max(1, active_players - 1))
            if include_keras_equity
            else math.nan
        ),
        **_board_features(board),
    }


def feature_vector(features: Mapping[str, object]) -> list[object]:
    return [features.get(name, 0.0) for name in FEATURE_NAMES]


def all_live_combos(state: PokerTableState) -> dict[tuple[Card, Card], float]:
    dead_cards = [*state.hero_cards, *state.community_cards]
    return uniform_combo_range(dead_cards)


def uniform_combo_range(dead_cards: Iterable[Card]) -> dict[tuple[Card, Card], float]:
    combos = list(_available_combos(dead_cards))
    if not combos:
        return {}
    probability = 1.0 / len(combos)
    return {combo: probability for combo in combos}


def update_combo_range(
    prior: Mapping[tuple[Card, Card], float],
    observed_action_bucket: str,
    state: PokerTableState,
    player: PlayerSnapshot,
    model,
    context: ActionContext | None = None,
    min_likelihood: float = 1e-6,
) -> dict[tuple[Card, Card], float]:
    if observed_action_bucket not in ACTION_BUCKETS:
        raise ValueError(f"Unknown action bucket: {observed_action_bucket}")

    posterior = {}
    for combo, prior_probability in prior.items():
        features = build_action_features(state, player, combo, context)
        action_probabilities = model.action_probabilities(features)
        likelihood = max(float(action_probabilities.get(observed_action_bucket, 0.0)), min_likelihood)
        posterior[combo] = prior_probability * likelihood

    return normalize_combo_range(posterior)


def normalize_combo_range(combo_range: Mapping[tuple[Card, Card], float]) -> dict[tuple[Card, Card], float]:
    total = sum(max(0.0, probability) for probability in combo_range.values())
    if total <= 0.0:
        return uniform_from_combos(combo_range)
    return {
        combo: max(0.0, probability) / total
        for combo, probability in combo_range.items()
    }


def uniform_from_combos(combo_range: Mapping[tuple[Card, Card], float]) -> dict[tuple[Card, Card], float]:
    if not combo_range:
        return {}
    probability = 1.0 / len(combo_range)
    return {combo: probability for combo in combo_range}


def remove_dead_card_combos(
    combo_range: Mapping[tuple[Card, Card], float],
    dead_cards: Iterable[Card],
) -> dict[tuple[Card, Card], float]:
    dead = {str(card) for card in dead_cards}
    live_range = {
        combo: probability
        for combo, probability in combo_range.items()
        if str(combo[0]) not in dead and str(combo[1]) not in dead
    }
    return normalize_combo_range(live_range)


def sample_combo(
    combo_range: Mapping[tuple[Card, Card], float],
    rng: random.Random | None = None,
) -> tuple[Card, Card] | None:
    if not combo_range:
        return None

    rng = rng or random
    combos = list(combo_range)
    weights = [max(0.0, combo_range[combo]) for combo in combos]
    if sum(weights) <= 0.0:
        weights = [1.0] * len(combos)
    return rng.choices(combos, weights=weights, k=1)[0]


def sample_combos_without_overlap(
    player_ranges: Mapping[str, Mapping[tuple[Card, Card], float]],
    dead_cards: Iterable[Card],
    rng: random.Random | None = None,
) -> dict[str, tuple[Card, Card]]:
    rng = rng or random
    sampled = {}
    dead = set(str(card) for card in dead_cards)

    for player_name, combo_range in player_ranges.items():
        live_range = remove_dead_card_combos(combo_range, [Card(card[:-1], card[-1]) for card in dead])
        combo = sample_combo(live_range, rng)
        if combo is None:
            continue
        sampled[player_name] = combo
        dead.update(str(card) for card in combo)

    return sampled


def _available_combos(dead_cards: Iterable[Card]) -> Iterable[tuple[Card, Card]]:
    dead = {str(card) for card in dead_cards}
    deck = [
        Card(rank=rank, suit=suit)
        for suit in SUITS
        for rank in RANKS
        if f"{rank}{suit}" not in dead
    ]
    yield from combinations(deck, 2)


def _street_from_board(board: list[Card]) -> str:
    return {
        0: "preflop",
        3: "flop",
        4: "turn",
        5: "river",
    }.get(len(board), "unknown")


def _effective_stack_bb(state: PokerTableState, player: PlayerSnapshot) -> float:
    player_stack = player.stack_bb or 0.0
    opponent_stacks = [
        opponent.stack_bb or 0.0
        for opponent in state.players
        if opponent.name != player.name and opponent.status not in {"folded", "offline"}
    ]
    if not opponent_stacks:
        return player_stack
    return min(player_stack, max(opponent_stacks))


def _position_bucket(state: PokerTableState, player: PlayerSnapshot) -> str:
    active_players = [p for p in state.players if p.status not in {"folded", "offline"}]
    if len(active_players) <= 2:
        if player.is_dealer:
            return "heads_up_button"
        return "heads_up_big_blind"
    if player.is_dealer:
        return "button"
    small_blind_seat = _next_active_seat(active_players, state.dealer_seat_index)
    big_blind_seat = _next_active_seat(active_players, small_blind_seat)
    if player.seat_index == small_blind_seat:
        return "small_blind"
    if player.seat_index == big_blind_seat:
        return "big_blind"

    fraction = _position_fraction(state, player)
    if fraction < 0.34:
        return "early"
    if fraction < 0.67:
        return "middle"
    return "late"


def _position_fraction(state: PokerTableState, player: PlayerSnapshot) -> float:
    active_players = sorted(
        [p for p in state.players if p.status not in {"folded", "offline"}],
        key=lambda value: value.seat_index,
    )
    if len(active_players) <= 1:
        return 0.0
    ordered = _ordered_after_dealer(active_players, state.dealer_seat_index)
    seat_order = [p.seat_index for p in ordered]
    try:
        return seat_order.index(player.seat_index) / (len(seat_order) - 1)
    except ValueError:
        return 0.0


def _ordered_after_dealer(players: list[PlayerSnapshot], dealer_seat: int | None) -> list[PlayerSnapshot]:
    if dealer_seat is None:
        return players
    return sorted(players, key=lambda player: ((player.seat_index - dealer_seat) % 100))


def _next_active_seat(players: list[PlayerSnapshot], seat_index: int | None) -> int | None:
    if seat_index is None or not players:
        return None
    seats = sorted(player.seat_index for player in players)
    for seat in seats:
        if seat > seat_index:
            return seat
    return seats[0]


def _stack_rank_fraction(state: PokerTableState, player: PlayerSnapshot) -> float:
    candidates = [p for p in state.players if p.status != "offline"]
    if player not in candidates:
        candidates.append(player)
    if len(candidates) <= 1:
        return 1.0
    sorted_stacks = sorted((p.stack_bb or 0.0 for p in candidates))
    player_stack = player.stack_bb or 0.0
    rank = min(range(len(sorted_stacks)), key=lambda index: abs(sorted_stacks[index] - player_stack))
    return rank / (len(sorted_stacks) - 1)


def _board_features(board: list[Card]) -> dict[str, float]:
    rank_counts = {}
    suit_counts = {}
    for card in board:
        rank_counts[RANK_VALUES[card.rank]] = rank_counts.get(RANK_VALUES[card.rank], 0) + 1
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

    high_rank = max(rank_counts, default=0)
    max_suit = max(suit_counts.values(), default=0)
    return {
        "board_card_count": len(board),
        "board_paired": float(any(count >= 2 for count in rank_counts.values())),
        "board_two_pair_or_better": float(sum(1 for count in rank_counts.values() if count >= 2) >= 2),
        "board_trips_or_better": float(any(count >= 3 for count in rank_counts.values())),
        "board_max_suit_fraction": min(max_suit, 5) / 5.0,
        "board_three_flush": float(max_suit >= 3),
        "board_four_flush": float(max_suit >= 4),
        "board_made_flush": float(max_suit >= 5),
        "board_straight_run_fraction": _max_straight_run_length(rank_counts) / 5.0,
        "board_high_rank": high_rank / 14.0 if high_rank else 0.0,
        "board_ace_high": float(high_rank == 14),
    }


def _max_straight_run_length(rank_counts: Mapping[int, int]) -> int:
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


def _safe_keras_equity(hole: list[Card], board: list[Card], opponent_count: int) -> float:
    global _KERAS_EQUITY_AVAILABLE
    if not _KERAS_EQUITY_AVAILABLE:
        return math.nan
    try:
        return predict_equity(hole, board, opponent_count)
    except Exception:
        _KERAS_EQUITY_AVAILABLE = False
        return math.nan

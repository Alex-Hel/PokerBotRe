from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
import random
from pathlib import Path
from typing import Iterable, Mapping

from models.EquityModel import predict_equity, predict_equity_batch
from PokerState import Card, PlayerSnapshot, PokerEvent, PokerTableState, RANKS, RANK_VALUES, SUITS


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

RAISE_BUCKET_RATIOS = (
    ("half_pot_raise", 0.5),
    ("three_quarter_pot_raise", 0.75),
    ("pot_raise", 1.0),
)

POSTFLOP_FEATURE_NAMES = (
    "card_bucket",
    "suitedness",
    "paired",
    "street",
    "players_left_fraction",
    "players_in_hand_fraction",
    "seat_bucket",
    "stack_rank_fraction",
    "stack_bb",
    "pot_bb",
    "pot_stack_fraction",
    "facing_amount_bb",
    "facing_amount_stack_fraction",
    "amount_put_in_pot_bb",
    "amount_put_in_pot_stack_fraction",
    "call_ev",
    "stack_to_pot_ratio",
    "previous_raises_by_self_in_hand",
    "previous_raises_in_hand",
    "last_aggressor_is_self",
    "can_check",
    "keras_equity",
    "board_max_rank_count",
    "board_two_pair_or_better",
    "personal_flush_fraction",
    "board_flush_fraction",
    "personal_straight_fraction",
    "board_straight_fraction",
    "has_highest_pair",
)

PREFLOP_FEATURE_NAMES = (
    "facing_amount_bb",
    "stack_bb",
    "facing_amount_stack_fraction",
    "card_bucket",
    "suitedness",
    "keras_equity",
    "seat_bucket",
    "call_ev",
    "paired",
    "pot_bb",
    "players_left_fraction",
    "players_in_hand_fraction",
    "previous_raises_in_hand",
    "previous_raises_by_self_in_hand",
    "stack_to_pot_ratio",
    "amount_put_in_pot_bb",
    "amount_put_in_pot_stack_fraction",
    "can_check",
)

FEATURE_NAMES = POSTFLOP_FEATURE_NAMES
ALL_INTERPRETED_FEATURE_NAMES = tuple(dict.fromkeys((*POSTFLOP_FEATURE_NAMES, *PREFLOP_FEATURE_NAMES)))
DEFAULT_PREFLOP_MODEL_PATH = Path(__file__).with_name("preflop_ranging_action_model.cbm")
DEFAULT_POSTFLOP_MODEL_PATH = Path(__file__).with_name("postflop_ranging_action_model.cbm")

_KERAS_EQUITY_AVAILABLE = True


@dataclass(frozen=True)
class ActionContext:
    facing_amount_bb: float = 0.0
    previous_raises_by_self_in_hand: int = 0
    previous_raises_in_hand: int = 0
    last_aggressor_name: str | None = None
    can_check: bool = False


@dataclass(frozen=True)
class RangeUpdate:
    player_name: str
    action_bucket: str
    average_equity: float | None
    combo_count: int


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

    def action_probabilities_batch(self, feature_rows: Iterable[Mapping[str, object]]) -> list[dict[str, float]]:
        rows = [[features.get(name, 0.0) for name in self.feature_names] for features in feature_rows]
        if not rows:
            return []

        probabilities_by_row = self.model.predict_proba(rows)
        results = []
        for probabilities in probabilities_by_row:
            result = {bucket: 0.0 for bucket in ACTION_BUCKETS}
            for action_name, probability in zip(self.classes, probabilities):
                if action_name in result:
                    result[action_name] = float(probability)
            results.append(result)
        return results

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
    if normalized in ACTION_BUCKETS:
        return normalized
    if is_all_in or normalized in {"jam", "shove", "all_in"}:
        return "all_in"

    amount = max(0.0, amount_bb or 0.0)
    pot = max(0.0, pot_bb or 0.0)
    if amount <= 1.0 or pot <= 0.0:
        return "min_raise"

    ratio = amount / pot
    if ratio < 0.25:
        return "min_raise"
    return min(
        RAISE_BUCKET_RATIOS,
        key=lambda bucket: abs(ratio - bucket[1]),
    )[0]


def build_action_features(
    state: PokerTableState,
    player: PlayerSnapshot,
    hole_cards: Iterable[Card],
    context: ActionContext | None = None,
    include_keras_equity: bool = True,
) -> dict[str, object]:
    context = context or ActionContext()
    hole = list(hole_cards)
    players_left = len(state.players)
    starting_players = state.starting_player_count or players_left or 1
    active_players = len([player for player in state.players if player.status not in {"folded", "offline"}])
    stack_bb = player.stack_bb or 0.0
    pot_bb = state.pot_bb or 0.0
    facing_amount_bb = max(0.0, context.facing_amount_bb)
    board = state.community_cards
    keras_equity = (
        _safe_keras_equity(hole, board, max(1, active_players - 1))
        if include_keras_equity
        else math.nan
    )

    return {
        **_hole_bucket_features(hole),
        "street": _street_from_board(board),
        "players_left_fraction": players_left / starting_players,
        "players_in_hand_fraction": active_players / max(players_left, 1),
        "seat_bucket": _seat_bucket(state, player),
        "stack_rank_fraction": _stack_rank_fraction(state, player),
        "stack_bb": stack_bb,
        "pot_bb": pot_bb,
        "pot_stack_fraction": _stack_fraction(pot_bb, stack_bb),
        "facing_amount_bb": facing_amount_bb,
        "facing_amount_stack_fraction": _stack_fraction(facing_amount_bb, stack_bb),
        "amount_put_in_pot_bb": player.bet_bb or 0.0,
        "amount_put_in_pot_stack_fraction": _stack_fraction(player.bet_bb or 0.0, stack_bb),
        "call_ev": _call_ev(keras_equity, facing_amount_bb, pot_bb),
        "stack_to_pot_ratio": stack_bb / pot_bb if pot_bb else 0.0,
        "previous_raises_by_self_in_hand": context.previous_raises_by_self_in_hand,
        "previous_raises_in_hand": context.previous_raises_in_hand,
        "last_aggressor_is_self": float(context.last_aggressor_name == player.name),
        "can_check": float(context.can_check),
        "keras_equity": keras_equity,
        **_board_features(hole, board),
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

    combos = list(prior)
    feature_rows = _build_combo_feature_rows(state, player, combos, context)
    action_probabilities_by_combo = (
        model.action_probabilities_batch(feature_rows)
        if hasattr(model, "action_probabilities_batch")
        else [model.action_probabilities(features) for features in feature_rows]
    )

    posterior = {}
    for combo, action_probabilities in zip(combos, action_probabilities_by_combo):
        prior_probability = prior[combo]
        likelihood = max(float(action_probabilities.get(observed_action_bucket, 0.0)), min_likelihood)
        posterior[combo] = prior_probability * likelihood

    return normalize_combo_range(posterior)


def _build_combo_feature_rows(
    state: PokerTableState,
    player: PlayerSnapshot,
    combos: list[tuple[Card, Card]],
    context: ActionContext | None,
) -> list[dict[str, object]]:
    active_players = len([value for value in state.players if value.status not in {"folded", "offline"}])
    opponent_count = max(1, active_players - 1)
    rows = [
        build_action_features(
            state,
            player,
            combo,
            context,
            include_keras_equity=False,
        )
        for combo in combos
    ]

    equities = _safe_keras_equity_batch(combos, state.community_cards, opponent_count)
    if equities is None:
        return rows

    call_bb = max(0.0, (context or ActionContext()).facing_amount_bb)
    pot_bb = state.pot_bb or 0.0
    for row, equity in zip(rows, equities):
        row["keras_equity"] = equity
        row["call_ev"] = _call_ev(equity, call_bb, pot_bb)
    return rows


class LiveRangingTracker:
    def __init__(
        self,
        preflop_model_path: str | Path = DEFAULT_PREFLOP_MODEL_PATH,
        postflop_model_path: str | Path = DEFAULT_POSTFLOP_MODEL_PATH,
        min_likelihood: float = 1e-6,
    ) -> None:
        self.preflop_model = self._load_model(preflop_model_path)
        self.postflop_model = self._load_model(postflop_model_path)
        self.min_likelihood = min_likelihood
        self.player_ranges: dict[str, dict[tuple[Card, Card], float]] = {}
        self.last_updates: list[RangeUpdate] = []
        self.previous_raises_in_hand = 0
        self.previous_raises_by_player: dict[str, int] = {}
        self.last_aggressor_name: str | None = None

    @property
    def available(self) -> bool:
        return self.preflop_model is not None or self.postflop_model is not None

    def process_snapshot(
        self,
        previous: PokerTableState | None,
        current: PokerTableState,
        events: Iterable[PokerEvent],
    ) -> None:
        self.last_updates = []
        if previous is None or self._is_new_hand(previous, current):
            self.reset()

        if previous is not None:
            self._sync_ranges(previous)
            event_list = list(events)
            explicit_actor_names = {
                event.player_name
                for event in event_list
                if event.event_type in {"player_folded", "player_called", "player_raised", "player_bet_changed"}
            }
            for event in event_list:
                self._process_action_event(previous, current, event)
            self._process_inferred_check(previous, current, explicit_actor_names)

        self._sync_ranges(current)

    def reset(self) -> None:
        self.player_ranges = {}
        self.previous_raises_in_hand = 0
        self.previous_raises_by_player = {}
        self.last_aggressor_name = None

    def _process_action_event(
        self,
        previous: PokerTableState,
        current: PokerTableState,
        event: PokerEvent,
    ) -> None:
        if event.player_name is None:
            return

        previous_player = _find_player(previous, event.player_name)
        current_player = _find_player(current, event.player_name)
        if previous_player is None or previous_player.is_hero:
            return
        if previous.current_player_name and event.player_name != previous.current_player_name:
            return

        action_bucket = self._event_action_bucket(previous, previous_player, current_player, event)
        if action_bucket is None:
            return

        model = self._model_for_state(previous)
        if model is None:
            return

        player_name = previous_player.name or f"seat_{previous_player.seat_index}"
        prior = self.player_ranges.get(player_name)
        if not prior:
            prior = all_live_combos(previous)

        facing_amount_bb = _facing_amount_bb(previous, previous_player)
        context = ActionContext(
            facing_amount_bb=facing_amount_bb,
            previous_raises_by_self_in_hand=self.previous_raises_by_player.get(player_name, 0),
            previous_raises_in_hand=self.previous_raises_in_hand,
            last_aggressor_name=self.last_aggressor_name,
            can_check=facing_amount_bb <= 0.0,
        )
        self.player_ranges[player_name] = update_combo_range(
            prior,
            action_bucket,
            previous,
            previous_player,
            model,
            context,
            min_likelihood=self.min_likelihood,
        )
        updated_range = self.player_ranges[player_name]
        self.last_updates.append(
            RangeUpdate(
                player_name=player_name,
                action_bucket=action_bucket,
                average_equity=average_combo_range_equity(updated_range, previous),
                combo_count=len(updated_range),
            )
        )

        if action_bucket in {"min_raise", "half_pot_raise", "three_quarter_pot_raise", "pot_raise", "all_in"}:
            self.previous_raises_in_hand += 1
            self.previous_raises_by_player[player_name] = self.previous_raises_by_player.get(player_name, 0) + 1
            self.last_aggressor_name = player_name

    def _process_inferred_check(
        self,
        previous: PokerTableState,
        current: PokerTableState,
        explicit_actor_names: set[str | None],
    ) -> None:
        actor_name = previous.current_player_name
        if (
            actor_name is None
            or actor_name == current.current_player_name
            or actor_name in explicit_actor_names
        ):
            return

        actor = _find_player(previous, actor_name)
        if actor is None or actor.is_hero or _facing_amount_bb(previous, actor) > 0.0:
            return

        self._process_action_event(
            previous,
            current,
            PokerEvent(
                event_type="player_checked",
                timestamp=current.timestamp,
                player_name=actor_name,
                description=f"{actor_name} checked",
            ),
        )

    def _sync_ranges(self, state: PokerTableState) -> None:
        live_names = set()
        dead_cards = [*state.hero_cards, *state.community_cards]

        for player in state.players:
            if player.is_hero or player.status == "offline":
                player.hole_combo_range = {}
                player.hole_card_distribution = {}
                continue

            player_name = player.name or f"seat_{player.seat_index}"
            live_names.add(player_name)
            if len(player.cards) >= 2:
                combo = _canonical_combo(player.cards[:2])
                combo_range = {combo: 1.0}
            else:
                combo_range = self.player_ranges.get(player_name)
                if combo_range:
                    # Keep the existing posterior across streets; only remove newly dead cards.
                    combo_range = remove_dead_card_combos(combo_range, dead_cards)
                else:
                    combo_range = uniform_combo_range(dead_cards)

            self.player_ranges[player_name] = combo_range
            player.hole_combo_range = dict(combo_range)
            player.hole_card_distribution = _combo_distribution_for_display(combo_range)

        stale_names = set(self.player_ranges) - live_names
        for player_name in stale_names:
            del self.player_ranges[player_name]
            self.previous_raises_by_player.pop(player_name, None)

    def _event_action_bucket(
        self,
        previous: PokerTableState,
        previous_player: PlayerSnapshot,
        current_player: PlayerSnapshot | None,
        event: PokerEvent,
    ) -> str | None:
        if event.event_type == "player_checked":
            return "check"
        if event.event_type == "player_folded":
            return "fold"
        if event.event_type == "player_called":
            return "call"
        if event.event_type not in {"player_raised", "player_bet_changed"}:
            return None

        previous_bet_bb = previous_player.bet_bb or 0.0
        current_bet_bb = (
            current_player.bet_bb
            if current_player and current_player.bet_bb is not None
            else event.amount_bb
        ) or previous_bet_bb
        added_amount_bb = max(0.0, current_bet_bb - previous_bet_bb)
        is_all_in = bool(
            current_player
            and (
                current_player.status == "all_in"
                or (current_player.stack_bb is not None and current_player.stack_bb <= 0.0)
            )
        )
        if event.event_type == "player_bet_changed":
            facing_amount_bb = _facing_amount_bb(previous, previous_player)
            previous_max_bet_bb = max((player.bet_bb or 0.0 for player in previous.players), default=0.0)
            if facing_amount_bb > 0.0 and current_bet_bb <= previous_max_bet_bb:
                return "all_in" if is_all_in else "call"
            return "check" if added_amount_bb <= 0.0 else classify_action_bucket(
                "raise",
                amount_bb=added_amount_bb,
                pot_bb=previous.pot_bb,
                is_all_in=is_all_in,
            )
        return classify_action_bucket(
            "raise",
            amount_bb=added_amount_bb,
            pot_bb=previous.pot_bb,
            is_all_in=is_all_in,
        )

    def _model_for_state(self, state: PokerTableState):
        if _street_from_board(state.community_cards) == "preflop":
            return self.preflop_model
        return self.postflop_model

    def _load_model(self, model_path: str | Path):
        path = Path(model_path)
        if not path.exists():
            return None
        return CatBoostActionModel(path)

    def _is_new_hand(self, previous: PokerTableState, current: PokerTableState) -> bool:
        previous_hero = tuple(str(card) for card in previous.hero_cards)
        current_hero = tuple(str(card) for card in current.hero_cards)
        if previous_hero and current_hero and previous_hero != current_hero:
            return True
        if len(current.community_cards) < len(previous.community_cards):
            return True
        if (
            previous.pot_bb is not None
            and current.pot_bb is not None
            and current.pot_bb < previous.pot_bb
            and not current.community_cards
        ):
            return True
        return False


def normalize_combo_range(combo_range: Mapping[tuple[Card, Card], float]) -> dict[tuple[Card, Card], float]:
    total = sum(max(0.0, probability) for probability in combo_range.values())
    if total <= 0.0:
        return uniform_from_combos(combo_range)
    return {
        combo: max(0.0, probability) / total
        for combo, probability in combo_range.items()
    }


def average_combo_range_equity(
    combo_range: Mapping[tuple[Card, Card], float],
    state: PokerTableState,
) -> float | None:
    combos = list(combo_range)
    if not combos:
        return None

    active_players = len([player for player in state.players if player.status not in {"folded", "offline"}])
    equities = _safe_keras_equity_batch(combos, state.community_cards, max(1, active_players - 1))
    if equities is None:
        return None

    return sum(
        max(0.0, combo_range[combo]) * equity
        for combo, equity in zip(combos, equities)
    )


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


def _combo_distribution_for_display(combo_range: Mapping[tuple[Card, Card], float]) -> dict[str, float]:
    return {
        f"{combo[0]}{combo[1]}": probability
        for combo, probability in sorted(
            combo_range.items(),
            key=lambda item: (-item[1], str(item[0][0]), str(item[0][1])),
        )
    }


def _canonical_combo(cards: Iterable[Card]) -> tuple[Card, Card]:
    first, second = sorted(list(cards)[:2], key=lambda card: (RANK_VALUES[card.rank], card.suit), reverse=True)
    return first, second


def _find_player(state: PokerTableState, player_name: str) -> PlayerSnapshot | None:
    return next((player for player in state.players if player.name == player_name), None)


def _facing_amount_bb(state: PokerTableState, player: PlayerSnapshot) -> float:
    max_bet = max((value.bet_bb or 0.0 for value in state.players), default=0.0)
    return max(0.0, max_bet - (player.bet_bb or 0.0))


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


def _hole_bucket_features(hole: list[Card]) -> dict[str, object]:
    ordered = sorted(hole, key=lambda card: RANK_VALUES[card.rank], reverse=True)
    return {
        "card_bucket": f"{ordered[0].rank}{ordered[1].rank}",
        "suitedness": "suited" if ordered[0].suit == ordered[1].suit else "offsuit",
        "paired": float(ordered[0].rank == ordered[1].rank),
    }


def _call_ev(keras_equity: float, call_bb: float, pot_bb: float) -> float:
    if math.isnan(keras_equity):
        return 0.0
    return keras_equity * (pot_bb + call_bb) - call_bb


def _stack_fraction(amount_bb: float, stack_bb: float) -> float:
    if stack_bb <= 0.0:
        return 0.0
    return amount_bb / stack_bb


def _seat_bucket(state: PokerTableState, player: PlayerSnapshot) -> str:
    seated_players = [p for p in state.players if p.status != "offline"]
    if len(seated_players) <= 2:
        if player.is_dealer:
            return "heads_up_button"
        return "heads_up_big_blind"
    if player.is_dealer:
        return "button"
    small_blind_seat = _next_active_seat(seated_players, state.dealer_seat_index)
    big_blind_seat = _next_active_seat(seated_players, small_blind_seat)
    if player.seat_index == small_blind_seat:
        return "small_blind"
    if player.seat_index == big_blind_seat:
        return "big_blind"
    return "middle"


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


def _board_features(hole: list[Card], board: list[Card]) -> dict[str, float]:
    rank_counts = {}
    for card in board:
        rank_counts[RANK_VALUES[card.rank]] = rank_counts.get(RANK_VALUES[card.rank], 0) + 1

    return {
        "board_max_rank_count": max(rank_counts.values(), default=0),
        "board_two_pair_or_better": float(sum(1 for count in rank_counts.values() if count >= 2) >= 2),
        "personal_flush_fraction": _flush_fraction([*hole, *board], len(board)),
        "board_flush_fraction": _flush_fraction(board, len(board)),
        "personal_straight_fraction": _straight_fraction([*hole, *board], len(board)),
        "board_straight_fraction": _straight_fraction(board, len(board)),
        "has_highest_pair": _has_highest_pair(hole, board),
    }


def _flush_fraction(cards: list[Card], board_card_count: int) -> float:
    suit_counts = {}
    for card in cards:
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

    cards_to_draw = max(0, 5 - board_card_count)
    max_suit_count = max(suit_counts.values(), default=0)
    if cards_to_draw == 0:
        return 0.0 if max_suit_count == 0 else 99.0
    return max_suit_count / cards_to_draw


def _straight_fraction(cards: list[Card], board_card_count: int) -> float:
    ranks = {RANK_VALUES[card.rank] for card in cards}
    if 14 in ranks:
        ranks.add(1)

    straight_windows = [
        {1, 2, 3, 4, 5},
        {2, 3, 4, 5, 6},
        {3, 4, 5, 6, 7},
        {4, 5, 6, 7, 8},
        {5, 6, 7, 8, 9},
        {6, 7, 8, 9, 10},
        {7, 8, 9, 10, 11},
        {8, 9, 10, 11, 12},
        {9, 10, 11, 12, 13},
        {10, 11, 12, 13, 14},
    ]
    cards_off = min((len(window - ranks) for window in straight_windows), default=5)
    cards_to_draw = max(0, 5 - board_card_count)
    if cards_to_draw == 0:
        return 0.0 if cards_off == 0 else 99.0
    return cards_off / cards_to_draw


def _has_highest_pair(hole: list[Card], board: list[Card]) -> float:
    rank_counts = {}
    for card in [*hole, *board]:
        rank_counts[RANK_VALUES[card.rank]] = rank_counts.get(RANK_VALUES[card.rank], 0) + 1

    paired_ranks = {rank for rank, count in rank_counts.items() if count >= 2}
    if not paired_ranks:
        return 0.0

    highest_pair_rank = max(paired_ranks)
    return float(any(RANK_VALUES[card.rank] == highest_pair_rank for card in hole))


def _safe_keras_equity(hole: list[Card], board: list[Card], opponent_count: int) -> float:
    global _KERAS_EQUITY_AVAILABLE
    if not _KERAS_EQUITY_AVAILABLE:
        return math.nan
    try:
        return predict_equity(hole, board, opponent_count)
    except Exception:
        _KERAS_EQUITY_AVAILABLE = False
        return math.nan


def _safe_keras_equity_batch(
    combos: Iterable[tuple[Card, Card]],
    board: list[Card],
    opponent_count: int,
) -> list[float] | None:
    global _KERAS_EQUITY_AVAILABLE
    if not _KERAS_EQUITY_AVAILABLE:
        return None
    try:
        return predict_equity_batch(
            ((combo, board, opponent_count) for combo in combos),
            batch_size=4096,
        )
    except Exception:
        _KERAS_EQUITY_AVAILABLE = False
        return None

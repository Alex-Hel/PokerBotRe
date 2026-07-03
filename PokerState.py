from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import random
from time import time
import re


RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUITS = ("c", "d", "h", "s")
RANK_VALUES = {rank: index + 2 for index, rank in enumerate(RANKS)}


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str
    raw: str = ""

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


@dataclass
class PokerEvent:
    event_type: str
    timestamp: float
    player_name: str | None = None
    amount: int | None = None
    amount_bb: float | None = None
    description: str = ""


@dataclass
class ActionSnapshot:
    name: str
    text: str
    amount: int | None = None
    amount_bb: float | None = None
    enabled: bool = True


@dataclass
class PlayerSnapshot:
    seat_index: int
    name: str
    stack: int | None = None
    stack_bb: float | None = None
    bet: int | None = None
    bet_bb: float | None = None
    status: str = "unknown"
    cards: list[Card] = field(default_factory=list)
    is_hero: bool = False
    is_current: bool = False
    is_dealer: bool = False
    is_offline: bool = False
    has_turn_timer: bool = False
    time_bank_percent: float | None = None
    normal_time_percent: float | None = None
    hole_card_class_combos: dict[str, int] = field(default_factory=dict)
    hole_card_class_distribution: dict[str, float] = field(default_factory=dict)
    hole_combo_range: dict[tuple[Card, Card], float] = field(default_factory=dict)
    hole_card_distribution: dict[str, float] = field(default_factory=dict)


@dataclass
class PokerTableState:
    timestamp: float
    url: str
    game_type: str = ""
    pot: int | None = None
    pot_bb: float | None = None
    blinds: list[int] = field(default_factory=list)
    starting_player_count: int | None = None
    community_cards: list[Card] = field(default_factory=list)
    hero_cards: list[Card] = field(default_factory=list)
    players: list[PlayerSnapshot] = field(default_factory=list)
    current_player_name: str | None = None
    dealer_seat_index: int | None = None
    visible_actions: list[ActionSnapshot] = field(default_factory=list)
    available_actions: list[ActionSnapshot] = field(default_factory=list)
    remaining_deck: list[Card] = field(default_factory=list)
    was_hero_turn: bool = False

    @property
    def hero(self) -> PlayerSnapshot | None:
        return next((player for player in self.players if player.is_hero), None)

    @property
    def is_hero_turn(self) -> bool:
        return self.was_hero_turn

    @property
    def enabled_actions(self) -> dict[str, ActionSnapshot]:
        return {
            action.name: action
            for action in self.available_actions
            if action.enabled
        }

    @property
    def active_opponents(self) -> list[PlayerSnapshot]:
        return [
            player
            for player in self.players
            if not player.is_hero and player.status != "folded"
        ]

    @property
    def opponent_hole_card_class_distributions(self) -> dict[str, dict[str, float]]:
        return {
            player.name or f"seat_{player.seat_index}": player.hole_card_class_distribution
            for player in self.players
            if not player.is_hero
        }

    @property
    def small_blind(self) -> int | None:
        return self.blinds[0] if self.blinds else None

    @property
    def big_blind(self) -> int | None:
        return self.blinds[-1] if self.blinds else None

    def monte_carlo(
        self,
        simulations: int = 250,
        include_tie_equity: bool = True,
        seed: int | None = None,
    ) -> float:
        if simulations <= 0 or len(self.hero_cards) < 2:
            return 0.0

        opponents = self.active_opponents
        if not opponents:
            return 1.0

        rng = random.Random(seed)
        hero_score_total = 0.0
        completed_simulations = 0

        for _ in range(simulations):
            deck = list(self.remaining_deck)
            rng.shuffle(deck)

            board = list(self.community_cards)
            opponent_hands = []

            for opponent in opponents:
                hand = list(opponent.cards)
                if len(hand) >= 2:
                    hand = hand[:2]
                    _remove_cards_from_deck(deck, hand)
                else:
                    sampled = _sample_combo_from_range(opponent.hole_combo_range, deck, rng)
                    if sampled is not None:
                        hand = list(sampled)
                        _remove_cards_from_deck(deck, hand)
                    else:
                        while len(hand) < 2 and deck:
                            hand.append(deck.pop())
                if len(hand) < 2:
                    break
                opponent_hands.append(hand[:2])

            if len(opponent_hands) != len(opponents):
                continue

            while len(board) < 5 and deck:
                board.append(deck.pop())

            if len(board) < 5:
                continue

            hero_score = best_holdem_score([*self.hero_cards[:2], *board])
            opponent_scores = [
                best_holdem_score([*hand, *board])
                for hand in opponent_hands
            ]
            best_opponent_score = max(opponent_scores)

            completed_simulations += 1
            if hero_score > best_opponent_score:
                hero_score_total += 1.0
            elif include_tie_equity and hero_score == best_opponent_score:
                tied_opponents = sum(1 for score in opponent_scores if score == hero_score)
                hero_score_total += 1.0 / (tied_opponents + 1)

        if completed_simulations == 0:
            return 0.0

        return hero_score_total / completed_simulations


def _sample_combo_from_range(
    combo_range: dict[tuple[Card, Card], float],
    deck: list[Card],
    rng: random.Random,
) -> tuple[Card, Card] | None:
    if not combo_range:
        return None

    available = {str(card) for card in deck}
    candidates = []
    weights = []
    for combo, probability in combo_range.items():
        first, second = combo
        if str(first) not in available or str(second) not in available:
            continue
        candidates.append((first, second))
        weights.append(max(0.0, probability))

    if not candidates:
        return None
    if sum(weights) <= 0.0:
        weights = [1.0] * len(candidates)
    return rng.choices(candidates, weights=weights, k=1)[0]


def _remove_cards_from_deck(deck: list[Card], cards: list[Card]) -> None:
    used = {str(card) for card in cards}
    deck[:] = [card for card in deck if str(card) not in used]


def best_holdem_score(cards: list[Card]) -> tuple:
    return max(evaluate_five_card_hand(list(hand)) for hand in combinations(cards, 5))


def evaluate_five_card_hand(cards: list[Card]) -> tuple:
    values = sorted((RANK_VALUES[card.rank] for card in cards), reverse=True)
    suits = [card.suit for card in cards]
    counts = {value: values.count(value) for value in set(values)}
    count_groups = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)

    is_flush = len(set(suits)) == 1
    straight_high = get_straight_high(values)

    if is_flush and straight_high:
        return 8, straight_high

    if count_groups[0][1] == 4:
        quad = count_groups[0][0]
        kicker = max(value for value in values if value != quad)
        return 7, quad, kicker

    if count_groups[0][1] == 3 and count_groups[1][1] == 2:
        return 6, count_groups[0][0], count_groups[1][0]

    if is_flush:
        return 5, *values

    if straight_high:
        return 4, straight_high

    if count_groups[0][1] == 3:
        trips = count_groups[0][0]
        kickers = sorted((value for value in values if value != trips), reverse=True)
        return 3, trips, *kickers

    pairs = [value for value, count in count_groups if count == 2]
    if len(pairs) == 2:
        high_pair, low_pair = sorted(pairs, reverse=True)
        kicker = max(value for value in values if value not in pairs)
        return 2, high_pair, low_pair, kicker

    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted((value for value in values if value != pair), reverse=True)
        return 1, pair, *kickers

    return 0, *values


def get_straight_high(values: list[int]) -> int | None:
    unique_values = sorted(set(values), reverse=True)
    if 14 in unique_values:
        unique_values.append(1)

    for index in range(len(unique_values) - 4):
        window = unique_values[index:index + 5]
        if window[0] - window[4] == 4:
            return window[0]

    return None


def hole_card_class(cards: list[Card]) -> str | None:
    if len(cards) < 2:
        return None

    first, second = cards[:2]
    ranks = sorted((first.rank, second.rank), key=lambda rank: RANK_VALUES[rank], reverse=True)
    if ranks[0] == ranks[1]:
        return f"{ranks[0]}{ranks[1]}"

    suited_marker = "s" if first.suit == second.suit else "o"
    return f"{ranks[0]}{ranks[1]}{suited_marker}"


def hole_card_class_combos(cards: list[Card]) -> dict[str, int]:
    combos: dict[str, int] = {}
    for first, second in combinations(cards, 2):
        hand_class = hole_card_class([first, second])
        if hand_class is not None:
            combos[hand_class] = combos.get(hand_class, 0) + 1

    return dict(sorted(combos.items(), key=lambda item: hole_card_class_sort_key(item[0])))


def hole_card_class_distribution(cards: list[Card]) -> dict[str, float]:
    combos = hole_card_class_combos(cards)
    total = sum(combos.values())
    if total == 0:
        return {}
    return {hand_class: count / total for hand_class, count in combos.items()}


def hole_card_class_sort_key(hand_class: str) -> tuple:
    high = RANK_VALUES.get(hand_class[0], 0)
    low = RANK_VALUES.get(hand_class[1], high)
    pair_first = 0 if len(hand_class) == 2 else 1
    suited_first = 0 if hand_class.endswith("s") else 1
    return pair_first, -high, -low, suited_first


class PokerTableScraper:
    def __init__(self, driver, hero_name: str | None = None) -> None:
        self.driver = driver
        self.hero_name = hero_name
        self.full_deck = [Card(rank, suit) for suit in SUITS for rank in RANKS]

    def scrape(self) -> PokerTableState:
        data = self.driver.execute_script(
            """
            const text = (root, selector) =>
                (root.querySelector(selector)?.innerText || '').trim();

            const percent = (root, selector) => {
                const value = root.querySelector(selector)?.style?.width || '';
                const number = parseFloat(value.replace('%', ''));
                return Number.isFinite(number) ? number : null;
            };

            const cardRaw = (element) => ({
                text: (element.innerText || '').trim(),
                className: element.className || '',
                value: (element.querySelector('.card .value')?.innerText || '').trim(),
                suit: (element.querySelector('.card .suit:not(.sub-suit)')?.innerText || '').trim(),
                ariaLabel: element.getAttribute('aria-label') || '',
                title: element.getAttribute('title') || '',
                dataRank: element.getAttribute('data-rank') || '',
                dataSuit: element.getAttribute('data-suit') || '',
            });

            const actionRaw = (element) => ({
                text: (element.innerText || '').trim(),
                className: element.className || '',
                disabled: Boolean(element.disabled),
            });

            const players = [...document.querySelectorAll('.table-player')].map((player, index) => ({
                seatIndex: index,
                className: player.className || '',
                name: text(player, '.table-player-name a'),
                stack: text(player, '.table-player-stack .chips-value'),
                bet: text(player, '.table-player-bet-value .chips-value'),
                statusText: text(player, '.table-player-status-icon') || text(player, '.player-hand-message .name') || text(player, '.player-hand-message'),
                hasTurnTimer: Boolean(player.querySelector('.time-to-talk')),
                timeBankPercent: percent(player, '.time-to-talk .time-bank'),
                normalTimePercent: percent(player, '.time-to-talk .normal-time'),
                isHero: player.matches('.you-player'),
            isCurrent: player.matches('.decision-current'),
            isDealer: Boolean(player.querySelector('.dealer-button-ctn')),
            isOffline: player.matches('.offline'),
            cards: [...player.querySelectorAll('.table-player-cards .card-container.flipped')].map(cardRaw),
        }));

            const heroPlayer = document.querySelector('.table-player.you-player');
            const isHeroTurn = Boolean(heroPlayer?.matches('.decision-current') && heroPlayer.querySelector('.time-to-talk'));

            return {
                url: window.location.href,
                gameType: text(document, '.table-game-type'),
                pot: text(document, '.table-pot-size .main-value'),
                isHeroTurn,
                blinds: [...document.querySelectorAll('.blind-value-ctn .chips-value, .blind-value .chips-value')]
                    .map((element) => (element.innerText || '').trim()),
                communityCards: [...document.querySelectorAll('.table-cards .card-container.flipped')]
                    .map(cardRaw),
                actions: isHeroTurn
                    ? [...document.querySelectorAll('.game-decisions-ctn .action-buttons button.action-button')].map(actionRaw)
                    : [],
                players,
            };
            """
        )

        blinds = [
            blind for blind in (self._parse_int(value) for value in data.get("blinds", []))
            if blind is not None
        ]
        big_blind = blinds[-1] if blinds else None
        pot = self._parse_int(data.get("pot"))
        players = [
            self._build_player(raw_player, big_blind)
            for raw_player in data.get("players", [])
        ]
        parsed_actions = [
            action for action in (self._parse_action(value, big_blind) for value in data.get("actions", []))
            if action is not None
        ]
        available_actions = [action for action in parsed_actions if action.enabled]
        known_cards = [
            *self._parse_cards(data.get("communityCards", [])),
            *(card for player in players for card in player.cards),
        ]
        remaining_deck = self._remaining_deck(known_cards)
        hero_cards = next((player.cards for player in players if player.is_hero), [])
        self._set_hole_card_class_distributions(players, remaining_deck)
        starting_player_count = len(players) if players else None

        return PokerTableState(
            timestamp=time(),
            url=data.get("url", ""),
            game_type=data.get("gameType", ""),
            pot=pot,
            pot_bb=self._to_big_blinds(pot, big_blind),
            blinds=blinds,
            starting_player_count=starting_player_count,
            community_cards=self._parse_cards(data.get("communityCards", [])),
            hero_cards=hero_cards,
            players=players,
            current_player_name=next((player.name for player in players if player.is_current), None),
            dealer_seat_index=next((player.seat_index for player in players if player.is_dealer), None),
            visible_actions=parsed_actions,
            available_actions=available_actions,
            remaining_deck=remaining_deck,
            was_hero_turn=bool(data.get("isHeroTurn")),
        )

    def _build_player(self, raw_player: dict, big_blind: int | None = None) -> PlayerSnapshot:
        name = raw_player.get("name", "")
        stack = self._parse_int(raw_player.get("stack"))
        bet = self._parse_int(raw_player.get("bet"))
        return PlayerSnapshot(
            seat_index=int(raw_player.get("seatIndex", 0)),
            name=name,
            stack=stack,
            stack_bb=self._to_big_blinds(stack, big_blind),
            bet=bet,
            bet_bb=self._to_big_blinds(bet, big_blind),
            status=self._parse_status(raw_player),
            cards=self._parse_cards(raw_player.get("cards", [])),
            is_hero=bool(raw_player.get("isHero")) or bool(self.hero_name and name == self.hero_name),
            is_current=bool(raw_player.get("isCurrent")),
            is_dealer=bool(raw_player.get("isDealer")),
            is_offline=bool(raw_player.get("isOffline")) or "offline" in str(raw_player.get("statusText", "")).lower(),
            has_turn_timer=bool(raw_player.get("hasTurnTimer")),
            time_bank_percent=raw_player.get("timeBankPercent"),
            normal_time_percent=raw_player.get("normalTimePercent"),
        )

    def _set_hole_card_class_distributions(
        self,
        players: list[PlayerSnapshot],
        remaining_deck: list[Card],
    ) -> None:
        unknown_combos = hole_card_class_combos(remaining_deck)
        unknown_total = sum(unknown_combos.values())
        unknown_distribution = (
            {hand_class: count / unknown_total for hand_class, count in unknown_combos.items()}
            if unknown_total
            else {}
        )

        for player in players:
            known_class = hole_card_class(player.cards)
            if known_class is not None:
                player.hole_card_class_combos = {known_class: 1}
                player.hole_card_class_distribution = {known_class: 1.0}
                continue

            if player.is_hero:
                player.hole_card_class_combos = {}
                player.hole_card_class_distribution = {}
                continue

            player.hole_card_class_combos = dict(unknown_combos)
            player.hole_card_class_distribution = dict(unknown_distribution)

    def _parse_cards(self, raw_cards: list[dict]) -> list[Card]:
        cards = []
        for raw_card in raw_cards:
            card = self._parse_card(raw_card)
            if card is not None:
                cards.append(card)
        return cards

    def _parse_card(self, raw_card: dict) -> Card | None:
        value = str(raw_card.get("value", "")).strip()
        suit = str(raw_card.get("suit", "")).strip().lower()
        if value and suit:
            rank = "T" if value == "10" else value.upper()
            if rank in RANKS and suit in SUITS:
                return Card(rank=rank, suit=suit, raw=f"{value}{suit}")

        raw = " ".join(str(value) for value in raw_card.values() if value)
        normalized = raw.lower()

        rank = self._parse_rank(normalized)
        suit = self._parse_suit(normalized)

        if rank and suit:
            return Card(rank=rank, suit=suit, raw=raw)
        return None

    def _parse_rank(self, raw: str) -> str | None:
        rank_words = {
            "ace": "A",
            "king": "K",
            "queen": "Q",
            "jack": "J",
            "ten": "T",
        }
        for word, rank in rank_words.items():
            if word in raw:
                return rank

        match = re.search(r"(?<![a-z0-9])(a|k|q|j|10|[2-9]|t)(?![a-z0-9])", raw)
        if not match:
            return None
        value = match.group(1).upper()
        return "T" if value == "10" else value

    def _parse_suit(self, raw: str) -> str | None:
        suit_words = {
            "club": "c",
            "diamond": "d",
            "heart": "h",
            "spade": "s",
        }
        for word, suit in suit_words.items():
            if word in raw:
                return suit

        match = re.search(r"(?<![a-z0-9])([cdhs])(?![a-z0-9])", raw)
        return match.group(1) if match else None

    def _parse_int(self, value) -> int | None:
        if value is None:
            return None
        digits = re.sub(r"[^\d-]", "", str(value))
        return int(digits) if digits else None

    def _to_big_blinds(self, amount: int | None, big_blind: int | None) -> float | None:
        if amount is None or not big_blind:
            return None
        return amount / big_blind

    def _parse_action(self, raw_action: dict, big_blind: int | None = None) -> ActionSnapshot | None:
        text = str(raw_action.get("text", "")).strip()
        if not text:
            return None

        normalized = text.lower()
        action_name = next(
            (name for name in ("fold", "check", "call", "raise") if normalized.startswith(name)),
            None,
        )
        if action_name is None:
            return None

        amount = self._parse_int(text)
        return ActionSnapshot(
            name=action_name,
            text=text,
            amount=amount,
            amount_bb=self._to_big_blinds(amount, big_blind),
            enabled=not bool(raw_action.get("disabled")),
        )

    def _parse_status(self, raw_player: dict) -> str:
        class_name = raw_player.get("className", "").lower()
        status_text = raw_player.get("statusText", "").strip().lower()

        if "offline" in class_name or "offline" in status_text:
            return "offline"
        if "fold" in class_name or "fold" in status_text:
            return "folded"
        if "all-in" in class_name or "all in" in status_text or "all-in" in status_text:
            return "all_in"
        if raw_player.get("hasTurnTimer"):
            return "acting"
        return "active"

    def _remaining_deck(self, known_cards: list[Card]) -> list[Card]:
        known = {str(card) for card in known_cards}
        return [card for card in self.full_deck if str(card) not in known]


class PokerEventDetector:
    def detect(self, previous: PokerTableState | None, current: PokerTableState) -> list[PokerEvent]:
        if previous is None:
            return [
                PokerEvent(
                    event_type="state_initialized",
                    timestamp=current.timestamp,
                    description="Initial table state captured",
                )
            ]

        events = []
        events.extend(self._detect_blind_events(previous, current))
        events.extend(self._detect_board_events(previous, current))
        events.extend(self._detect_turn_events(previous, current))
        events.extend(self._detect_action_button_events(previous, current))
        events.extend(self._detect_player_events(previous, current))
        return events

    def _detect_blind_events(self, previous: PokerTableState, current: PokerTableState) -> list[PokerEvent]:
        if previous.blinds == current.blinds:
            return []

        previous_blinds = "/".join(str(blind) for blind in previous.blinds) or "-"
        current_blinds = "/".join(str(blind) for blind in current.blinds) or "-"
        return [
            PokerEvent(
                event_type="blinds_changed",
                timestamp=current.timestamp,
                amount=current.big_blind,
                amount_bb=1.0 if current.big_blind else None,
                description=f"Blinds changed from {previous_blinds} to {current_blinds}",
            )
        ]

    def _detect_board_events(self, previous: PokerTableState, current: PokerTableState) -> list[PokerEvent]:
        previous_board = [str(card) for card in previous.community_cards]
        current_board = [str(card) for card in current.community_cards]

        if len(current_board) <= len(previous_board):
            return []

        new_cards = current_board[len(previous_board):]
        return [
            PokerEvent(
                event_type="board_cards_dealt",
                timestamp=current.timestamp,
                description=f"Board dealt: {' '.join(new_cards)}",
            )
        ]

    def _detect_turn_events(self, previous: PokerTableState, current: PokerTableState) -> list[PokerEvent]:
        if previous.current_player_name == current.current_player_name:
            return []

        return [
            PokerEvent(
                event_type="turn_changed",
                timestamp=current.timestamp,
                player_name=current.current_player_name,
                description=f"Turn changed to {current.current_player_name}",
            )
        ]

    def _detect_action_button_events(self, previous: PokerTableState, current: PokerTableState) -> list[PokerEvent]:
        previous_actions = set(previous.enabled_actions)
        current_actions = set(current.enabled_actions)

        if previous_actions == current_actions:
            return []

        return [
            PokerEvent(
                event_type="available_actions_changed",
                timestamp=current.timestamp,
                description=f"Available actions: {', '.join(sorted(current_actions)) or '-'}",
            )
        ]

    def _detect_player_events(self, previous: PokerTableState, current: PokerTableState) -> list[PokerEvent]:
        events = []
        previous_players = {player.name: player for player in previous.players if player.name}
        current_players = {player.name for player in current.players if player.name}
        previous_max_bet = max((player.bet or 0 for player in previous.players), default=0)

        for current_player in current.players:
            if not current_player.name:
                continue

            previous_player = previous_players.get(current_player.name)
            if previous_player is None:
                events.append(
                    PokerEvent(
                        event_type="player_joined",
                        timestamp=current.timestamp,
                        player_name=current_player.name,
                        description=f"{current_player.name} appeared at seat {current_player.seat_index}",
                    )
                )
                continue

            events.extend(
                self._detect_single_player_events(
                    previous_player,
                    current_player,
                    current.timestamp,
                    previous_max_bet,
                )
            )

        for previous_player in previous.players:
            if previous_player.name and previous_player.name not in current_players:
                events.append(
                    PokerEvent(
                        event_type="player_left",
                        timestamp=current.timestamp,
                        player_name=previous_player.name,
                        description=f"{previous_player.name} left the table",
                    )
                )

        return events

    def _detect_single_player_events(
        self,
        previous: PlayerSnapshot,
        current: PlayerSnapshot,
        timestamp: float,
        previous_max_bet: int,
    ) -> list[PokerEvent]:
        events = []

        if previous.status != "folded" and current.status == "folded":
            events.append(
                PokerEvent(
                    event_type="player_folded",
                    timestamp=timestamp,
                    player_name=current.name,
                    description=f"{current.name} folded",
                )
            )

        if not previous.is_offline and current.is_offline:
            events.append(
                PokerEvent(
                    event_type="player_went_offline",
                    timestamp=timestamp,
                    player_name=current.name,
                    description=f"{current.name} went offline",
                )
            )

        if previous.is_offline and not current.is_offline:
            events.append(
                PokerEvent(
                    event_type="player_came_online",
                    timestamp=timestamp,
                    player_name=current.name,
                    description=f"{current.name} came back online",
                )
            )

        previous_bet = previous.bet or 0
        current_bet = current.bet or 0
        if current_bet > previous_bet:
            event_type = "player_bet_changed"
            if current_bet > previous_max_bet:
                event_type = "player_raised"
            elif current_bet == previous_max_bet and previous_max_bet > 0:
                event_type = "player_called"

            events.append(
                PokerEvent(
                    event_type=event_type,
                    timestamp=timestamp,
                    player_name=current.name,
                    amount=current_bet,
                    amount_bb=current.bet_bb,
                    description=(
                        f"{current.name} {event_type.removeprefix('player_').replace('_', ' ')} "
                        f"to {self._format_amount(current_bet, current.bet_bb)}"
                    ),
                )
            )

        if previous.stack is not None and current.stack is not None and current.stack != previous.stack:
            events.append(
                PokerEvent(
                    event_type="player_stack_changed",
                    timestamp=timestamp,
                    player_name=current.name,
                    amount=current.stack,
                    amount_bb=current.stack_bb,
                    description=(
                        f"{current.name} stack changed from "
                        f"{self._format_amount(previous.stack, previous.stack_bb)} to "
                        f"{self._format_amount(current.stack, current.stack_bb)}"
                    ),
                )
            )

        return events

    def _format_amount(self, amount: int | None, amount_bb: float | None) -> str:
        if amount_bb is not None:
            return f"{amount_bb:.2f}bb"
        if amount is not None:
            return str(amount)
        return "-"

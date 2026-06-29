from __future__ import annotations

from dataclasses import dataclass, field
from time import time
import re


RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUITS = ("c", "d", "h", "s")


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
    description: str = ""


@dataclass
class ActionSnapshot:
    name: str
    text: str
    amount: int | None = None
    enabled: bool = True


@dataclass
class PlayerSnapshot:
    seat_index: int
    name: str
    stack: int | None = None
    bet: int | None = None
    status: str = "unknown"
    cards: list[Card] = field(default_factory=list)
    is_hero: bool = False
    is_current: bool = False
    is_dealer: bool = False
    has_turn_timer: bool = False
    time_bank_percent: float | None = None
    normal_time_percent: float | None = None


@dataclass
class PokerTableState:
    timestamp: float
    url: str
    game_type: str = ""
    pot: int | None = None
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

    @property
    def hero(self) -> PlayerSnapshot | None:
        return next((player for player in self.players if player.is_hero), None)

    @property
    def is_hero_turn(self) -> bool:
        hero = self.hero
        return bool(hero and hero.has_turn_timer and self.available_actions)

    @property
    def enabled_actions(self) -> dict[str, ActionSnapshot]:
        return {
            action.name: action
            for action in self.available_actions
            if action.enabled
        }


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
                cards: [...player.querySelectorAll('.table-player-cards .card-container.flipped')].map(cardRaw),
            }));

            const heroPlayer = document.querySelector('.table-player.you-player');
            const isHeroTurn = Boolean(heroPlayer?.matches('.decision-current') && heroPlayer.querySelector('.time-to-talk'));

            return {
                url: window.location.href,
                gameType: text(document, '.table-game-type'),
                pot: text(document, '.table-pot-size .main-value'),
                blinds: [...document.querySelectorAll('.blind-value-ctn .chips-value')]
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

        players = [
            self._build_player(raw_player)
            for raw_player in data.get("players", [])
        ]
        parsed_actions = [
            action for action in (self._parse_action(value) for value in data.get("actions", []))
            if action is not None
        ]
        known_cards = [
            *self._parse_cards(data.get("communityCards", [])),
            *(card for player in players for card in player.cards),
        ]
        remaining_deck = self._remaining_deck(known_cards)
        hero_cards = next((player.cards for player in players if player.is_hero), [])
        starting_player_count = len(players) if players else None

        return PokerTableState(
            timestamp=time(),
            url=data.get("url", ""),
            game_type=data.get("gameType", ""),
            pot=self._parse_int(data.get("pot")),
            blinds=[
                blind for blind in (self._parse_int(value) for value in data.get("blinds", []))
                if blind is not None
            ],
            starting_player_count=starting_player_count,
            community_cards=self._parse_cards(data.get("communityCards", [])),
            hero_cards=hero_cards,
            players=players,
            current_player_name=next((player.name for player in players if player.is_current), None),
            dealer_seat_index=next((player.seat_index for player in players if player.is_dealer), None),
            visible_actions=parsed_actions,
            available_actions=[action for action in parsed_actions if action.enabled],
            remaining_deck=remaining_deck,
        )

    def _build_player(self, raw_player: dict) -> PlayerSnapshot:
        name = raw_player.get("name", "")
        return PlayerSnapshot(
            seat_index=int(raw_player.get("seatIndex", 0)),
            name=name,
            stack=self._parse_int(raw_player.get("stack")),
            bet=self._parse_int(raw_player.get("bet")),
            status=self._parse_status(raw_player),
            cards=self._parse_cards(raw_player.get("cards", [])),
            is_hero=bool(raw_player.get("isHero")) or bool(self.hero_name and name == self.hero_name),
            is_current=bool(raw_player.get("isCurrent")),
            is_dealer=bool(raw_player.get("isDealer")),
            has_turn_timer=bool(raw_player.get("hasTurnTimer")),
            time_bank_percent=raw_player.get("timeBankPercent"),
            normal_time_percent=raw_player.get("normalTimePercent"),
        )

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

    def _parse_action(self, raw_action: dict) -> ActionSnapshot | None:
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

        return ActionSnapshot(
            name=action_name,
            text=text,
            amount=self._parse_int(text),
            enabled=not bool(raw_action.get("disabled")),
        )

    def _parse_status(self, raw_player: dict) -> str:
        class_name = raw_player.get("className", "").lower()
        status_text = raw_player.get("statusText", "").strip().lower()

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
        events.extend(self._detect_board_events(previous, current))
        events.extend(self._detect_turn_events(previous, current))
        events.extend(self._detect_action_button_events(previous, current))
        events.extend(self._detect_player_events(previous, current))
        return events

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
                    description=f"{current.name} {event_type.removeprefix('player_').replace('_', ' ')} to {current_bet}",
                )
            )

        if previous.stack is not None and current.stack is not None and current.stack != previous.stack:
            events.append(
                PokerEvent(
                    event_type="player_stack_changed",
                    timestamp=timestamp,
                    player_name=current.name,
                    amount=current.stack,
                    description=f"{current.name} stack changed from {previous.stack} to {current.stack}",
                )
            )

        if not previous.has_turn_timer and current.has_turn_timer:
            events.append(
                PokerEvent(
                    event_type="player_timer_started",
                    timestamp=timestamp,
                    player_name=current.name,
                    description=f"{current.name}'s timer started",
                )
            )

        if previous.has_turn_timer and not current.has_turn_timer:
            events.append(
                PokerEvent(
                    event_type="player_timer_stopped",
                    timestamp=timestamp,
                    player_name=current.name,
                    description=f"{current.name}'s timer stopped",
                )
            )

        return events

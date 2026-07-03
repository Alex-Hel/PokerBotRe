from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import random
from pathlib import Path

from PokerState import Card, PlayerSnapshot, PokerTableState, RANKS, SUITS, best_holdem_score
from models.RangingModel import ACTION_BUCKETS, ActionContext, CatBoostActionModel, build_action_features


DEFAULT_MODEL_PATH = Path("models/ranging_action_model.cbm")
STARTING_STACK_BB = 50
SMALL_BLIND_BB = 0.5
BIG_BLIND_BB = 1.0


@dataclass
class SimPlayer:
    seat: int
    name: str
    is_human: bool = False
    stack: float = STARTING_STACK_BB
    cards: list[Card] = field(default_factory=list)
    street_bet: float = 0.0
    total_bet: float = 0.0
    folded: bool = False
    all_in: bool = False

    @property
    def active(self) -> bool:
        return not self.folded and self.stack > 0


class CatBoostPokerSimulator:
    def __init__(self, model_path: Path, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.model = CatBoostActionModel(model_path)
        self.players = [
            SimPlayer(0, "Hero", is_human=True),
            SimPlayer(1, "CatBoost_1"),
            SimPlayer(2, "CatBoost_2"),
            SimPlayer(3, "CatBoost_3"),
        ]
        self.dealer_index = 0
        self.board: list[Card] = []
        self.deck: list[Card] = []
        self.pot = 0.0
        self.previous_raises_in_hand = 0
        self.previous_raises_by_seat = {player.seat: 0 for player in self.players}
        self.street_action_number = 0
        self.last_aggressor_seat: int | None = None

    def play(self, hands: int) -> None:
        for hand_number in range(1, hands + 1):
            if self.players[0].stack <= 0:
                print("Hero is out of chips.")
                return
            print(f"\n=== Hand {hand_number} ===")
            self.play_hand()
            self.dealer_index = (self.dealer_index + 1) % len(self.players)

    def play_hand(self) -> None:
        self.reset_hand()
        self.post_blinds()
        self.deal_hole_cards()
        self.print_table()

        preflop_first = self.next_seat(self.big_blind_seat())
        if not self.betting_round(preflop_first):
            self.award_if_one_left()
            return

        for street_cards, first_actor in (
            (3, self.first_postflop_seat()),
            (1, self.first_postflop_seat()),
            (1, self.first_postflop_seat()),
        ):
            self.deal_board(street_cards)
            self.reset_street()
            self.print_table()
            if not self.betting_round(first_actor):
                self.award_if_one_left()
                return

        self.showdown()

    def reset_hand(self) -> None:
        self.board = []
        self.deck = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.rng.shuffle(self.deck)
        self.pot = 0.0
        self.previous_raises_in_hand = 0
        self.previous_raises_by_seat = {player.seat: 0 for player in self.players}
        self.street_action_number = 0
        self.last_aggressor_seat = None

        for player in self.players:
            player.cards = []
            player.street_bet = 0.0
            player.total_bet = 0.0
            player.folded = player.stack <= 0
            player.all_in = player.stack <= 0

    def post_blinds(self) -> None:
        self.commit_chips(self.players[self.small_blind_seat()], SMALL_BLIND_BB)
        self.commit_chips(self.players[self.big_blind_seat()], BIG_BLIND_BB)

    def deal_hole_cards(self) -> None:
        for _ in range(2):
            for player in self.players:
                if player.stack > 0 or player.total_bet > 0:
                    player.cards.append(self.deck.pop())

    def deal_board(self, count: int) -> None:
        self.board.extend(self.deck.pop() for _ in range(count))

    def reset_street(self) -> None:
        for player in self.players:
            player.street_bet = 0.0
        self.street_action_number = 0
        self.last_aggressor_seat = None

    def betting_round(self, first_seat: int) -> bool:
        acted_since_raise: set[int] = set()
        seat = first_seat
        guard = 0

        while guard < 200:
            guard += 1
            player = self.players[seat]
            if player.active and not player.all_in:
                current_bet = max(p.street_bet for p in self.players)
                facing = max(0.0, current_bet - player.street_bet)
                action_bucket = self.choose_action(player, facing)
                raised = self.apply_action(player, action_bucket, facing)
                self.street_action_number += 1

                if self.count_unfolded_players() == 1:
                    return False

                if raised:
                    acted_since_raise = {player.seat}
                else:
                    acted_since_raise.add(player.seat)

                if self.round_complete(acted_since_raise):
                    return True

            seat = self.next_seat(seat)

        raise RuntimeError("Betting round did not terminate")

    def choose_action(self, player: SimPlayer, facing: float) -> str:
        legal = self.legal_actions(player, facing)
        if player.is_human:
            return self.prompt_human_action(player, facing, legal)

        state = self.make_state()
        snapshot = next(value for value in state.players if value.seat_index == player.seat)
        context = ActionContext(
            facing_amount_bb=facing,
            previous_raises_by_self_in_hand=self.previous_raises_by_seat[player.seat],
            previous_raises_in_hand=self.previous_raises_in_hand,
            last_aggressor_name=self.players[self.last_aggressor_seat].name if self.last_aggressor_seat is not None else None,
            can_check=facing <= 0,
        )
        features = build_action_features(
            state,
            snapshot,
            player.cards,
            context,
            include_keras_equity=False,
        )
        probabilities = self.model.action_probabilities(features)
        action = self.sample_legal_action(probabilities, legal)
        print(
            f"{player.name} [{self.cards_text(player.cards)}]: "
            f"{action} ({self.format_probabilities(probabilities, legal)})"
        )
        return action

    def prompt_human_action(self, player: SimPlayer, facing: float, legal: list[str]) -> str:
        print(f"\nYour cards: {' '.join(str(card) for card in player.cards)}")
        print(f"Board: {self.board_text()} | Pot: {self.pot:.2f}bb | To call: {facing:.2f}bb")
        print(f"Your stack: {player.stack:.2f}bb | Legal: {', '.join(legal)}")
        aliases = {
            "f": "fold",
            "x": "check",
            "c": "call" if facing > 0 else "check",
            "m": "min_raise",
            "h": "half_pot_raise",
            "t": "three_quarter_pot_raise",
            "p": "pot_raise",
            "a": "all_in",
        }
        while True:
            raw = input("Action [f/x/c/m/h/t/p/a]: ").strip().lower()
            action = aliases.get(raw, raw)
            if action in legal:
                return action
            print("Invalid action.")

    def legal_actions(self, player: SimPlayer, facing: float) -> list[str]:
        legal = []
        if facing > 0:
            legal.extend(["fold", "call"])
        else:
            legal.append("check")
        if self.can_raise(player, facing):
            legal.extend([
                "min_raise",
                "half_pot_raise",
                "three_quarter_pot_raise",
                "pot_raise",
                "all_in",
            ])
        return legal

    def can_raise(self, player: SimPlayer, facing: float) -> bool:
        return player.stack > facing and self.count_unfolded_players() > 1

    def sample_legal_action(self, probabilities: dict[str, float], legal: list[str]) -> str:
        weights = [max(0.0, probabilities.get(action, 0.0)) for action in legal]
        if sum(weights) <= 0:
            weights = [1.0] * len(legal)
        return self.rng.choices(legal, weights=weights, k=1)[0]

    def apply_action(self, player: SimPlayer, action: str, facing: float) -> bool:
        raised = False
        if action == "fold":
            player.folded = True
            print(f"{player.name} folds")
            return False
        if action == "check":
            print(f"{player.name} checks")
            return False
        if action == "call":
            amount = min(player.stack, facing)
            self.commit_chips(player, amount)
            print(f"{player.name} calls {amount:.2f}bb")
            return False

        amount = self.raise_amount(player, action, facing)
        before_bet = player.street_bet
        self.commit_chips(player, amount)
        if player.street_bet > before_bet + facing:
            raised = True
            self.previous_raises_in_hand += 1
            self.previous_raises_by_seat[player.seat] += 1
            self.last_aggressor_seat = player.seat
        print(f"{player.name} {action} to {player.street_bet:.2f}bb")
        return raised

    def raise_amount(self, player: SimPlayer, action: str, facing: float) -> float:
        if action == "all_in":
            return player.stack
        if action == "min_raise":
            desired_add = facing + BIG_BLIND_BB
        elif action == "half_pot_raise":
            desired_add = facing + self.pot * 0.5
        elif action == "three_quarter_pot_raise":
            desired_add = facing + self.pot * 0.75
        elif action == "pot_raise":
            desired_add = facing + self.pot
        else:
            desired_add = facing
        return min(player.stack, max(facing, desired_add))

    def commit_chips(self, player: SimPlayer, amount: float) -> None:
        amount = min(player.stack, max(0.0, amount))
        player.stack -= amount
        player.street_bet += amount
        player.total_bet += amount
        self.pot += amount
        if player.stack <= 0:
            player.all_in = True

    def round_complete(self, acted_since_raise: set[int]) -> bool:
        active = [player for player in self.players if player.active and not player.all_in]
        if not active:
            return True
        current_bet = max(player.street_bet for player in self.players)
        return all(
            player.seat in acted_since_raise and abs(player.street_bet - current_bet) < 1e-9
            for player in active
        )

    def award_if_one_left(self) -> None:
        winner = next(player for player in self.players if not player.folded)
        winner.stack += self.pot
        print(f"{winner.name} wins {self.pot:.2f}bb")
        self.print_stacks()

    def showdown(self) -> None:
        contenders = [player for player in self.players if not player.folded]
        scored = [
            (best_holdem_score([*player.cards, *self.board]), player)
            for player in contenders
        ]
        best_score = max(score for score, _ in scored)
        winners = [player for score, player in scored if score == best_score]
        share = self.pot / len(winners)
        for winner in winners:
            winner.stack += share
        print("\nShowdown")
        for player in contenders:
            print(f"{player.name}: {' '.join(str(card) for card in player.cards)}")
        print(f"Winners: {', '.join(player.name for player in winners)} win {share:.2f}bb each")
        self.print_stacks()

    def make_state(self) -> PokerTableState:
        snapshots = [
            PlayerSnapshot(
                seat_index=player.seat,
                name=player.name,
                stack=int(round(player.stack)),
                stack_bb=player.stack,
                bet=int(round(player.street_bet)),
                bet_bb=player.street_bet,
                status="folded" if player.folded else "active",
                cards=list(player.cards),
                is_hero=player.is_human,
                is_dealer=player.seat == self.dealer_index,
            )
            for player in self.players
        ]
        return PokerTableState(
            timestamp=0.0,
            url="simulator",
            game_type="th",
            pot=int(round(self.pot)),
            pot_bb=self.pot,
            blinds=[int(SMALL_BLIND_BB), int(BIG_BLIND_BB)],
            starting_player_count=len(self.players),
            community_cards=list(self.board),
            hero_cards=list(self.players[0].cards),
            players=snapshots,
            dealer_seat_index=self.dealer_index,
        )

    def print_table(self) -> None:
        print(f"Board: {self.board_text()} | Pot: {self.pot:.2f}bb")
        self.print_stacks()

    def print_stacks(self) -> None:
        print("Stacks: " + ", ".join(f"{player.name}={player.stack:.2f}bb" for player in self.players))

    def board_text(self) -> str:
        return " ".join(str(card) for card in self.board) or "-"

    def cards_text(self, cards: list[Card]) -> str:
        return " ".join(str(card) for card in cards) or "-"

    def count_unfolded_players(self) -> int:
        return sum(1 for player in self.players if not player.folded)

    def small_blind_seat(self) -> int:
        return self.next_seat(self.dealer_index)

    def big_blind_seat(self) -> int:
        return self.next_seat(self.small_blind_seat())

    def first_postflop_seat(self) -> int:
        return self.next_active_from(self.dealer_index)

    def next_seat(self, seat: int) -> int:
        return (seat + 1) % len(self.players)

    def next_active_from(self, seat: int) -> int:
        next_seat = self.next_seat(seat)
        while self.players[next_seat].folded or self.players[next_seat].all_in:
            next_seat = self.next_seat(next_seat)
        return next_seat

    def format_probabilities(self, probabilities: dict[str, float], legal: list[str]) -> str:
        return ", ".join(
            f"{action}={probabilities.get(action, 0.0):.2f}"
            for action in ACTION_BUCKETS
            if action in legal
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    simulator = CatBoostPokerSimulator(args.model, seed=args.seed)
    simulator.play(args.hands)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from models import RangingModel
from models.EquityModel import predict_equity_batch
from PokerState import Card, PlayerSnapshot, PokerTableState
from models.RangingModel import (
    ACTION_BUCKETS,
    FEATURE_NAMES,
    ActionContext,
    build_action_features,
    classify_action_bucket,
)


RAW_DIR = Path("")
INTERPRETED_DIR = Path("../interpreted")
_PROGRESS_COUNTS: dict[str, int] = {}
_COMPUTE_KERAS_EQUITY = True
_EQUITY_JOBS: list[tuple[dict, list[Card], list[Card], int]] = []
OUTPUT_COLUMNS = (
    "hand_id",
    "hand_number",
    "actor",
    "actor_seat",
    "hole_card_1",
    "hole_card_2",
    "action_bucket",
    "raw_action",
    "raw_amount",
    *FEATURE_NAMES,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-keras-equity",
        action="store_true",
        help="Leave keras_equity as nan for faster conversion.",
    )
    parser.add_argument(
        "--keras-batch-size",
        type=int,
        default=4096,
        help="Batch size for Keras equity prediction.",
    )
    args = parser.parse_args()
    global _COMPUTE_KERAS_EQUITY
    if args.skip_keras_equity:
        _COMPUTE_KERAS_EQUITY = False
        RangingModel._KERAS_EQUITY_AVAILABLE = False

    INTERPRETED_DIR.mkdir(parents=True, exist_ok=True)
    poker_now_rows = convert_poker_now_dataset(RAW_DIR / "poker_now")
    fill_keras_equity("poker_now keras equity", batch_size=args.keras_batch_size)
    write_rows(INTERPRETED_DIR / "internal_ranging_rows.csv", poker_now_rows)

    kaggle_rows = convert_kaggle_dataset(RAW_DIR / "kaggle")
    fill_keras_equity("kaggle keras equity", batch_size=args.keras_batch_size)
    write_rows(INTERPRETED_DIR / "external_ranging_rows.csv", kaggle_rows)

    print(f"wrote {len(poker_now_rows)} PokerNow rows")
    print(f"wrote {len(kaggle_rows)} Kaggle/HM rows")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(row)
            if index % 100 == 0:
                print(f"{path.name}: wrote {index} rows")


def fill_keras_equity(label: str, batch_size: int = 4096) -> None:
    if not _COMPUTE_KERAS_EQUITY or not _EQUITY_JOBS:
        _EQUITY_JOBS.clear()
        return

    total = len(_EQUITY_JOBS)
    for start in range(0, total, batch_size):
        batch = _EQUITY_JOBS[start:start + batch_size]
        predictions = predict_equity_batch(
            ((hole_cards, board, opponent_count) for _, hole_cards, board, opponent_count in batch),
            batch_size=batch_size,
        )
        for (row, _, _, _), prediction in zip(batch, predictions):
            row["keras_equity"] = prediction
        print(f"{label}: filled {min(start + batch_size, total)} / {total} rows")

    _EQUITY_JOBS.clear()


def convert_poker_now_dataset(raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        hands = data.get("hands", [])
        starting_player_count = max((len(hand.get("players", [])) for hand in hands), default=0)

        for hand in hands:
            rows.extend(add_rows_with_progress(
                "poker_now generated",
                parse_poker_now_hand(path, hand, starting_player_count),
            ))
    return rows


def parse_poker_now_hand(path: Path, hand: dict, starting_player_count: int) -> list[dict]:
    big_blind = float(hand.get("bigBlind") or 1)
    small_blind = float(hand.get("smallBlind") or big_blind / 2)
    dealer_seat = hand.get("dealerSeat")
    known_cards = poker_now_known_cards(hand)
    players = {
        int(player["seat"]): {
            "name": player.get("name", f"seat_{player['seat']}"),
            "stack": float(player.get("stack") or 0),
            "bet": 0.0,
            "status": "active",
        }
        for player in hand.get("players", [])
    }

    board: list[Card] = []
    current_street_bets = {seat: 0.0 for seat in players}
    pot = 0.0
    previous_raises = 0
    previous_raises_by_seat = {seat: 0 for seat in players}
    street_action_number = 0
    last_aggressor_seat = None
    rows = []

    for event in hand.get("events", []):
        payload = event.get("payload", {})
        event_type = payload.get("type")
        seat = payload.get("seat")

        if event_type in {2, 3, 4, 5, 1} and seat in players:
            blind_amount = float(payload.get("value") or 0)
            current_street_bets[seat] = current_street_bets.get(seat, 0.0) + blind_amount
            players[seat]["bet"] += blind_amount
            players[seat]["stack"] -= blind_amount
            pot += blind_amount
            continue

        if event_type == 9:
            board.extend(parse_card(card) for card in payload.get("cards", []))
            current_street_bets = {player_seat: 0.0 for player_seat in players}
            street_action_number = 0
            continue

        if event_type not in {0, 7, 8, 11} or seat not in players:
            continue

        actor_cards = known_cards.get(seat)
        max_bet = max(current_street_bets.values(), default=0.0)
        facing_amount = max(0.0, max_bet - current_street_bets.get(seat, 0.0))
        raw_action, target_amount, is_all_in = poker_now_action_details(event_type, payload)
        amount_added = 0.0

        if event_type == 0:
            action_bucket = "fold"
        elif event_type == 11:
            action_bucket = "check"
        elif event_type == 7:
            action_bucket = "call"
        else:
            action_bucket = classify_action_bucket(
                "raise",
                amount_bb=(target_amount - current_street_bets.get(seat, 0.0)) / big_blind,
                pot_bb=pot / big_blind,
                is_all_in=is_all_in,
            )

        if actor_cards:
            state = make_state(
                players,
                board,
                pot,
                small_blind,
                big_blind,
                dealer_seat,
                starting_player_count,
            )
            player_snapshot = next(player for player in state.players if player.seat_index == seat)
            context = ActionContext(
                facing_amount_bb=facing_amount / big_blind,
                previous_raises_by_self_in_hand=previous_raises_by_seat.get(seat, 0),
                previous_raises_in_hand=previous_raises,
                street_action_number=street_action_number,
                last_aggressor_name=players.get(last_aggressor_seat, {}).get("name"),
                can_check=facing_amount <= 0,
                can_raise=True,
                is_facing_all_in=False,
            )
            rows.append(make_row(
                hand_id=str(hand.get("id", "")),
                hand_number=str(hand.get("number", "")),
                actor=players[seat]["name"],
                actor_seat=seat,
                hole_cards=actor_cards,
                action_bucket=action_bucket,
                raw_action=raw_action,
                raw_amount=target_amount,
                state=state,
                player=player_snapshot,
                context=context,
            ))

        if event_type == 0:
            players[seat]["status"] = "folded"
        elif event_type == 7:
            target = float(payload.get("value") or max_bet)
            amount_added = max(0.0, target - current_street_bets.get(seat, 0.0))
            current_street_bets[seat] = max(current_street_bets.get(seat, 0.0), target)
        elif event_type == 8:
            target = float(payload.get("value") or 0)
            amount_added = max(0.0, target - current_street_bets.get(seat, 0.0))
            current_street_bets[seat] = max(current_street_bets.get(seat, 0.0), target)
            previous_raises += 1
            previous_raises_by_seat[seat] = previous_raises_by_seat.get(seat, 0) + 1
            last_aggressor_seat = seat

        if amount_added:
            players[seat]["bet"] += amount_added
            players[seat]["stack"] -= amount_added
            pot += amount_added
        street_action_number += 1

    return rows


def poker_now_known_cards(hand: dict) -> dict[int, list[Card]]:
    known = {}
    for event in hand.get("events", []):
        payload = event.get("payload", {})
        if payload.get("type") == 12 and payload.get("seat") is not None:
            known[int(payload["seat"])] = parse_cards_list(payload.get("cards", []))
        if payload.get("type") == 10 and payload.get("seat") is not None and payload.get("cards"):
            known[int(payload["seat"])] = parse_cards_list(payload.get("cards", []))
    return {seat: cards for seat, cards in known.items() if len(cards) == 2}


def poker_now_action_details(event_type: int, payload: dict) -> tuple[str, float, bool]:
    if event_type == 0:
        return "fold", 0.0, False
    if event_type == 7:
        return "call", float(payload.get("value") or 0), bool(payload.get("allIn"))
    if event_type == 8:
        return "raise", float(payload.get("value") or 0), bool(payload.get("allIn"))
    if event_type == 11:
        return "check", 0.0, False
    return "unknown", 0.0, False


def convert_kaggle_dataset(raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        hands = [f"Game started at:{chunk}" for chunk in text.split("Game started at:") if chunk.strip()]
        starting_player_count = max((len(parse_kaggle_players(hand)) for hand in hands), default=0)
        for index, hand_text in enumerate(hands, start=1):
            rows.extend(add_rows_with_progress(
                "kaggle generated",
                parse_kaggle_hand(path, hand_text, index, starting_player_count),
            ))
    return rows


def add_rows_with_progress(label: str, new_rows: list[dict]) -> list[dict]:
    count = _PROGRESS_COUNTS.get(label, 0)
    for _ in new_rows:
        count += 1
        if count % 100 == 0:
            print(f"{label}: {count} rows")
    _PROGRESS_COUNTS[label] = count
    return new_rows


def parse_kaggle_hand(path: Path, hand_text: str, fallback_number: int, starting_player_count: int) -> list[dict]:
    lines = [line.strip() for line in hand_text.splitlines() if line.strip()]
    game_id = parse_first(r"Game ID:\s+(\S+)", hand_text) or str(fallback_number)
    blind_text = parse_first(r"Game ID:\s+\S+\s+([0-9.]+)/([0-9.]+)", hand_text)
    blind_match = re.search(r"Game ID:\s+\S+\s+([0-9.]+)/([0-9.]+)", hand_text)
    small_blind = float(blind_match.group(1)) if blind_match else 0.5
    big_blind = float(blind_match.group(2)) if blind_match else 1.0
    dealer_seat = int(parse_first(r"Seat\s+(\d+)\s+is the button", hand_text) or 0)
    players = parse_kaggle_players(hand_text)
    known_cards = kaggle_known_cards(hand_text)
    board: list[Card] = []
    current_street_bets = {seat: 0.0 for seat in players}
    pot = 0.0
    previous_raises = 0
    previous_raises_by_seat = {seat: 0 for seat in players}
    street_action_number = 0
    last_aggressor_seat = None
    rows = []

    name_to_seat = {value["name"]: seat for seat, value in players.items()}

    for line in lines:
        board_cards = parse_board_line(line)
        if board_cards is not None:
            board = board_cards
            current_street_bets = {seat: 0.0 for seat in players}
            street_action_number = 0
            continue

        blind_match = re.match(r"Player (.+) has (small|big) blind \(([0-9.]+)\)", line)
        if blind_match:
            name, _, amount_text = blind_match.groups()
            seat = name_to_seat.get(name)
            if seat in players:
                amount = float(amount_text)
                current_street_bets[seat] += amount
                players[seat]["bet"] += amount
                players[seat]["stack"] -= amount
                pot += amount
            continue

        action_match = re.match(r"Player (.+) (folds|checks|calls|bets|raises)(?: \(([0-9.]+)\))?", line)
        if not action_match:
            continue

        name, raw_action, amount_text = action_match.groups()
        seat = name_to_seat.get(name)
        if seat not in players:
            continue

        amount = float(amount_text or 0)
        max_bet = max(current_street_bets.values(), default=0.0)
        facing_amount = max(0.0, max_bet - current_street_bets.get(seat, 0.0))

        if raw_action == "folds":
            action_bucket = "fold"
        elif raw_action == "checks":
            action_bucket = "check"
        elif raw_action == "calls":
            action_bucket = "call"
        else:
            action_bucket = classify_action_bucket(
                "raise",
                amount_bb=amount / big_blind,
                pot_bb=pot / big_blind,
                is_all_in=amount >= players[seat]["stack"],
            )

        actor_cards = known_cards.get(name)
        if actor_cards:
            state = make_state(
                players,
                board,
                pot,
                small_blind,
                big_blind,
                dealer_seat,
                starting_player_count,
            )
            player_snapshot = next(player for player in state.players if player.seat_index == seat)
            context = ActionContext(
                facing_amount_bb=facing_amount / big_blind,
                previous_raises_by_self_in_hand=previous_raises_by_seat.get(seat, 0),
                previous_raises_in_hand=previous_raises,
                street_action_number=street_action_number,
                last_aggressor_name=players.get(last_aggressor_seat, {}).get("name"),
                can_check=facing_amount <= 0,
                can_raise=True,
                is_facing_all_in=False,
            )
            rows.append(make_row(
                hand_id=game_id,
                hand_number=str(fallback_number),
                actor=name,
                actor_seat=seat,
                hole_cards=actor_cards,
                action_bucket=action_bucket,
                raw_action=raw_action,
                raw_amount=amount,
                state=state,
                player=player_snapshot,
                context=context,
            ))

        if raw_action == "folds":
            players[seat]["status"] = "folded"
        elif raw_action == "calls":
            contribution = amount
            current_street_bets[seat] += contribution
            players[seat]["bet"] += contribution
            players[seat]["stack"] -= contribution
            pot += contribution
        elif raw_action in {"bets", "raises"}:
            contribution = amount
            current_street_bets[seat] += contribution
            players[seat]["bet"] += contribution
            players[seat]["stack"] -= contribution
            pot += contribution
            previous_raises += 1
            previous_raises_by_seat[seat] = previous_raises_by_seat.get(seat, 0) + 1
            last_aggressor_seat = seat
        street_action_number += 1

    return rows


def parse_kaggle_players(hand_text: str) -> dict[int, dict]:
    players = {}
    for match in re.finditer(r"Seat\s+(\d+):\s+(.+?)\s+\(([0-9.]+)\)\.", hand_text):
        seat = int(match.group(1))
        players[seat] = {
            "name": match.group(2),
            "stack": float(match.group(3)),
            "bet": 0.0,
            "status": "active",
        }
    return players


def kaggle_known_cards(hand_text: str) -> dict[str, list[Card]]:
    known: dict[str, list[Card]] = {}
    for match in re.finditer(r"Player (.+?) received card: \[([2-9TJQKA]|10)([cdhs])\]", hand_text):
        known.setdefault(match.group(1), []).append(parse_card(match.group(2) + match.group(3)))
    for match in re.finditer(r"Player (.+?) shows: .*?\[([^\]]+)\]", hand_text):
        cards = parse_cards_text(match.group(2))
        if len(cards) == 2:
            known[match.group(1)] = cards
    return {name: cards for name, cards in known.items() if len(cards) == 2}


def parse_board_line(line: str) -> list[Card] | None:
    if not line.startswith("*** "):
        return None
    cards = re.findall(r"([2-9TJQKA]|10)([cdhs])", line)
    if not cards:
        return None
    return [parse_card(rank + suit) for rank, suit in cards]


def make_state(
    players: dict[int, dict],
    board: list[Card],
    pot: float,
    small_blind: float,
    big_blind: float,
    dealer_seat: int | None,
    starting_player_count: int,
) -> PokerTableState:
    snapshots = [
        PlayerSnapshot(
            seat_index=seat,
            name=value["name"],
            stack=int(round(value["stack"])),
            stack_bb=value["stack"] / big_blind,
            bet=int(round(value["bet"])),
            bet_bb=value["bet"] / big_blind,
            status=value["status"],
            is_dealer=seat == dealer_seat,
        )
        for seat, value in sorted(players.items())
    ]
    return PokerTableState(
        timestamp=0.0,
        url="",
        game_type="th",
        pot=int(round(pot)),
        pot_bb=pot / big_blind,
        blinds=[int(round(small_blind)), int(round(big_blind))],
        starting_player_count=starting_player_count or len(players),
        community_cards=list(board),
        players=snapshots,
        dealer_seat_index=dealer_seat,
    )


def make_row(
    hand_id: str,
    hand_number: str,
    actor: str,
    actor_seat: int,
    hole_cards: list[Card],
    action_bucket: str,
    raw_action: str,
    raw_amount: float,
    state: PokerTableState,
    player: PlayerSnapshot,
    context: ActionContext,
) -> dict:
    features = build_action_features(
        state,
        player,
        hole_cards,
        context,
        include_keras_equity=False,
    )
    row = {
        "hand_id": hand_id,
        "hand_number": hand_number,
        "actor": actor,
        "actor_seat": actor_seat,
        "hole_card_1": str(hole_cards[0]),
        "hole_card_2": str(hole_cards[1]),
        "action_bucket": action_bucket if action_bucket in ACTION_BUCKETS else "call",
        "raw_action": raw_action,
        "raw_amount": raw_amount,
        **features,
    }
    if _COMPUTE_KERAS_EQUITY:
        active_players = len([
            player
            for player in state.players
            if player.status not in {"folded", "offline"}
        ])
        _EQUITY_JOBS.append((
            row,
            list(hole_cards),
            list(state.community_cards),
            max(1, active_players - 1),
        ))
    return row


def parse_card(raw: str) -> Card:
    value = raw.strip()
    suit = value[-1].lower()
    rank = value[:-1].upper()
    if rank == "10":
        rank = "T"
    return Card(rank=rank, suit=suit, raw=raw)


def parse_cards_list(raw_cards: list) -> list[Card]:
    cards = []
    for raw_card in raw_cards:
        if not raw_card:
            continue
        try:
            cards.append(parse_card(str(raw_card)))
        except (IndexError, ValueError):
            continue
    return cards


def parse_cards_text(raw: str) -> list[Card]:
    return [parse_card(rank + suit) for rank, suit in re.findall(r"([2-9TJQKA]|10)([cdhs])", raw)]


def parse_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


if __name__ == "__main__":
    main()

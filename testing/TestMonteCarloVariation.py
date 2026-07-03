from __future__ import annotations

from statistics import mean, pstdev

from PokerState import Card, PlayerSnapshot, PokerTableState


def main() -> None:
    hero_cards = [Card("A", "s"), Card("K", "d")]
    board = [Card("Q", "s"), Card("J", "h"), Card("2", "c")]
    known_cards = {str(card) for card in [*hero_cards, *board]}
    remaining_deck = [
        Card(rank, suit)
        for suit in ("c", "d", "h", "s")
        for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
        if f"{rank}{suit}" not in known_cards
    ]

    state = PokerTableState(
        timestamp=0.0,
        url="test",
        pot_bb=12.0,
        blinds=[1, 2],
        community_cards=board,
        hero_cards=hero_cards,
        dealer_seat_index=0,
        remaining_deck=remaining_deck,
        players=[
            PlayerSnapshot(
                seat_index=0,
                name="Hero",
                stack_bb=100.0,
                bet_bb=0.0,
                cards=hero_cards,
                is_hero=True,
                is_dealer=True,
            ),
            PlayerSnapshot(
                seat_index=1,
                name="Villain 1",
                stack_bb=100.0,
                bet_bb=0.0,
                status="active",
            ),
            PlayerSnapshot(
                seat_index=2,
                name="Villain 2",
                stack_bb=100.0,
                bet_bb=0.0,
                status="active",
            ),
        ],
    )

    simulations = 2000
    results = []
    for run_number in range(1, 11):
        equity = state.monte_carlo(simulations=simulations)
        results.append(equity)
        print(f"run {run_number:02d}: {equity:.4f}")

    print()
    print(f"mean:   {mean(results):.4f}")
    print(f"stddev: {pstdev(results):.4f}")
    print(f"min:    {min(results):.4f}")
    print(f"max:    {max(results):.4f}")


if __name__ == "__main__":
    main()

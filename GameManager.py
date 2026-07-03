from PokerNow import PokerClient
from PokerNow import ElementHelper
import undetected_chromedriver as webdriver
from undetected_chromedriver import WebElement
from pathlib import Path
import random
import time
import requests
import zipfile
from selenium.common.exceptions import JavascriptException, NoAlertPresentException, StaleElementReferenceException, TimeoutException, UnexpectedAlertPresentException
from selenium.webdriver.common.by import By

from models.EquityModel import predict_state_equity
from models.RangingModel import LiveRangingTracker, average_combo_range_equity
from PokerState import PokerEvent, PokerEventDetector, PokerTableScraper, PokerTableState


class GameManager:
    driver = None
    element_helper = None
    client = None
    gm = None
    mouse_x = None
    mouse_y = None
    table_scraper = None
    table_state = None
    previous_table_state = None
    table_events = None

    def __init__(self, hero_name: str | None = None) -> None:
        self.cookie_path = Path(__file__).resolve().with_name('cookie_file.pkl')
        self.had_saved_cookies = self.cookie_path.is_file()

        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-first-run --no-service-autorun --password-store=basic')

        # Download and extract the extension
        response = requests.get(
            'https://github.com/NopeCHALLC/nopecha-extension/releases/latest/download/chromium.zip'
        )
        with open('chromium.zip', 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile('chromium.zip', 'r') as z:
            z.extractall('chromium')

        options.add_argument('--load-extension=chromium')

        self.driver = webdriver.Chrome(options=options, version_main=149)
        self.client = PokerClient(self.driver, cookie_path=str(self.cookie_path))
        self.element_helper = ElementHelper(self.driver)
        self.mouse_x = None
        self.mouse_y = None
        self.gm = self.client.game_state_manager
        self.table_scraper = PokerTableScraper(self.driver, hero_name=hero_name)
        self.event_detector = PokerEventDetector()
        self.ranging_tracker = LiveRangingTracker()
        self.previous_table_state: PokerTableState | None = None
        self.table_state: PokerTableState | None = None
        self.table_events: list[PokerEvent] = []
        self.last_equity_overlay: dict[str, str] | None = None
        self.last_opponent_equity_by_seat: dict[str, str] = {}

    def start_gameplay_loop(self) -> None:
        self.client.navigate('https://network.pokernow.com/sng_tournaments')
        self.driver.maximize_window()

        if not self.had_saved_cookies:
            input(f'enter login, press enter when done ({self.cookie_path})')
            self.client.cookie_manager.save_cookies()
            self.had_saved_cookies = True

        while True:
            self.join_game()
            self.play_game()
            self.leave_game()


    def join_game(self) -> None:
        time.sleep(3)
        self.wait_for_recaptcha()

        room_selector = "a.button-1.big.green[href*='/table']"  # join room button
        self.accept_alert_if_present()

        queue_selector = "button.button-1.big.blue"
        while self.is_element_present_accepting_alerts(queue_selector):
            self.wait_for_recaptcha()
            self.wait_for_element_accepting_alerts(queue_selector, timeout=10)
            element = self.get_element_accepting_alerts(queue_selector)
            if element is None:
                continue
            self.human_click(element, queue_selector)
            self.accept_alert_if_present()
            time.sleep(3)

        self.wait_for_element_accepting_alerts(room_selector, timeout=900)
        element = self.get_element_accepting_alerts(room_selector)
        if element is None:
            raise TimeoutException(f"Timed out waiting for selector: {room_selector}")
        previous_windows = set(self.driver.window_handles)
        self.human_click(element, room_selector)
        self.open_room_link_if_needed(room_selector, previous_windows)

        self.switch_to_new_table_window(previous_windows, timeout=15)
        self.wait_for_table_loaded(timeout=300)

    def play_game(self) -> None:
        while True:
            time.sleep(1)
            self.ignore_vc()
            self.accept_tos()
            self.im_back()
            state = self.update_table_state()
            was_hero_turn = bool(self.previous_table_state and self.previous_table_state.is_hero_turn)
            if state.is_hero_turn and not was_hero_turn:
                sampling_estimates = self.sampling_equity_estimates(state)
                self.update_equity_overlay(state, estimates=sampling_estimates)
                estimates = self.calculate_equity_estimates(state)
                self.print_equity_estimates(estimates)
                self.update_equity_overlay(
                    state,
                    estimates=estimates,
                    refresh_opponent_equities=True,
                )
            else:
                self.update_equity_overlay(state)
            if self.hero_left_table(state) or len(state.players) == 1:
                print("[state-event] Hero left table; ending play_game loop")
                return

    def print_equity_estimates(self, estimates: dict[str, str]) -> None:
        print(
            "equity: "
            f"ranged={estimates['ranged_equity']}, "
            f"keras={estimates['keras_equity']}"
        )

    def sampling_equity_estimates(self, state: PokerTableState) -> dict[str, str]:
        return {
            "ranged_equity": "sampling...",
            "keras_equity": "sampling...",
            "cards": self.format_cards(state),
            "updated": time.strftime("%H:%M:%S"),
        }

    def calculate_equity_estimates(self, state: PokerTableState) -> dict[str, str]:
        monte_carlo_equity = state.monte_carlo()
        try:
            keras_equity = predict_state_equity(state)
            keras_text = f"{keras_equity:.3f}"
        except Exception as exc:
            keras_text = f"unavailable ({exc})"

        return {
            "ranged_equity": f"{monte_carlo_equity:.3f}",
            "keras_equity": keras_text,
            "cards": self.format_cards(state),
            "updated": time.strftime("%H:%M:%S"),
        }

    def update_equity_overlay(
        self,
        state: PokerTableState,
        estimates: dict[str, str] | None = None,
        refresh_opponent_equities: bool = False,
    ) -> None:
        if estimates is not None:
            self.last_equity_overlay = estimates

        if refresh_opponent_equities:
            self.last_opponent_equity_by_seat = self.opponent_equity_by_seat(state)

        overlay = self.last_equity_overlay or {
            "ranged_equity": "-",
            "keras_equity": "-",
            "cards": self.format_cards(state),
            "updated": "waiting for turn",
        }
        self.inject_equity_overlay(overlay, self.opponent_equity_overlays(state))

    def opponent_equity_by_seat(self, state: PokerTableState) -> dict[str, str]:
        equities = {}
        for player in state.players:
            if player.is_hero or player.status in {"folded", "offline"}:
                continue

            mean_equity = average_combo_range_equity(player.hole_combo_range, state)
            equities[str(player.seat_index)] = f"{mean_equity:.3f}" if mean_equity is not None else "-"
        return equities

    def opponent_equity_overlays(self, state: PokerTableState) -> list[dict[str, str]]:
        overlays = []
        for player in state.players:
            if player.is_hero or player.status in {"folded", "offline"}:
                continue

            seat_index = str(player.seat_index)
            overlays.append({
                "seat_index": seat_index,
                "mean_equity": self.last_opponent_equity_by_seat.get(seat_index, "-"),
            })
        return overlays

    def reset_equity_overlays(self) -> None:
        self.last_equity_overlay = None
        self.last_opponent_equity_by_seat = {}

    def should_reset_equity_overlays(
        self,
        previous: PokerTableState | None,
        current: PokerTableState,
    ) -> bool:
        if previous is None:
            return True

        previous_hero_cards = tuple(str(card) for card in previous.hero_cards)
        current_hero_cards = tuple(str(card) for card in current.hero_cards)
        if previous_hero_cards and not current_hero_cards:
            return True
        if previous_hero_cards and current_hero_cards and previous_hero_cards != current_hero_cards:
            return True
        if len(current.community_cards) < len(previous.community_cards):
            return True
        return False

    def inject_equity_overlay(
        self,
        overlay: dict[str, str],
        opponent_overlays: list[dict[str, str]],
    ) -> None:
        try:
            self.driver.execute_script(
                """
                const data = arguments[0];
                const opponentData = arguments[1] || [];
                const id = 'pokerbot-equity-overlay';
                let root = document.getElementById(id);
                if (!root) {
                    root = document.createElement('div');
                    root.id = id;
                    document.body.appendChild(root);
                }

                root.innerHTML = '';
                Object.assign(root.style, {
                    position: 'fixed',
                    top: '50%',
                    bottom: 'auto',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    zIndex: '2147483647',
                    minWidth: '180px',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.22)',
                    background: 'rgba(8, 12, 18, 0.88)',
                    color: '#f8fafc',
                    fontFamily: 'Arial, sans-serif',
                    fontSize: '13px',
                    lineHeight: '1.35',
                    boxShadow: '0 8px 24px rgba(0,0,0,0.35)',
                    pointerEvents: 'none',
                    textAlign: 'left'
                });

                const addRow = (label, value) => {
                    const row = document.createElement('div');
                    Object.assign(row.style, {
                        display: 'flex',
                        justifyContent: 'space-between',
                        gap: '12px',
                        whiteSpace: 'nowrap'
                    });

                    const key = document.createElement('span');
                    key.textContent = label;
                    key.style.color = '#cbd5e1';

                    const val = document.createElement('span');
                    val.textContent = value;
                    val.style.fontWeight = '700';

                    row.appendChild(key);
                    row.appendChild(val);
                    root.appendChild(row);
                };

                addRow('Ranged equity', data.ranged_equity || '-');
                addRow('Keras equity', data.keras_equity || '-');

                const meta = document.createElement('div');
                meta.textContent = `${data.cards || ''} ${data.updated || ''}`.trim();
                Object.assign(meta.style, {
                    marginTop: '6px',
                    color: '#94a3b8',
                    fontSize: '11px'
                });
                root.appendChild(meta);

                const clampHudPosition = (targetLeft, targetTop) => {
                    const margin = 12;
                    const halfWidth = root.offsetWidth / 2;
                    const clampedLeft = Math.min(
                        window.innerWidth - halfWidth - margin,
                        Math.max(halfWidth + margin, targetLeft)
                    );
                    const clampedTop = Math.min(
                        window.innerHeight - root.offsetHeight - margin,
                        Math.max(margin, targetTop)
                    );
                    root.style.left = `${clampedLeft}px`;
                    root.style.right = 'auto';
                    root.style.bottom = 'auto';
                    root.style.top = `${clampedTop}px`;
                    root.style.transform = 'translateX(-50%)';
                };

                const board = document.querySelector('.table-cards');
                const boardHasCards = Boolean(board?.querySelector('.card-container.flipped'));
                if (board && board.offsetParent !== null && boardHasCards) {
                    const rect = board.getBoundingClientRect();
                    clampHudPosition(
                        rect.left + rect.width / 2,
                        rect.bottom + 10
                    );
                } else {
                    root.style.left = '50%';
                    root.style.right = 'auto';
                    root.style.bottom = 'auto';
                    root.style.top = '50%';
                    root.style.transform = 'translate(-50%, -50%)';
                }

                const existingOpponentOverlays = [
                    ...document.querySelectorAll('.pokerbot-opponent-equity-overlay')
                ];
                for (const node of existingOpponentOverlays) {
                    node.remove();
                }

                const players = [...document.querySelectorAll('.table-player')];
                for (const item of opponentData) {
                    const player = players[Number(item.seat_index)];
                    if (!player) {
                        continue;
                    }

                    const badge = document.createElement('div');
                    badge.className = 'pokerbot-opponent-equity-overlay';
                    badge.textContent = `Mean equity ${item.mean_equity || '-'}`;
                    Object.assign(badge.style, {
                        position: 'absolute',
                        left: '50%',
                        top: 'calc(100% + 6px)',
                        transform: 'translateX(-50%)',
                        zIndex: '2147483647',
                        padding: '4px 7px',
                        borderRadius: '6px',
                        background: 'rgba(8, 12, 18, 0.86)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        color: '#f8fafc',
                        fontFamily: 'Arial, sans-serif',
                        fontSize: '11px',
                        fontWeight: '700',
                        lineHeight: '1.2',
                        whiteSpace: 'nowrap',
                        pointerEvents: 'none',
                        boxShadow: '0 4px 14px rgba(0,0,0,0.28)'
                    });

                    if (getComputedStyle(player).position === 'static') {
                        player.style.position = 'relative';
                    }
                    player.appendChild(badge);
                }
                """,
                overlay,
                opponent_overlays,
            )
        except JavascriptException:
            return

    def format_cards(self, state: PokerTableState) -> str:
        hero_cards = " ".join(str(card) for card in state.hero_cards) or "--"
        board_cards = " ".join(str(card) for card in state.community_cards) or "--"
        return f"{hero_cards} | {board_cards}"

    def update_table_state(self) -> PokerTableState:
        current_state = self.table_scraper.scrape()
        if self.table_state and self.table_state.starting_player_count is not None:
            current_state.starting_player_count = self.table_state.starting_player_count

        if self.should_reset_equity_overlays(self.table_state, current_state):
            self.reset_equity_overlays()
        
        self.table_events = self.event_detector.detect(self.table_state, current_state)
        self.ranging_tracker.process_snapshot(self.table_state, current_state, self.table_events)
        self.previous_table_state = self.table_state
        self.table_state = current_state

        for event in self.table_events:
            print(f"[state-event] {event.description}")
        for update in self.ranging_tracker.last_updates:
            equity_text = (
                f"{update.average_equity:.3f}"
                if update.average_equity is not None
                else "unavailable"
            )
            print(
                f"[range] {update.player_name} {update.action_bucket}: "
                f"avg_equity={equity_text}, combos={update.combo_count}"
            )

        was_hero_turn = bool(self.previous_table_state and self.previous_table_state.was_hero_turn)
        if current_state.was_hero_turn and not was_hero_turn:
            print("[state-event] Hero turn started")
            print(
                "[state-event] Available actions: "
                f"{', '.join(self.format_action(action) for action in current_state.available_actions) or '-'}"
            )

        return current_state

    def format_action(self, action) -> str:
        if action.amount_bb is not None:
            return f"{action.name.title()} {action.amount_bb:.2f}bb"
        return action.text

    def hero_left_table(self, current_state: PokerTableState) -> bool:
        previous_hero = self.previous_table_state.hero if self.previous_table_state else None
        current_hero = current_state.hero

        if previous_hero is None:
            return False

        if current_hero is not None:
            return False

        return True

    def accept_tos(self) -> None:
        selector = "#accept-tos-button"  # accept tos
        if self.is_element_present_accepting_alerts(selector):
            element = self.get_element_accepting_alerts(selector)
            if element is not None:
                self.human_click(element, selector)

    def im_back(self) -> None:
        selector = "button.button-1.action-button.green.highlighted.iamback"  # I am back!
        if self.is_element_present_accepting_alerts(selector):
            element = self.get_element_accepting_alerts(selector)
            if element is not None:
                self.human_click(element, selector)

    def ignore_vc(self) -> None:
        selector = "button.button-1.gray.highlighted" # ignore voice / video chat button
        keywords = ("voice", "video", "audio", "camera", "microphone", "chat")
        for element in self.driver.find_elements(By.CSS_SELECTOR, selector):
            label = " ".join(
                value.lower()
                for value in (
                    element.text,
                    element.get_attribute("aria-label"),
                    element.get_attribute("title"),
                    element.get_attribute("class"),
                )
                if value
            )
            if any(keyword in label for keyword in keywords):
                self.human_click(element)
                return

    def leave_game(self) -> None:
        selector = "a.button-1.green.highlighted.med-button"  # leave game button
        self.wait_for_element_accepting_alerts(selector, timeout=10)
        element = self.get_element_accepting_alerts(selector)
        if element is not None:
            self.human_click(element, selector)

    def human_click(self, element: WebElement, selector: str | None = None) -> None:
        if selector is not None:
            rect = self.get_rect_from_selector(selector)
            if rect is None:
                self.wait_for_element_accepting_alerts(selector, timeout=5)
                rect = self.get_rect_from_selector(selector)
            if rect is None:
                return
        else:
            try:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
                    element,
                )
                rect = self.driver.execute_script(
                    """
                    const rect = arguments[0].getBoundingClientRect();
                    return {
                        left: rect.left,
                        top: rect.top,
                        width: rect.width,
                        height: rect.height,
                        viewportWidth: window.innerWidth,
                        viewportHeight: window.innerHeight
                    };
                    """,
                    element,
                )
            except (JavascriptException, StaleElementReferenceException):
                return

        if rect["width"] <= 0 or rect["height"] <= 0:
            return

        center_x = rect["left"] + rect["width"] / 2
        center_y = rect["top"] + rect["height"] / 2

        target_x = center_x + random.uniform(-rect["width"] * 0.25, rect["width"] * 0.25)
        target_y = center_y + random.uniform(-rect["height"] * 0.25, rect["height"] * 0.25)

        if self.mouse_x is None or self.mouse_y is None:
            self.mouse_x = max(
                5,
                min(
                    target_x + random.uniform(-120, 120),
                    rect["viewportWidth"] - 5,
                ),
            )
            self.mouse_y = max(
                5,
                min(
                    target_y + random.uniform(-90, 90),
                    rect["viewportHeight"] - 5,
                ),
            )

        start_x = self.mouse_x
        start_y = self.mouse_y

        control_1 = (
            start_x + random.uniform(-30, 30),
            start_y + random.uniform(-30, 30),
        )
        control_2 = (
            target_x + random.uniform(-30, 30),
            target_y + random.uniform(-30, 30),
        )

        def bezier_point(t: float) -> tuple[float, float]:
            x = (
                ((1 - t) ** 3 * start_x)
                + (3 * ((1 - t) ** 2) * t * control_1[0])
                + (3 * (1 - t) * (t ** 2) * control_2[0])
                + ((t ** 3) * target_x)
            )
            y = (
                ((1 - t) ** 3 * start_y)
                + (3 * ((1 - t) ** 2) * t * control_1[1])
                + (3 * (1 - t) * (t ** 2) * control_2[1])
                + ((t ** 3) * target_y)
            )
            return x, y

        steps = random.randint(14, 22)

        for step in range(1, steps + 1):
            point_x, point_y = bezier_point(step / steps)
            if not self.dispatch_mouse_event(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseMoved",
                    "x": point_x,
                    "y": point_y,
                    "button": "none",
                },
            ):
                self.mouse_x = point_x
                self.mouse_y = point_y
                return
            time.sleep(random.uniform(0.01, 0.04))

        time.sleep(random.uniform(0.08, 0.25))
        if not self.dispatch_mouse_event(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": target_x,
                "y": target_y,
                "button": "left",
                "clickCount": 1,
            },
        ):
            self.mouse_x = target_x
            self.mouse_y = target_y
            return

        time.sleep(random.uniform(0.04, 0.12))
        self.dispatch_mouse_event(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": target_x,
                "y": target_y,
                "button": "left",
                "clickCount": 1,
            },
        )

        self.mouse_x = target_x
        self.mouse_y = target_y
        self.accept_alert_if_present()

    def get_rect_from_selector(self, selector: str) -> dict | None:
        return self.driver.execute_script(
            """
            const element = document.querySelector(arguments[0]);
            if (!element) {
                return null;
            }
            element.scrollIntoView({block: 'center', inline: 'center'});
            const rect = element.getBoundingClientRect();
            return {
                left: rect.left,
                top: rect.top,
                width: rect.width,
                height: rect.height,
                viewportWidth: window.innerWidth,
                viewportHeight: window.innerHeight
            };
            """,
            selector,
        )

    def dispatch_mouse_event(self, command: str, params: dict) -> bool:
        try:
            self.driver.execute_cdp_cmd(command, params)
            return True
        except UnexpectedAlertPresentException:
            self.accept_alert_if_present()
            return False

    def accept_alert_if_present(self) -> bool:
        try:
            alert = self.driver.switch_to.alert
            print(f"Accepting alert: {alert.text}")
            alert.accept()
            time.sleep(0.5)
            return True
        except NoAlertPresentException:
            return False

    def wait_for_element_accepting_alerts(self, selector: str, timeout: int = 10) -> None:
        deadline = time.monotonic() + timeout
        last_error = None

        while time.monotonic() < deadline:
            try:
                if self.driver.find_elements(By.CSS_SELECTOR, selector):
                    return
            except UnexpectedAlertPresentException as error:
                last_error = error
                self.accept_alert_if_present()

        if last_error is not None:
            raise last_error
        raise TimeoutException(f"Timed out waiting for selector: {selector}")

    def get_element_accepting_alerts(self, selector: str) -> WebElement | None:
        deadline = time.monotonic() + 3

        while time.monotonic() < deadline:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                return elements[0] if elements else None
            except UnexpectedAlertPresentException:
                self.accept_alert_if_present()

        return None

    def switch_to_new_table_window(self, previous_windows: set[str], timeout: int = 15) -> None:
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            current_windows = self.driver.window_handles
            new_windows = [window for window in current_windows if window not in previous_windows]

            if new_windows:
                self.driver.close()
                self.driver.switch_to.window(new_windows[-1])
                return

            if "/games/" in self.driver.current_url:
                return

            time.sleep(0.25)

        self.driver.switch_to.window(self.driver.window_handles[-1])

    def open_room_link_if_needed(self, selector: str, previous_windows: set[str]) -> None:
        deadline = time.monotonic() + 5

        while time.monotonic() < deadline:
            if len(self.driver.window_handles) > len(previous_windows):
                return
            if "/games/" in self.driver.current_url:
                return
            time.sleep(0.25)

        room_href = self.driver.execute_script(
            """
            const element = document.querySelector(arguments[0]);
            return element ? element.href : null;
            """,
            selector,
        )

        if room_href:
            self.driver.execute_script("window.open(arguments[0], '_blank');", room_href)

    def wait_for_table_loaded(self, timeout: int = 60) -> None:
        deadline = time.monotonic() + timeout
        table_selectors = [
            ".table-player",
            ".table-game-type",
            ".game-decisions-ctn",
            ".table-cards",
        ]

        while time.monotonic() < deadline:
            self.accept_alert_if_present()

            if "/games/" in self.driver.current_url and any(
                self.is_element_present_accepting_alerts(selector)
                for selector in table_selectors
            ):
                return

            time.sleep(1)

        raise TimeoutException("Timed out waiting for the PokerNow table to load")

    def is_element_present_accepting_alerts(self, selector: str) -> bool:
        deadline = time.monotonic() + 3

        while time.monotonic() < deadline:
            try:
                return bool(self.driver.find_elements(By.CSS_SELECTOR, selector))
            except UnexpectedAlertPresentException:
                self.accept_alert_if_present()

        return False

    def recaptcha_needs_manual_completion(self) -> bool:
        try:
            return bool(
                self.driver.execute_script(
                    """
                    const visible = (element) => {
                        if (!element) return false;
                        const style = window.getComputedStyle(element);
                        const rect = element.getBoundingClientRect();
                        return style.display !== 'none'
                            && style.visibility !== 'hidden'
                            && Number(style.opacity || '1') > 0
                            && rect.width > 0
                            && rect.height > 0;
                    };

                    const recaptchaNodes = [
                        ...document.querySelectorAll(
                            '.g-recaptcha, [data-sitekey], iframe[src*="recaptcha"], iframe[title*="reCAPTCHA"], iframe[title*="recaptcha"]'
                        )
                    ];

                    if (!recaptchaNodes.some(visible)) {
                        return false;
                    }

                    const responses = [
                        ...document.querySelectorAll(
                            'textarea[name="g-recaptcha-response"], textarea[id*="g-recaptcha-response"]'
                        )
                    ];

                    return !responses.some((element) => (element.value || '').trim().length > 0);
                    """
                )
            )
        except Exception:
            return False

    def wait_for_recaptcha(self) -> None:
        while self.recaptcha_needs_manual_completion():
            time.sleep(3)

    def close(self) -> None:
        self.client.cookie_manager.save_cookies()
        if self.driver is not None:
            try:
                self.driver.quit()
            except OSError:
                pass
            self.driver = None

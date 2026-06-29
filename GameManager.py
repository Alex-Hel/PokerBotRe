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


class GameManager:
    driver = None
    element_helper = None
    client = None
    gm = None
    mouse_x = None
    mouse_y = None

    def __init__(self) -> None:
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
        self.client = PokerClient(self.driver, cookie_path='cookie_file.pkl')
        self.element_helper = ElementHelper(self.driver)
        self.mouse_x = None
        self.mouse_y = None
        self.gm = self.client.game_state_manager

    def start_gameplay_loop(self) -> None:
        self.client.navigate('https://network.pokernow.com/sng_tournaments')

        file_path = Path('cookie_file.pkl') #get login cookies / info
        if not file_path.is_file():
            input('enter login, press enter when done')
            self.client.cookie_manager.save_cookies()

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
            element = self.element_helper.get_element(queue_selector)
            self.human_click(element, queue_selector)
            self.accept_alert_if_present()
            time.sleep(3)

        self.wait_for_element_accepting_alerts(room_selector, timeout=900)
        element = self.element_helper.get_element(room_selector)
        self.human_click(element, room_selector)

        time.sleep(1)
        self.driver.switch_to.window(self.driver.window_handles[-1])

    def play_game(self) -> None:
        while True:
            time.sleep(1)
            self.ignore_vc()
            self.accept_tos()
            self.im_back()

            available_actions = self.client.action_helper.get_available_actions()
            if available_actions:
                if 'check' in available_actions:
                    print("Detected turn from action buttons: check")
                    self.human_click(available_actions['check'])
                    continue
                if 'fold' in available_actions:
                    print("Detected turn from action buttons: fold")
                    self.human_click(available_actions['fold'])
                    continue

            state = self.gm.get_game_state()
            if state.is_your_turn:
                available_actions = self.client.action_helper.get_available_actions()
                if 'check' in available_actions:
                    self.human_click(available_actions['check'])
                elif 'fold' in available_actions:
                    self.human_click(available_actions['fold'])

    def accept_tos(self) -> None:
        selector = "#accept-tos-button"  # accept tos
        if self.is_element_present_accepting_alerts(selector):
            element = self.element_helper.get_element(selector)
            self.human_click(element, selector)

    def im_back(self) -> None:
        selector = "button.button-1.action-button.green.highlighted.iamback"  # I am back!
        if self.is_element_present_accepting_alerts(selector):
            element = self.element_helper.get_element(selector)
            self.human_click(element, selector)

    def ignore_vc(self) -> None:
        selector = "button.button-1.gray.highlighted" # ignore voice / video chat button
        if self.is_element_present_accepting_alerts(selector):
            element = self.element_helper.get_element(selector)
            self.human_click(element, selector)

    def leave_game(self) -> None:
        selector = "a.button-1.green.highlighted.med-button"  # leave game button
        self.element_helper.wait_for_element(selector, timeout=10)
        element = self.element_helper.get_element(selector)
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
                self.element_helper.wait_for_element(selector, timeout=1)
                return
            except UnexpectedAlertPresentException as error:
                last_error = error
                self.accept_alert_if_present()
            except TimeoutException as error:
                last_error = error

        if last_error is not None:
            raise last_error
        raise TimeoutException(f"Timed out waiting for selector: {selector}")

    def is_element_present_accepting_alerts(self, selector: str) -> bool:
        deadline = time.monotonic() + 3

        while time.monotonic() < deadline:
            try:
                return self.element_helper.is_element_present(selector)
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

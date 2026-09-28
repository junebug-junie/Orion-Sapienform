"""Every RMP portal URL and DOM selector in one place.

Login and usage-download selectors below are verified against the live portal
(2026-09-28). Billing / forecast selectors are still UNVERIFIED guesses from the
PacifiCorp portal shape (nburns/pacificpower-import). A selector that stops matching
surfaces as an error status (login_form_failed / download_failed / wrong_day_download /
bill_rows_empty), never as a quiet success.
"""

USAGE_PATH = "/secure/my-account/energy-usage"
BILLING_PATH = "/secure/my-account/billing-payment-history"

LOGIN_URL_MARKERS = ("b2clogin.com", "b2c_1a_pac_signin", "/signin", "/login")
LOGGED_IN_MARKER = "text=Sign Out"

# Verified 2026-09-28 against the live /idm/login page: the sign-in form is an Azure B2C
# page embedded in this iframe.
LOGIN_FRAME = "iframe#loginframe"
LOGIN_USERNAME = "#signInName"
LOGIN_PASSWORD = "#password"
LOGIN_SUBMIT = "button#next"
LOGIN_WAIT_SEC = 60.0

# Verified 2026-09-28 on /secure/my-account/energy-usage. The Green Button download follows
# the page's period dropdown: only "One Day" is hourly (One Week/Month are daily, Two Year
# monthly), so the fetcher downloads one day at a time. The "Show usage through" date input
# carries min/max attributes (e.g. 2026-09-26T00:00:00+00:00) = the days RMP has; entering a
# date outside them leaves the download link dead for the rest of the session.
USAGE_PERIOD_OPTIONS = ("Two Year", "One Year", "One Month", "One Week", "One Day")
USAGE_PERIOD_ONE_DAY = "One Day"
USAGE_THROUGH_INPUT = "input[placeholder^='Show usage through']"
GREEN_BUTTON_DOWNLOAD = "a:has-text('DOWNLOAD GREEN BUTTON DATA')"
DOWNLOAD_WAIT_SEC = 30.0
SETTLE_AFTER_DATE_SEC = 1.5

BILL_ROW = "[data-testid='billing-history-row']"
BILL_ROW_FIELDS = {
    "billing_period": "[data-testid='billing-period']",
    "kwh": "[data-testid='kwh-used']",
    "current_charges": "[data-testid='current-charges']",
    "amount_due": "[data-testid='amount-due']",
    "due_date": "[data-testid='due-date']",
}

FORECAST_PANEL = "[data-testid='bill-projection']"
FORECAST_FIELDS = {
    "billing_period": "[data-testid='projection-period']",
    "projected_total": "[data-testid='projected-amount']",
    "projected_kwh": "[data-testid='projected-kwh']",
}

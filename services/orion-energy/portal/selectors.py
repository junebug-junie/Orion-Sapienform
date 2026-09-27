"""Every RMP portal URL and DOM selector in one place.

UNVERIFIED: written from the PacifiCorp portal shape (nburns/pacificpower-import) and
not yet checked against a live rockymountainpower.net session. The first live spike
edits only this file. A selector that stops matching surfaces as an error status
(empty download / bill_rows_empty), never as a quiet success.
"""

USAGE_PATH = "/secure/my-account/energy-usage"
BILLING_PATH = "/secure/my-account/billing-payment-history"

LOGIN_URL_MARKERS = ("b2clogin.com", "b2c_1a_pac_signin", "/signin", "/login")
LOGGED_IN_MARKER = "text=Sign Out"

GREEN_BUTTON_OPEN = "text=Green Button"
GREEN_BUTTON_FROM = "input[name='startDate']"
GREEN_BUTTON_TO = "input[name='endDate']"
GREEN_BUTTON_DOWNLOAD = "button:has-text('Download')"
GREEN_BUTTON_DATE_FORMAT = "%m/%d/%Y"

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

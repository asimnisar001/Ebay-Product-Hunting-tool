"""
eBay Product Hunter
--------------------
A Streamlit application for researching eBay products: searching active
listings via the official eBay Browse API, viewing price statistics,
running a landed-cost / profit calculator, and maintaining a session
watchlist with CSV export.

Setup
-----
1. Create an eBay Developer account: https://developer.ebay.com
2. Create a "Keyset" (Application) and get your Client ID / Client Secret.
3. Set the following environment variables (or enter them in the sidebar
   at runtime):
       EBAY_CLIENT_ID
       EBAY_CLIENT_SECRET
       EBAY_ENV        ("SANDBOX" or "PRODUCTION", defaults to PRODUCTION)
       EBAY_MARKETPLACE ("EBAY_US", "EBAY_GB", etc. defaults to EBAY_US)

Run
---
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EBAY_OAUTH_URLS = {
    "PRODUCTION": "https://api.ebay.com/identity/v1/oauth2/token",
    "SANDBOX": "https://api.sandbox.ebay.com/identity/v1/oauth2/token",
}
EBAY_BROWSE_URLS = {
    "PRODUCTION": "https://api.ebay.com/buy/browse/v1/item_summary/search",
    "SANDBOX": "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search",
}
DEFAULT_SCOPE = "https://api.ebay.com/oauth/api_scope"

MARKETPLACE_FEE_DEFAULTS = {
    "eBay final value fee (%)": 13.25,
    "Payment processing fee (%)": 2.9,
    "Payment processing fixed ($)": 0.30,
}

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class EbayCredentials:
    client_id: str
    client_secret: str
    environment: str = "PRODUCTION"
    marketplace_id: str = "EBAY_US"


@dataclass
class SearchStats:
    count: int = 0
    avg_price: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    median_price: float = 0.0
    currency: str = "USD"


# --------------------------------------------------------------------------- #
# eBay API client
# --------------------------------------------------------------------------- #


class EbayApiError(Exception):
    """Raised when the eBay API returns an error response."""


def _basic_auth_header(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return base64.b64encode(raw).decode("utf-8")


@st.cache_data(show_spinner=False, ttl=3300)  # eBay tokens last ~2hrs; refresh at 55min
def get_oauth_token(client_id: str, client_secret: str, environment: str) -> str:
    """Fetch an OAuth2 client-credentials access token from eBay.

    Cached by Streamlit so we don't request a new token on every search.
    """
    if not client_id or not client_secret:
        raise EbayApiError(
            "Missing eBay API credentials. Set EBAY_CLIENT_ID / EBAY_CLIENT_SECRET "
            "environment variables or enter them in the sidebar."
        )

    url = EBAY_OAUTH_URLS.get(environment, EBAY_OAUTH_URLS["PRODUCTION"])
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {_basic_auth_header(client_id, client_secret)}",
    }
    data = {"grant_type": "client_credentials", "scope": DEFAULT_SCOPE}

    resp = requests.post(url, headers=headers, data=data, timeout=15)
    if resp.status_code != 200:
        raise EbayApiError(f"OAuth token request failed ({resp.status_code}): {resp.text[:300]}")

    payload = resp.json()
    token = payload.get("access_token")
    if not token:
        raise EbayApiError("OAuth response did not contain an access_token.")
    return token


def search_ebay_items(
    creds: EbayCredentials,
    keyword: str,
    limit: int = 50,
    condition: str | None = None,
    sort: str | None = None,
) -> list[dict[str, Any]]:
    """Search active eBay listings via the Browse API.

    Returns a list of raw item summary dicts from eBay.
    """
    token = get_oauth_token(creds.client_id, creds.client_secret, creds.environment)
    url = EBAY_BROWSE_URLS.get(creds.environment, EBAY_BROWSE_URLS["PRODUCTION"])

    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": creds.marketplace_id,
        "Content-Type": "application/json",
    }
    params: dict[str, Any] = {"q": keyword, "limit": min(max(limit, 1), 200)}
    filters = []
    if condition and condition != "Any":
        filters.append(f"conditions:{{{condition.upper()}}}")
    if filters:
        params["filter"] = ",".join(filters)
    if sort:
        params["sort"] = sort

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    if resp.status_code != 200:
        raise EbayApiError(f"Search request failed ({resp.status_code}): {resp.text[:300]}")

    return resp.json().get("itemSummaries", [])


def items_to_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten raw eBay item summaries into a tidy DataFrame."""
    rows = []
    for item in items:
        price_obj = item.get("price", {}) or {}
        shipping_options = item.get("shippingOptions", [{}])
        shipping_cost = None
        if shipping_options:
            shipping_cost_obj = shipping_options[0].get("shippingCost", {})
            shipping_cost = shipping_cost_obj.get("value")

        rows.append(
            {
                "Title": item.get("title"),
                "Price": float(price_obj.get("value", 0) or 0),
                "Currency": price_obj.get("currency", "USD"),
                "Shipping": float(shipping_cost) if shipping_cost is not None else 0.0,
                "Condition": item.get("condition"),
                "Seller": (item.get("seller") or {}).get("username"),
                "Seller Feedback %": (item.get("seller") or {}).get("feedbackPercentage"),
                "Listing Type": (item.get("buyingOptions") or [None])[0],
                "Item URL": item.get("itemWebUrl"),
                "Image URL": (item.get("image") or {}).get("imageUrl"),
                "Item ID": item.get("itemId"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Landed Price"] = df["Price"] + df["Shipping"]
    return df


def compute_stats(df: pd.DataFrame) -> SearchStats:
    if df.empty:
        return SearchStats()
    return SearchStats(
        count=len(df),
        avg_price=round(df["Landed Price"].mean(), 2),
        min_price=round(df["Landed Price"].min(), 2),
        max_price=round(df["Landed Price"].max(), 2),
        median_price=round(df["Landed Price"].median(), 2),
        currency=df["Currency"].mode().iat[0] if not df["Currency"].mode().empty else "USD",
    )


# --------------------------------------------------------------------------- #
# Profit calculator
# --------------------------------------------------------------------------- #


def calculate_profit(
    sale_price: float,
    item_cost: float,
    shipping_cost: float,
    other_costs: float,
    final_value_fee_pct: float,
    payment_fee_pct: float,
    payment_fee_fixed: float,
) -> dict[str, float]:
    """Compute net profit, margin, and ROI for a single sale."""
    final_value_fee = sale_price * (final_value_fee_pct / 100)
    payment_fee = sale_price * (payment_fee_pct / 100) + payment_fee_fixed
    total_fees = final_value_fee + payment_fee
    total_cost = item_cost + shipping_cost + other_costs + total_fees

    net_profit = sale_price - total_cost
    margin = (net_profit / sale_price * 100) if sale_price else 0.0
    roi = (net_profit / (item_cost + shipping_cost + other_costs) * 100) if (item_cost + shipping_cost + other_costs) else 0.0
    breakeven_price = total_cost - net_profit if sale_price else 0.0  # == cost basis + fees at that price (approx)

    return {
        "final_value_fee": round(final_value_fee, 2),
        "payment_fee": round(payment_fee, 2),
        "total_fees": round(total_fees, 2),
        "total_cost": round(total_cost, 2),
        "net_profit": round(net_profit, 2),
        "margin_pct": round(margin, 2),
        "roi_pct": round(roi, 2),
    }


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #


def init_session_state() -> None:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = pd.DataFrame()
    if "last_results" not in st.session_state:
        st.session_state.last_results = pd.DataFrame()


def sidebar_credentials() -> EbayCredentials:
    st.sidebar.header("eBay API Credentials")
    client_id = st.sidebar.text_input(
        "Client ID", value=os.getenv("EBAY_CLIENT_ID", ""), type="default"
    )
    client_secret = st.sidebar.text_input(
        "Client Secret", value=os.getenv("EBAY_CLIENT_SECRET", ""), type="password"
    )
    environment = st.sidebar.selectbox(
        "Environment", ["PRODUCTION", "SANDBOX"],
        index=0 if os.getenv("EBAY_ENV", "PRODUCTION") == "PRODUCTION" else 1,
    )
    marketplace_id = st.sidebar.selectbox(
        "Marketplace",
        ["EBAY_US", "EBAY_GB", "EBAY_DE", "EBAY_AU", "EBAY_CA", "EBAY_FR", "EBAY_IT"],
        index=0,
    )
    st.sidebar.caption(
        "Credentials are kept only in this browser session and never written to disk."
    )
    return EbayCredentials(client_id, client_secret, environment, marketplace_id)


def render_search_tab(creds: EbayCredentials) -> None:
    st.subheader("Search eBay Listings")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        keyword = st.text_input("Product name, keyword, UPC, or brand", "")
    with col2:
        condition = st.selectbox("Condition", ["Any", "NEW", "USED"])
    with col3:
        limit = st.number_input("Results", min_value=5, max_value=200, value=50, step=5)

    sort_choice = st.selectbox(
        "Sort by", ["Best Match", "Price: Low to High", "Price: High to Low", "Newly Listed"]
    )
    sort_map = {
        "Best Match": None,
        "Price: Low to High": "price",
        "Price: High to Low": "-price",
        "Newly Listed": "newlyListed",
    }

    if st.button("Search", type="primary", use_container_width=True) and keyword.strip():
        with st.spinner("Querying eBay Browse API..."):
            try:
                items = search_ebay_items(
                    creds, keyword.strip(), int(limit), condition, sort_map[sort_choice]
                )
                df = items_to_dataframe(items)
                st.session_state.last_results = df
            except EbayApiError as exc:
                st.error(str(exc))
                return

    df = st.session_state.last_results
    if df.empty:
        st.info("Run a search to see results here.")
        return

    stats = compute_stats(df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Listings found", stats.count)
    m2.metric("Avg. landed price", f"{stats.avg_price} {stats.currency}")
    m3.metric("Lowest", f"{stats.min_price} {stats.currency}")
    m4.metric("Highest", f"{stats.max_price} {stats.currency}")
    m5.metric("Median", f"{stats.median_price} {stats.currency}")

    fig = px.histogram(
        df, x="Landed Price", nbins=20, title="Price Distribution (item + shipping)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[
            [
                "Title", "Price", "Shipping", "Landed Price", "Currency",
                "Condition", "Seller", "Seller Feedback %", "Listing Type", "Item URL",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"ebay_search_{int(time.time())}.csv",
        mime="text/csv",
    )

    st.markdown("**Add a listing to your watchlist:**")
    if not df.empty:
        selected_title = st.selectbox("Select a listing", df["Title"].tolist())
        if st.button("Add to watchlist"):
            row = df[df["Title"] == selected_title].head(1)
            st.session_state.watchlist = pd.concat(
                [st.session_state.watchlist, row], ignore_index=True
            ).drop_duplicates(subset=["Item ID"])
            st.success(f"Added '{selected_title[:60]}...' to watchlist.")


def render_profit_tab() -> None:
    st.subheader("Profit & ROI Calculator")

    col1, col2 = st.columns(2)
    with col1:
        sale_price = st.number_input("Expected sale price", min_value=0.0, value=30.0, step=0.5)
        item_cost = st.number_input("Item / supplier cost", min_value=0.0, value=10.0, step=0.5)
        shipping_cost = st.number_input("Shipping cost (to buyer)", min_value=0.0, value=4.0, step=0.5)
        other_costs = st.number_input(
            "Other costs (packaging, storage, ads, returns)", min_value=0.0, value=1.0, step=0.5
        )
    with col2:
        fvf_pct = st.number_input(
            "eBay final value fee (%)", min_value=0.0, value=MARKETPLACE_FEE_DEFAULTS["eBay final value fee (%)"]
        )
        pay_pct = st.number_input(
            "Payment processing fee (%)", min_value=0.0, value=MARKETPLACE_FEE_DEFAULTS["Payment processing fee (%)"]
        )
        pay_fixed = st.number_input(
            "Payment processing fixed fee", min_value=0.0, value=MARKETPLACE_FEE_DEFAULTS["Payment processing fixed ($)"]
        )

    result = calculate_profit(
        sale_price, item_cost, shipping_cost, other_costs, fvf_pct, pay_pct, pay_fixed
    )

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net profit", f"${result['net_profit']}")
    c2.metric("Margin", f"{result['margin_pct']}%")
    c3.metric("ROI", f"{result['roi_pct']}%")
    c4.metric("Total fees", f"${result['total_fees']}")

    with st.expander("Full cost breakdown"):
        st.json(result)


def render_watchlist_tab() -> None:
    st.subheader("Watchlist")
    wl = st.session_state.watchlist
    if wl.empty:
        st.info("No items yet — add listings from the Search tab.")
        return

    st.dataframe(
        wl[["Title", "Price", "Shipping", "Landed Price", "Currency", "Seller", "Item URL"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download watchlist as CSV",
        data=wl.to_csv(index=False).encode("utf-8"),
        file_name="ebay_watchlist.csv",
        mime="text/csv",
    )
    if st.button("Clear watchlist"):
        st.session_state.watchlist = pd.DataFrame()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="eBay Product Hunter", page_icon="🔎", layout="wide")
    init_session_state()

    st.title("🔎 eBay Product Hunter")
    st.caption(
        "Search live eBay listings, analyze pricing, and calculate profit margins "
        "before you buy inventory."
    )

    creds = sidebar_credentials()

    tab_search, tab_profit, tab_watchlist = st.tabs(["Search", "Profit Calculator", "Watchlist"])
    with tab_search:
        render_search_tab(creds)
    with tab_profit:
        render_profit_tab()
    with tab_watchlist:
        render_watchlist_tab()
       [     UTC     ] Logs for ebay-appuct-hunting-tool-imtf2x6en3srpmlfdjgrgr.streamlit.app/

────────────────────────────────────────────────────────────────────────────────────────

[17:32:49] 🚀 Starting up repository: 'ebay-product-hunting-tool', branch: 'main', main module: 'aap.py'

[17:32:49] 🐙 Cloning repository...

[17:32:49] 🐙 Cloning into '/mount/src/ebay-product-hunting-tool'...

[17:32:50] 🐙 Cloned repository!

[17:32:50] 🐙 Pulling code changes from Github...

[17:32:50] 📦 Processing dependencies...


──────────────────────────────────────── uv ───────────────────────────────────────────


Using uv pip install.

Using Python 3.14.6 environment at /home/adminuser/venv

Resolved 45 packages in 400ms

Prepared 45 packages in 4.48s

Installed 45 packages in 576ms

 + altair==6.2.2

 + anyio==4.14.2

 + attrs==26.1.0

 + blinker==1.9.0

 + cachetools==7.1.4

 + certifi==2026.6.17[2026-07-16 17:32:56.089881] 

 + charset-normalizer==3.4.9

 + click==8.4.2

 + gitdb==4.0.12

 + gitpython==3.1.52

 + h11==0.16.0[2026-07-16 17:32:56.090402] 

 + httptools==0.8.0

 + idna==3.18

 + itsdangerous==2.2.0

 + jinja2==3.1.6[2026-07-16 17:32:56.090918] 

 + jsonschema==4.26.0

 + jsonschema-specifications==2025.9.1

 + markupsafe==3.0.3

 +[2026-07-16 17:32:56.091124]  narwhals==2.24.0

 + numpy==2.5.1

 + packaging==26.2

 + pandas==2.3.3

 + pillow==12.3.0

 + plotly==5.24.1

 + protobuf[2026-07-16 17:32:56.091574] ==7.35.1

 + pyarrow==24.0.0

 + pydeck==0.9.3

 + python-dateutil==2.9.0[2026-07-16 17:32:56.091750] .post0

 + python-multipart==0.0.32

 + pytz==2026.2

 [2026-07-16 17:32:56.091924] + referencing==0.37.0[2026-07-16 17:32:56.092079] 

 + requests==2.34.2

 + rpds-py==2026.6.3

 + six==1.17.0

 + smmap==5.0.3

 + starlette==1.3.1

 [2026-07-16 17:32:56.092226] + streamlit==1.59.2

 + tenacity==9.1.4

 + toml==0.10.2[2026-07-16 17:32:56.092408] 

 + typing-extensions==4.16.0

 + tzdata==2026.3

 + urllib3==2.7.0

 + uvicorn[2026-07-16 17:32:56.092548] ==0.51.0

 + watchdog==6.0.0

 + websockets==[2026-07-16 17:32:56.092685] 16.1

Checking if Streamlit is installed

Found Streamlit version 1.59.2 in the environment

Installing rich for an improved exception logging

Using uv pip install.

Using Python 3.14.6 environment at /home/adminuser/venv

Resolved 4 packages in 133ms

Prepared 4 packages in 105ms

Installed 4 packages in 15ms

 + markdown-it-py==4.2.0

 +[2026-07-16 17:32:58.932791]  mdurl==0.1.2

 + pygments==2.20.0

 + rich==15.0.0


────────────────────────────────────────────────────────────────────────────────────────


[17:32:59] 🐍 Python dependencies were installed from /mount/src/ebay-product-hunting-tool/requirements.txt using uv.

Check if streamlit is installed

Streamlit is already installed

[17:33:00] 📦 Processed dependencies!

2026-07-16 17:33:02.608 Uvicorn server started on :::8501


if __name__ == "__main__":
    main()

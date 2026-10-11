"""Vertical profiles: what changes between a café, a salon, a gym, a retailer or a B2B supplier.
A profile is plain settings; a vertical model later is a profile plus weights trained on that
vertical's data. Hosts can register their own (register()) or pass the numbers directly."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Vertical:
    name: str
    window_days: int = 45          # "came back" = a visit/order within this many days
    churn_days: int = 90           # quiet this long = gone (the formula baseline)
    return_window_days: int = 14   # how long after a message a return still counts
    currency: str = "₹"
    customer_noun: str = "customers"
    item_noun: str = "items"
    mechanisms: tuple = field(default=("spend_more", "bundle", "add_on", "winback_offer", "referral",
                                       "new_arrival", "slow_mover", "loyalty_tier", "timing"))


_REGISTRY: dict[str, Vertical] = {}


def register(v: Vertical) -> Vertical:
    _REGISTRY[v.name] = v
    return v


def get(name: str | None) -> Vertical:
    """The profile for `name`, or the generic one."""
    return _REGISTRY.get((name or "").lower(), _REGISTRY["generic"])


register(Vertical("generic"))
register(Vertical("restaurant", window_days=45, churn_days=90, customer_noun="diners", item_noun="dishes"))
register(Vertical("cafe", window_days=21, churn_days=60, customer_noun="guests", item_noun="items on the menu"))
register(Vertical("salon", window_days=60, churn_days=120, customer_noun="clients", item_noun="services"))
register(Vertical("gym", window_days=14, churn_days=45, customer_noun="members", item_noun="plans",
                  mechanisms=("winback_offer", "referral", "loyalty_tier", "timing")))
register(Vertical("retail", window_days=60, churn_days=150, customer_noun="shoppers", item_noun="products"))
register(Vertical("b2b", window_days=90, churn_days=180, return_window_days=30, customer_noun="accounts",
                  item_noun="SKUs", mechanisms=("spend_more", "bundle", "add_on", "winback_offer", "new_arrival",
                                                "slow_mover", "timing")))


def with_overrides(v: Vertical, **changes) -> Vertical:
    return replace(v, **changes)

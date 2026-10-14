"""vertiq: decision models for customers, revenue and inventory, for any business vertical.

Plain data in, plain data out. vertiq never imports a web framework, an ORM or any host product:
a host (a loyalty app, a POS, an enterprise CRM) reads its own database into the data
contract (vertiq.contract), calls these functions, and stores the models (plain dicts) wherever
it likes.

  learn        logistic and ridge models over named features, AUC, calibration, tests, FDR control
  features     what is known about a customer (account) at a moment, from their events only
  churn        the chance a customer returns within a window, shared across businesses and verticals
  uplift       who a message or an offer actually brings back, learned from control groups
  value        revenue per campaign vs control, bill size, referrers
  affinity     choices: combos, cross-sell, next items, refill cycles
  insights     patterns tested before they are shown
  inventory    reorder, slow movers, new arrivals, margins
  offers       spend-more offers sized per customer
  trends       trends matched to what a business sells and a customer buys
  research     web tactics, kept only with a source and a mechanism that can test them
  dataset      checks, masking and exports for training a language model
  experiments  comparing message variants against each other and a control group
  verticals    per-vertical settings (windows, nouns, currency, tactics)
"""
__version__ = "0.2.1"

from sqlalchemy import Column, DateTime, Float, Index, String
from sqlalchemy.sql import func

from app.db import Base


class OrionEnergyAttributionHourlySQL(Base):
    """Who each host's above-rest energy can be charged to, one row per (UTC hour, host).
    Single writer: app/orion_energy_attribution.py. Arithmetic: orion/energy/act_energy.py
    ``attribute_host_window``, the same split the per-act meter uses.

    All energies are Wh over the covered part of the hour:

    * ``rest_wh``: the pinned rest (``rest_w``, config/physiology/rest_body.yaml) x covered time.
    * ``above_rest_wh``: chassis watts (at the wall) above ``rest_w``, clamped at 0 per second.
    * ``sponsored_wh``: the sponsored containers' CPU share of it (Juniper's work, e.g. the dojo).
    * ``metered_wh``: the share of GPU leases whose turn has an act: ATTRIBUTABLE to acts. What acts
      were actually charged is sum(grammar_acts.cost_json.wh_by_host); the physiology report sets the
      two side by side per day (an act fill that failed, fell back to GPU-only, or double counted a
      shared correlation_id makes them differ).
    * ``leased_unmetered_wh``: the share of GPU leases whose turn has NO act (part of unattributed).
    * ``unattributed_wh`` = above_rest - sponsored - metered: work not yet metered (CPU of
      non-sponsored processes, unleased cards, leases with no act, above-rest watts with no load).
      Stored as computed, NEVER clipped: a negative value means over-attribution and is logged.
    * ``split_unknown_sec``: seconds with a chassis reading but a hole in the split (a CPU or card
      reading missing); their above-rest energy is all in unattributed.

    Daily coverage (for the energy allowance's coverage gate) = sum(metered) /
    (sum(above_rest) - sum(sponsored)) over the day's rows. An hour with no chassis reading has no
    row: unknown, never 0.
    """

    __tablename__ = "orion_energy_attribution_hourly"

    hour_start = Column(DateTime(timezone=True), primary_key=True)
    host = Column(String, primary_key=True)
    rest_w = Column(Float, nullable=False)
    rest_wh = Column(Float, nullable=False)
    above_rest_wh = Column(Float, nullable=False)
    sponsored_wh = Column(Float, nullable=False)
    metered_wh = Column(Float, nullable=False)
    leased_unmetered_wh = Column(Float, nullable=False)
    unattributed_wh = Column(Float, nullable=False)
    covered_sec = Column(Float, nullable=False)
    split_unknown_sec = Column(Float, nullable=False)
    rest_source = Column(String, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (Index("idx_orion_energy_attribution_hourly_hour", "hour_start"),)

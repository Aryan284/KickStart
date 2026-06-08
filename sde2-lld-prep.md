# SDE2 LLD Interview Prep

DSA-flavored Low-Level Design problems asked in SDE2 interviews, with a consistent walkthrough template, working Python code, and follow-up handling.

## The Interview Template

Every problem follows this script:

1. **Clarifying questions** — input/output, edge cases, constraints
2. **Data type decisions** — money, time, IDs, with one-line justifications
3. **Brute force** — naive approach with complexity, why it's insufficient
4. **Optimal approach** — key insight, data structure, complexity per method
5. **OOP structure** — classes, design patterns when they fit
6. **Trade-offs** — what you considered and rejected
7. **Code** — minimal libraries, clear comments, demo at the end

## Coding Style

- **No regex** — hand-rolled parsing shows you understand grammars
- **No dataclass / itertools** — plain classes are universally understood
- **`Decimal` for money** — never `float`
- **`datetime` (UTC) for time** — timezone-safe, microsecond precision
- Minimal libraries — standard library only

---

## Table of Contents

1. [Delivery Cost Tracking](#1-delivery-cost-tracking) — running totals (read-write asymmetry)
2. [Corporate Rules Engine](#2-corporate-rules-engine) — Strategy / Open-Closed
3. [Music Player (Spotify)](#3-music-player-spotify) — Set + OrderedDict
4. [Employee Access Management](#4-employee-access-management) — nested dict + Set
5. [Excel Spreadsheet](#5-excel-spreadsheet) — Strategy + DFS cycle detection
6. [Key-Value Store with Transactions](#6-key-value-store-with-transactions) — stack of dicts + sentinel

---

# 1. Delivery Cost Tracking

## Problem

A logistics company employs thousands of drivers. Track their work and calculate costs.

```
add_driver(driverId)
add_delivery(driverId, startTime, endTime)
get_total_cost()
```

Requirement expansions during the interview:
- **Phase 2:** `pay_up_to_time(timestamp)`, `get_cost_to_be_paid()`
- **Phase 3:** `get_max_active_drivers_in_last_24_hours(currentTime)`

## What This Interview Tests

The interviewer is **not** testing CRUD. They probe four things:

1. **Data modeling** — immutability, separation of concerns
2. **Scalability thinking** — read/write asymmetry, hot paths
3. **Complexity analysis** — O(1) reads, defending optimality
4. **Future extensibility** — designing so requirement expansions slot in cleanly

Common probes you should be ready for:
- *"What happens with millions of deliveries?"*
- *"What if `get_total_cost` is called thousands of times per second?"*

## Conversation Script (Turn-by-Turn)

### Phase 1: Three Initial APIs

**Interviewer:** "Design a delivery cost tracking system. Three APIs: `add_driver(driverId)`, `add_delivery(driverId, startTime, endTime)`, and `get_total_cost()`."

**You — clarifying questions:**

> "Before I design, a few clarifying questions:
> 1. **Cost calculation** — flat rate, hourly, distance-based?
> 2. **Read vs write ratio** — is `get_total_cost` queried frequently, like a dashboard?
> 3. **Scale** — thousands of deliveries a day, millions over time?
> 4. **Currency / precision** — assuming USD with cents-level precision since financial accuracy matters.
> 5. **Time format** — Unix epoch, UTC datetime?
> 6. **Are inputs trusted, or do I need to validate?**"

**Interviewer:** "Hourly rate × duration. `get_total_cost` runs thousands of times per second — it powers a live dashboard. Millions of deliveries over time. USD. Inputs valid."

**You — design:**

> "Two key observations:
>
> 1. **`get_total_cost` is overwhelmingly the hot path** — thousands of QPS, while `add_delivery` is much rarer. I should shift computation to write time so reads stay cheap.
>
> 2. **Financial data benefits from immutability.** Deliveries should be append-only — once recorded, never modified — for audit trails, dispute resolution, and historical reporting.
>
> Data model:
> ```
> Driver:   { id, hourly_rate }
> Delivery: { driver_id, start_time, end_time, cost }   # immutable
> ```
>
> Storage:
> - `drivers: Dict[driver_id → Driver]`
> - `deliveries: List[Delivery]` — append-only history
> - `total_cost: Decimal` — running aggregate
>
> The naive approach is to compute total cost on every read by iterating all deliveries — O(N) per call. With thousands of QPS and millions of records, that's a non-starter.
>
> **I'll maintain a running total updated at write time:**
>
> ```python
> def add_delivery(driver_id, start, end):
>     cost = drivers[driver_id].hourly_rate * duration_hours(start, end)
>     deliveries.append(Delivery(driver_id, start, end, cost))
>     total_cost += cost                                  # O(1)
>
> def get_total_cost():
>     return total_cost                                   # O(1)
> ```
>
> Both `add_delivery` and `get_total_cost` are now O(1). Classic read-write trade-off — when reads dominate, do extra work during writes.
>
> Notes on data types:
> - **`Decimal`, not `float`** — float drift compounds across millions of operations.
> - **Lock the rate at delivery time** — store `cost` directly on the `Delivery` so historical records reflect the rate that was in effect when the work was done.
> - **UTC datetime** for timestamps — timezone-safe."

**Probe — "Why store the deliveries list if you're maintaining a running total?"**

> "Two reasons:
> 1. **Future analytics** — questions like 'cost by driver' or 'cost in a time window' need history. Append is O(1) — cheap to maintain.
> 2. **Audit and dispute resolution** — if a delivery is contested, we need the record. Throwing away data closes off recovery options.
>
> In production this list lives in a database — the in-memory list is just a placeholder."

**Probe — "What if the running total drifts due to a bug?"**

> "Aggregates can drift from truth — silent data corruption is the worst kind. Two safeguards:
> 1. **Reconciliation job** — periodically recompute `total_cost` from the source list and alert on mismatch.
> 2. **Source-of-truth treatment** — the deliveries list is canonical. The aggregate is a *cache*. If we ever doubt it, recompute.
>
> Standard pattern: append-only event log + materialized aggregate."

### Phase 2: Payment Tracking

**Interviewer:** "Now add `pay_up_to_time(timestamp)` — marks all deliveries ending at or before this time as paid. And `get_cost_to_be_paid()` returns the unpaid total."

**You — clarifying first:**

> "How often is `pay_up_to_time` called? Real-time settlement, or batch payroll?"

**Interviewer:** "Batch — typically once a day or once a week."

**You — design:**

> "Confirms reads stay the hot path. `pay_up_to_time` is rare, so I can afford O(N) work there.
>
> Key principle: **don't mutate delivery records when payment happens.** Two reasons:
> 1. **Auditability** — disputes need the original record intact.
> 2. **Separation of concerns** — delivery creation and payment are independent operations.
>
> Instead of marking each delivery as paid, **track payments as a moving boundary on the timeline**:
>
> ```
> total_cost: Decimal           # all deliveries, lifetime
> total_paid: Decimal           # cumulative paid amount
> last_paid_time: datetime      # the moving boundary
> ```
>
> ```python
> def pay_up_to_time(t):
>     for d in deliveries:
>         if last_paid_time < d.end_time <= t:
>             total_paid += d.cost
>     last_paid_time = t
>
> def get_cost_to_be_paid():
>     return total_cost - total_paid                     # O(1)
> ```
>
> `get_cost_to_be_paid` stays O(1) — same hot-path optimization. `pay_up_to_time` is O(N) worst case, acceptable because:
> - It runs rarely (batch).
> - Each delivery is processed at most once across all `pay_up_to_time` calls — amortized O(1) per delivery.
> - Production: index by `end_time` for O(K log N), where K = newly paid items.
>
> **Crucially, delivery records are never mutated.** All historical data is preserved. The boundary is the only mutable state, and it only moves forward."

**Probe — "What if I need to know exactly which deliveries were paid in a specific batch?"**

> "Boundary loses that granularity. Two options:
> 1. **`payment_batches: List[(start_time, end_time, amount)]` log.** Each `pay_up_to_time` call appends an entry. Audit reconstruction is a list scan.
> 2. **Add a `paid_at` timestamp on each delivery.** One-time mutation, only metadata — original cost/times stay frozen.
>
> For most logistics systems I'd go with option 1 — keep deliveries fully immutable, track payments in a separate log."

**Probe — "Overlapping `pay_up_to_time` calls?"**

> "The boundary handles it naturally — we only count `last_paid_time < d.end_time <= t`. If someone calls `pay_up_to_time(t1)` then `pay_up_to_time(t2)` with `t2 < t1`, treat the second call as a no-op — the boundary already moved past `t2`. Document this idempotency clearly: calling with an earlier time should never roll back the boundary, because that would mean reverting payments."

**Optional follow-up — when the heap becomes the right answer:**

> "If `pay_up_to_time` becomes a hot path — say, real-time settlement — I'd switch to a min-heap of unpaid deliveries keyed on `end_time`. `pay_up_to_time` becomes O(K log N) amortized at the cost of O(log N) per `add_delivery`. For batch payroll, the running-totals approach wins because writes stay O(1) and history is fully preserved."

### Phase 3: Driver Analytics

**Interviewer:** "Add `get_max_active_drivers_in_last_24_hours(current_time)` — max drivers simultaneously active during any moment in the last 24 hours."

**You — recognize the shift:**

> "This is qualitatively different — **time-windowed analytics**, not transactional aggregation. The challenge is repeatedly answering the same question over shifting windows. Naive O(N) scans don't scale.
>
> Naive: filter deliveries overlapping the 24h window, run sweep-line, return peak. O(N log N) per query.
>
> Several scaling strategies:
>
> ### Option 1: Sliding-window cache + sweep-line
>
> Cache deliveries that intersect the 24h window in a sorted structure. As time advances, evict deliveries that fall outside.
>
> O(W log W) per query where W = window size. W << N.
>
> ### Option 2: Bucketed counts (best fit here)
>
> Divide timeline into fixed buckets (per-minute). For each bucket, track which drivers were active.
>
> ```python
> buckets: Dict[minute_index → Set[driver_id]]
>
> def add_delivery(driver_id, start, end):
>     for m in range(minute_of(start), minute_of(end) + 1):
>         buckets[m].add(driver_id)
>
> def get_max_active_drivers_in_last_24_hours(current_time):
>     end_min = minute_of(current_time)
>     start_min = end_min - 1440
>     return max(len(buckets[m]) for m in range(start_min, end_min + 1))
> ```
>
> Query is **O(1440)** — independent of N. Add is bounded by delivery duration. Memory: O(active driver-minutes).
>
> ### Option 3: Event stream + windowed aggregation (production)
>
> Push delivery events to Kafka, stream-process via Flink or Spark Structured Streaming, materialize windowed aggregates. Overkill for in-memory but correct architecture at scale.
>
> ### My pick: Option 2
>
> Bucketed counts give O(window) per query, O(work-done) memory, and generalize to any window size by changing bucket granularity."

**Probe — "Memory if buckets stay forever?"**

> "Two answers:
> 1. **Eviction.** Buckets older than 24h are no longer queryable — background job (or lazy on query) deletes them. Memory bounded by O(window × avg_active_drivers).
> 2. **Persistence.** For longer history, persist to a time-series database (InfluxDB, TimescaleDB, or Postgres with bucket indexing). In-memory cache for hot data, TSDB for older queries."

**Probe — "What if the question changes to 'last 7 days' or 'last hour'?"**

> "Bucket granularity determines query resolution. For 'last hour' use minute buckets. For 'last 7 days' use hour buckets to keep bucket count manageable. Or maintain a hierarchy of bucket sizes — minute, hour, day — like a roll-up cube. That's the materialized-view pattern in OLAP."

### Closing Meta-Probes

**Probe — "How would you scale to millions of deliveries per day?"**

> "Three layers:
> 1. **Persistence** — Postgres source-of-truth with `end_time` index for `pay_up_to_time`. Running totals materialized as a separate table or in Redis with `INCRBYFLOAT`.
> 2. **Hot-path caching** — `get_total_cost` and `get_cost_to_be_paid` from Redis with sub-ms latency, atomic on each write.
> 3. **Analytics** — bucketed counts in Redis (sorted sets) for short windows, time-series database for longer. Stream processing if real-time aggregation needed.
>
> Core invariant — immutable deliveries + materialized aggregates — stays the same. Only the storage tier changes."

**Probe — "Biggest risk in this design?"**

> "Aggregate drift. If `total_cost` and the deliveries list disagree, all reports are silently wrong. Mitigations:
> 1. **Reconciliation job** — periodic check, alert on drift.
> 2. **Idempotent writes** — unique delivery ID, duplicates are no-ops on retries.
> 3. **Source-of-truth treatment** — deliveries list is canonical, aggregates can always be recomputed."

**Probe — "If you had more time, what would you add?"**

> "Three things:
> 1. **Per-driver aggregates** — same pattern indexed by driver_id. Enables driver-level reporting.
> 2. **Idempotency keys** — guard against duplicate deliveries on network retries.
> 3. **Versioning of hourly rates** — currently locked at delivery time, but for retroactive corrections we may want a `rate_history` per driver."

## Code

```python
"""
Delivery Cost Tracking System.
Standard library only.

Design principles:
  - Reads dominate writes -> shift computation to write time
  - Deliveries are immutable -> never mutate, only append
  - Payment tracked as a moving boundary, not by mutating records
  - Bucketed counts for time-windowed analytics
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from collections import defaultdict


class Driver:
    """Driver record. hourly_rate as Decimal for financial precision."""

    def __init__(self, driver_id, hourly_rate):
        self.id = driver_id
        self.hourly_rate = hourly_rate


class Delivery:
    """
    Immutable delivery record. Cost is precomputed at construction so:
      - Reads stay O(1)
      - Rate is locked in at delivery time (correct payroll behavior
        if rates change later)
    """

    def __init__(self, driver_id, start_time, end_time, cost):
        self.driver_id = driver_id
        self.start_time = start_time
        self.end_time = end_time
        self.cost = cost


class DeliverySystem:
    """
    Core design:
      - Immutable deliveries list (append-only)
      - Running total_cost (O(1) reads)
      - Payment as moving boundary: total_paid + last_paid_time
      - Bucketed minute-counts for 24h window analytics

    All hot-path operations are O(1).
    pay_up_to_time is O(N) worst case but called rarely (batch payroll).
    """

    BUCKET_SIZE_SECONDS = 60  # one-minute buckets
    WINDOW_BUCKETS = 1440     # 24h = 1440 minutes

    def __init__(self):
        self._drivers = {}
        # Immutable history -- never mutated, only appended.
        self._deliveries = []
        # Running aggregates -- updated on every write.
        self._total_cost = Decimal("0")
        self._total_paid = Decimal("0")
        # Payment boundary -- only moves forward.
        self._last_paid_time = None
        # Bucketed counts for active-driver analytics.
        # bucket_index -> set of driver_ids active during that minute.
        self._buckets = defaultdict(set)

    # -----------------------------------------------------------------
    # Phase 1: core CRUD with running totals
    # -----------------------------------------------------------------

    def add_driver(self, driver_id, usd_hourly_rate):
        # Decimal(str(float)) avoids float binary noise -- 0.1 stays 0.1.
        rate = Decimal(str(usd_hourly_rate))
        self._drivers[driver_id] = Driver(driver_id, rate)

    def add_delivery(self, driver_id, start_time, end_time):
        """
        Record a delivery. O(1) amortized for the cost/total updates,
        O(duration_minutes) for bucket population (bounded by max
        delivery length, typically constant).
        """
        driver = self._drivers[driver_id]
        duration_seconds = (end_time - start_time).total_seconds()
        hours = Decimal(str(duration_seconds)) / Decimal("3600")
        cost = driver.hourly_rate * hours

        delivery = Delivery(driver_id, start_time, end_time, cost)
        self._deliveries.append(delivery)
        self._total_cost += cost

        # Populate buckets so the 24h-window query is O(window).
        start_bucket = self._bucket_index(start_time)
        end_bucket = self._bucket_index(end_time)
        for b in range(start_bucket, end_bucket + 1):
            self._buckets[b].add(driver_id)

    def get_total_cost(self):
        """O(1) -- the hot path. Just return the running aggregate."""
        return self._total_cost

    # -----------------------------------------------------------------
    # Phase 2: payment tracking via moving boundary
    # -----------------------------------------------------------------

    def pay_up_to_time(self, timestamp):
        """
        Mark deliveries ending at or before timestamp as paid.

        O(N) worst case -- acceptable since batch payroll runs rarely.
        Each delivery is processed AT MOST ONCE across all pay_up_to_time
        calls (the boundary only moves forward), so amortized O(1) per
        delivery across the system's lifetime.

        Idempotent against earlier timestamps -- if t < last_paid_time,
        no-op (we never roll back the boundary).
        """
        if self._last_paid_time is not None and timestamp <= self._last_paid_time:
            return  # idempotent: don't roll back

        for d in self._deliveries:
            within_window = (
                (self._last_paid_time is None
                 or d.end_time > self._last_paid_time)
                and d.end_time <= timestamp
            )
            if within_window:
                self._total_paid += d.cost
        self._last_paid_time = timestamp

    def get_cost_to_be_paid(self):
        """O(1) -- another hot path."""
        return self._total_cost - self._total_paid

    # -----------------------------------------------------------------
    # Phase 3: time-windowed analytics
    # -----------------------------------------------------------------

    def get_max_active_drivers_in_last_24_hours(self, current_time):
        """
        Max distinct drivers active during any single minute in the
        last 24 hours.

        O(WINDOW_BUCKETS) = O(1440) -- bounded, INDEPENDENT of N.

        Bucket granularity (1 minute) controls resolution. For different
        window sizes, change WINDOW_BUCKETS or use a coarser bucket size.
        """
        end_bucket = self._bucket_index(current_time)
        start_bucket = end_bucket - self.WINDOW_BUCKETS + 1

        max_active = 0
        for b in range(start_bucket, end_bucket + 1):
            count = len(self._buckets.get(b, ()))
            if count > max_active:
                max_active = count
        return max_active

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _bucket_index(self, ts):
        """Convert a datetime to its bucket index (minutes since epoch)."""
        return int(ts.timestamp()) // self.BUCKET_SIZE_SECONDS


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

def demo():
    system = DeliverySystem()
    system.add_driver(1, 10.00)
    system.add_driver(2, 15.00)

    base = datetime(2026, 6, 7, 9, 0, tzinfo=timezone.utc)

    # Driver 1: 1.5h delivery -> $15.00
    system.add_delivery(1, base, base + timedelta(hours=1, minutes=30))
    # Driver 1: 2h delivery overlapping -> $20.00
    system.add_delivery(1, base + timedelta(minutes=30),
                           base + timedelta(hours=2, minutes=30))
    # Driver 2: 1h delivery overlapping both -> $15.00
    system.add_delivery(2, base + timedelta(hours=1),
                           base + timedelta(hours=2))

    print("=== Phase 1 ===")
    print("Total cost:", system.get_total_cost())              # 50.00
    print("Cost to be paid:", system.get_cost_to_be_paid())    # 50.00

    print("\n=== Phase 2 ===")
    # Pay everything ending at or before +2h: covers delivery 1 ($15)
    # and delivery 3 ($15). Delivery 2 (ends at +2.5h) remains unpaid.
    system.pay_up_to_time(base + timedelta(hours=2))
    print("After pay_up_to_time(+2h):")
    print("  Total cost:", system.get_total_cost())            # 50.00 (unchanged)
    print("  Cost to be paid:", system.get_cost_to_be_paid())  # 20.00

    # Idempotent: paying with an earlier time is a no-op.
    system.pay_up_to_time(base + timedelta(hours=1))
    print("  After idempotent earlier call, to be paid:",
          system.get_cost_to_be_paid())                        # 20.00

    print("\n=== Phase 3 ===")
    # During [+1h, +1.5h]: drivers 1 (both deliveries) and 2 -> 2 unique.
    # Per-minute peak in last 24h: 2 distinct drivers.
    current = base + timedelta(hours=3)
    max_active = system.get_max_active_drivers_in_last_24_hours(current)
    print("Max active drivers in last 24h:", max_active)       # 2


if __name__ == "__main__":
    demo()
```

## Final Complexity

| Operation | Complexity | Notes |
|---|---|---|
| `add_driver` | O(1) | dict insert |
| `add_delivery` | O(1) amortized | + O(duration_minutes) for bucket fill (bounded) |
| `get_total_cost` | **O(1)** | hot path — running total |
| `pay_up_to_time` | O(N) worst case | amortized O(1) per delivery; batch operation |
| `get_cost_to_be_paid` | **O(1)** | hot path — derived from running totals |
| `get_max_active_drivers_in_last_24_hours` | **O(1440)** | independent of N — bucket scan |

## Why This Beats the Heap Variant for This Interview

| Concern | Running totals | Heap-based |
|---|---|---|
| `add_delivery` | **O(1)** | O(log N) |
| `get_total_cost` | O(1) | O(1) |
| `get_cost_to_be_paid` | O(1) | O(1) |
| `pay_up_to_time` (batch) | O(N), rare | O(K log N), rare |
| History preservation | ✅ Full | ❌ Loses paid items |
| Code complexity | Single source of truth | Heap + tie-breaker + sentinel |
| Bug surface | Tiny | Larger |

For the interview as framed (batch payroll, audit history required, hot reads), **running totals wins on 5 of 6 dimensions**. The heap is the right answer for *real-time settlement*, where `pay_up_to_time` is frequent — that's the upgrade path to mention.

## Three Sentences to Memorize

These map directly to the three phases — drop them at the right moments:

1. **Phase 1:** *"Reads dominate writes for this dashboard, so I'll shift computation to write time — maintain a running total updated on each delivery, making `get_total_cost` O(1). Deliveries are immutable; aggregates are derived state."*

2. **Phase 2:** *"Payment is a separate concern from delivery. I'll never mutate a delivery record when payment happens — instead I track a moving boundary on the timeline and a separate `total_paid` aggregate. Historical delivery data stays intact for audit and dispute resolution."*

3. **Phase 3:** *"For windowed analytics over millions of records, I avoid full scans by maintaining intermediate state. Bucketed counts give O(window_size) per query independent of N — generalize to any window granularity by choosing the bucket size."*

## Likely Cross-Questions

| Question | Answer |
|---|---|
| "What if `pay_up_to_time` becomes hot?" | Switch to min-heap of unpaid deliveries keyed on `end_time`; trades O(log N) write for O(K log N) payment |
| "What if running total drifts?" | Reconciliation job; deliveries list is canonical, aggregates are cache |
| "Concurrent writers?" | Single lock on aggregates, or atomic `INCRBYFLOAT` in Redis at scale |
| "How to detect duplicate deliveries on retries?" | Idempotency key (delivery_id); duplicate `add_delivery` is no-op |
| "Per-driver totals?" | Same pattern keyed by driver_id; `Dict[driver_id → Decimal]` running total |
| "Cost in a custom time range?" | Need indexed-by-time storage; segment tree or DB query |
| "Refund / cancel a delivery?" | Append-only is incompatible with deletes; store cancellation as a separate negative-amount event |
| "Memory of bucket counts?" | Evict buckets older than max query window; or persist to TSDB |

## What This Interview Actually Tests

| Tested skill | Where you give it |
|---|---|
| Data modeling | Immutable Delivery, separation of payment from delivery |
| Scalability thinking | Running totals, bucketed analytics, eviction strategies |
| Complexity analysis | O(1) reads, O(window) analytics, defended optimality |
| Future extensibility | Phase 2 and 3 slot in cleanly because Phase 1 was designed for it |
| Production awareness | Reconciliation, idempotency, persistence — only when probed |

---

# 2. Corporate Rules Engine

## Problem

Build a rules engine for corporate credit cards. Five rule types initially, more coming, eventually rules created via API.

```
evaluateRules(rules: list, expenses: list) -> List[Violation]
```

Rules:
- Ban by expense_type / vendor_type / vendor_name
- Max single expense
- Max trip total
- Max sum per expense_type per trip
- Max sum per vendor_type per trip

## Speaking Script

**[Clarify]** "Structured violations with rule_id and affected expense_ids. Trip-level violations flag all contributing expenses. Multiple violations per expense reported. Money as Decimal. Design must be open to new rule types — prompt mentions future API.

**[Return type]** "List of `Violation` objects with rule_id, description, expense_ids, optional trip_id. Flat list, flexible grouping by caller.

**[Brute]** "Nested loop. For trip-level rules, sum the trip from scratch each check. O(N²·R). Plus the engine becomes a giant if/else — every new rule means editing the engine. Violates Open/Closed.

**[Optimal]** Two ideas:
1. **Group expenses by trip_id once** via `defaultdict`. Trip rules iterate each group exactly once.
2. **Strategy pattern** — abstract `Rule` base, with `ExpenseRule` and `TripRule` subclasses. Adding a rule type = new class, engine untouched.

**[Generalization]** "Two of the prompt's rules collapse into one parameterized class — `MaxAmountRule` with optional scope, and `FieldAggregationLimitRule` with field/value/max."

**[Complexity]** O(N · R) — optimal since every (expense, rule) pair must be checked.

## Code

```python
from decimal import Decimal
from collections import defaultdict


class Violation:
    def __init__(self, rule_id, rule_description, expense_ids, trip_id=None):
        self.rule_id = rule_id
        self.rule_description = rule_description
        self.expense_ids = expense_ids
        self.trip_id = trip_id

    def __repr__(self):
        return (f"Violation(rule={self.rule_id}, "
                f"expenses={self.expense_ids}, trip={self.trip_id})")


class Rule:
    def __init__(self, rule_id, description):
        self.rule_id = rule_id
        self.description = description


class ExpenseRule(Rule):
    """Evaluated against a single expense."""
    def check(self, expense):
        raise NotImplementedError


class TripRule(Rule):
    """Evaluated against all expenses in a trip.
    Returns list of offending expense_ids (empty = no violation)."""
    def check(self, trip_expenses):
        raise NotImplementedError


class BanRule(ExpenseRule):
    def __init__(self, rule_id, field, banned_value):
        super().__init__(rule_id, f"{field} cannot be '{banned_value}'")
        self.field = field
        self.banned_value = banned_value

    def check(self, expense):
        return expense.get(self.field) == self.banned_value


class MaxAmountRule(ExpenseRule):
    def __init__(self, rule_id, max_amount, scope_field=None, scope_value=None):
        scope = f" at {scope_field}={scope_value}" if scope_field else ""
        super().__init__(rule_id, f"Single expense{scope} cannot exceed ${max_amount}")
        self.max_amount = Decimal(str(max_amount))
        self.scope_field = scope_field
        self.scope_value = scope_value

    def check(self, expense):
        if self.scope_field is not None:
            if expense.get(self.scope_field) != self.scope_value:
                return False
        return Decimal(expense["amount_usd"]) > self.max_amount


class TripTotalLimitRule(TripRule):
    def __init__(self, rule_id, max_total):
        super().__init__(rule_id, f"Trip total cannot exceed ${max_total}")
        self.max_total = Decimal(str(max_total))

    def check(self, trip_expenses):
        total = sum((Decimal(e["amount_usd"]) for e in trip_expenses),
                    Decimal("0"))
        if total > self.max_total:
            return [e["expense_id"] for e in trip_expenses]
        return []


class FieldAggregationLimitRule(TripRule):
    """Generalizes 'meal total per trip', 'restaurant total per trip', etc."""
    def __init__(self, rule_id, field, value, max_total):
        super().__init__(rule_id,
                         f"Total {field}={value} per trip cannot exceed ${max_total}")
        self.field = field
        self.value = value
        self.max_total = Decimal(str(max_total))

    def check(self, trip_expenses):
        matching = [e for e in trip_expenses if e.get(self.field) == self.value]
        total = sum((Decimal(e["amount_usd"]) for e in matching),
                    Decimal("0"))
        if total > self.max_total:
            return [e["expense_id"] for e in matching]
        return []


class RulesEngine:
    def __init__(self, rules):
        # Partition once - O(R) at construction, not per evaluate.
        self.expense_rules = [r for r in rules if isinstance(r, ExpenseRule)]
        self.trip_rules = [r for r in rules if isinstance(r, TripRule)]

    def evaluate(self, expenses):
        violations = []

        # Per-expense rules
        for expense in expenses:
            for rule in self.expense_rules:
                if rule.check(expense):
                    violations.append(Violation(
                        rule.rule_id, rule.description,
                        [expense["expense_id"]], expense.get("trip_id")))

        # Group by trip once
        trips = defaultdict(list)
        for expense in expenses:
            trips[expense["trip_id"]].append(expense)

        # Per-trip rules
        for trip_id, trip_expenses in trips.items():
            for rule in self.trip_rules:
                offending = rule.check(trip_expenses)
                if offending:
                    violations.append(Violation(
                        rule.rule_id, rule.description, offending, trip_id))

        return violations


def evaluate_rules(rules, expenses):
    return RulesEngine(rules).evaluate(expenses)


def demo():
    rules = [
        MaxAmountRule("R1", 75, "vendor_type", "restaurant"),
        BanRule("R2", "expense_type", "airfare"),
        BanRule("R3", "expense_type", "entertainment"),
        MaxAmountRule("R4", 250),
        TripTotalLimitRule("R5", 2000),
        FieldAggregationLimitRule("R6", "expense_type", "meal", 200),
    ]
    expenses = [
        {"expense_id": "001", "trip_id": "T1", "amount_usd": "49.99",
         "expense_type": "client_hosting", "vendor_type": "restaurant"},
        {"expense_id": "002", "trip_id": "T1", "amount_usd": "120.00",
         "expense_type": "meal", "vendor_type": "restaurant"},  # R1
        {"expense_id": "003", "trip_id": "T1", "amount_usd": "300.00",
         "expense_type": "airfare", "vendor_type": "airline"},  # R2 + R4
        {"expense_id": "004", "trip_id": "T1", "amount_usd": "150.00",
         "expense_type": "meal", "vendor_type": "restaurant"},  # R6 (with #002)
    ]
    for v in evaluate_rules(rules, expenses):
        print(v)


if __name__ == "__main__":
    demo()
```

## Two Sentences to Memorize

1. *"Adding a new rule type means adding a new class — the engine itself doesn't change. That's the Open/Closed Principle."*
2. *"The complexity is O(N·R) — linear in expenses times rules — which is optimal because every expense-rule pair must be checked at least once in the worst case."*

---

# 3. Music Player (Spotify)

## Problem

```
int addSong(string title)                       // incremental ids from 1
void playSong(int songId, int userId)
void printMostPlayedSongs()                     // by unique-user count desc
```

Follow-ups:
- `getLastThreeSongs(userId)` — last 3 unique songs played
- `star(songId, userId)` / `unstar(songId, userId)`
- `getLastNFavouriteSongs(userId, n)` — last N starred songs played

## Speaking Script

**[Clarify]** "Same user replaying counts as 1 unique. Tie-break print by song_id ascending. `getLastThree` returns most-recent first. Replay bumps song to most recent in user history. For follow-up, 'starred now' semantics — unstarring removes from favourites view.

**[Types]** "Per-song `set[userId]` for O(1) uniqueness — `len(set)` *is* the play count. Per-user history as `OrderedDict` — supports O(1) move-to-end on replay and O(1) FIFO eviction (better than list once N is arbitrary).

**[Brute]** "Flat list of plays. Print recomputes from scratch — O(P). Should be incremental.

**[Optimal]**
1. Each Song owns a set of unique listener IDs. `playSong` = set.add → O(1).
2. For print, sort on call — O(N log N). LFU bucketing for O(1) updates and O(N) print is overkill unless print is hot.
3. Per-user history as bounded `OrderedDict`. Replay: move-to-end. Cap with `popitem(last=False)` for memory bound.

**[Complexity]** addSong/playSong O(1), printMostPlayed O(N log N), getLastThree O(1), star/unstar O(1), getLastNFavourite O(P) where P = bounded history.

## Code

```python
from collections import OrderedDict


class Song:
    def __init__(self, song_id, title):
        self.id = song_id
        self.title = title
        self.unique_listeners = set()  # set.add is O(1), dedup is automatic

    def play_count(self):
        return len(self.unique_listeners)


class UserProfile:
    def __init__(self):
        # OrderedDict: O(1) move_to_end, O(1) popitem(last=False) for FIFO
        self.play_history = OrderedDict()
        self.starred = set()

    def record_play(self, song_id, max_history):
        if song_id in self.play_history:
            self.play_history.move_to_end(song_id)
        else:
            self.play_history[song_id] = None
            if len(self.play_history) > max_history:
                self.play_history.popitem(last=False)

    def star(self, song_id):
        self.starred.add(song_id)

    def unstar(self, song_id):
        self.starred.discard(song_id)


class MusicPlayer:
    MAX_HISTORY = 1000  # bounded per-user history

    def __init__(self):
        self._songs = {}
        self._next_id = 1
        self._users = {}

    def _profile(self, user_id):
        if user_id not in self._users:
            self._users[user_id] = UserProfile()
        return self._users[user_id]

    def add_song(self, song_title):
        song_id = self._next_id
        self._next_id += 1
        self._songs[song_id] = Song(song_id, song_title)
        return song_id

    def play_song(self, song_id, user_id):
        song = self._songs[song_id]
        song.unique_listeners.add(user_id)
        self._profile(user_id).record_play(song_id, self.MAX_HISTORY)

    def print_most_played_songs(self):
        ranked = sorted(self._songs.values(),
                        key=lambda s: (-s.play_count(), s.id))
        for song in ranked:
            print(song.title)

    def get_last_three_songs(self, user_id):
        history = self._profile(user_id).play_history
        recent = list(history.keys())[-3:]
        return list(reversed(recent))

    def star(self, song_id, user_id):
        if song_id in self._songs:
            self._profile(user_id).star(song_id)

    def unstar(self, song_id, user_id):
        if user_id in self._users:
            self._users[user_id].unstar(song_id)

    def get_last_n_favourite_songs(self, user_id, n):
        if user_id not in self._users:
            return []
        profile = self._users[user_id]
        result = []
        # Walk newest -> oldest, filter by current starred set
        for song_id in reversed(profile.play_history):
            if song_id in profile.starred:
                result.append(song_id)
                if len(result) == n:
                    break
        return result


def demo():
    mp = MusicPlayer()
    s1 = mp.add_song("Bohemian Rhapsody")
    s2 = mp.add_song("Imagine")
    s3 = mp.add_song("Hotel California")
    s4 = mp.add_song("Stairway to Heaven")

    mp.play_song(s1, 100)        # user 100 replaying counts once
    mp.play_song(s1, 100)
    mp.play_song(s2, 100)
    mp.play_song(s2, 200)
    mp.play_song(s2, 300)        # 3 unique
    mp.play_song(s3, 100)
    mp.play_song(s3, 200)        # 2 unique
    mp.play_song(s4, 400)        # 1 unique

    print("Most played:")
    mp.print_most_played_songs()

    print("\nUser 100 last 3:", mp.get_last_three_songs(100))

    mp.star(s2, 100)
    mp.star(s3, 100)
    print("\nUser 100 last 2 favourites:",
          mp.get_last_n_favourite_songs(100, 2))


if __name__ == "__main__":
    demo()
```

## Two Sentences to Memorize

1. *"Each Song owns a set of unique listener IDs — `set.add` is O(1), dedup is automatic, and the size of the set is the play count."*
2. *"OrderedDict gives O(1) move-to-end, O(1) FIFO eviction, and O(1) `in` — it's the right structure when you need ordered, dedupable, indexable history."*

---

# 4. Employee Access Management

## Problem

```
grant_access(employee_id, resource_id, access_type)
revoke_access(employee_id, resource_id, access_type)   # access_type=None = revoke all
retrieve_access(employee_id, resource_id) -> List[AccessType]
retrieve_resources(employee_id) -> List[resource_id]
```

Three access types: READ, WRITE, ADMIN.

## Speaking Script

**[Clarify]** "Access types are independent, not hierarchical. Granting twice or revoking nothing is silent no-op. `retrieve_resources` returns any resource the employee has any access to.

**[Types]** "AccessType as Enum for type safety. Storage as nested dict — `Dict[emp_id → Dict[res_id → Set[AccessType]]]`. Sparse, all-O(1), set gives automatic dedup.

**[Brute]** "Flat list of (emp, res, type) tuples. Every method O(N). Doesn't scale.

**[Optimal]** "Nested dict. Outer key is employee_id since every method scopes by employee. Inner key is resource. Innermost is a Set of access types.

**[Complexity]** grant/revoke/retrieve_access O(1). retrieve_resources O(R) where R = resources for this employee — output-bounded floor.

**[Critical detail]** "When a revoke empties the set, delete the resource entry. When that empties the employee dict, delete the employee. Otherwise `retrieve_resources` reports resources with no access — correctness bug AND memory leak."

## Code

```python
from enum import Enum
from collections import defaultdict


class AccessType(Enum):
    READ = "READ"
    WRITE = "WRITE"
    ADMIN = "ADMIN"


class AccessManager:
    """
    Storage: access_map[employee_id][resource_id] -> set[AccessType]

    All four operations O(1) or output-bounded. Cleanup of empty containers
    keeps retrieve_resources correct AND prevents memory leaks.
    """

    def __init__(self):
        self._access_map = defaultdict(lambda: defaultdict(set))

    def grant_access(self, employee_id, resource_id, access_type):
        self._access_map[employee_id][resource_id].add(access_type)

    def revoke_access(self, employee_id, resource_id, access_type=None):
        if employee_id not in self._access_map:
            return
        emp_resources = self._access_map[employee_id]
        if resource_id not in emp_resources:
            self._cleanup_employee(employee_id)
            return

        if access_type is None:
            del emp_resources[resource_id]
        else:
            emp_resources[resource_id].discard(access_type)
            if not emp_resources[resource_id]:
                del emp_resources[resource_id]

        self._cleanup_employee(employee_id)

    def retrieve_access(self, employee_id, resource_id):
        if employee_id not in self._access_map:
            return []
        if resource_id not in self._access_map[employee_id]:
            return []
        access_set = self._access_map[employee_id][resource_id]
        return sorted(access_set, key=lambda a: list(AccessType).index(a))

    def retrieve_resources(self, employee_id):
        if employee_id not in self._access_map:
            return []
        return list(self._access_map[employee_id].keys())

    def _cleanup_employee(self, employee_id):
        if (employee_id in self._access_map
                and not self._access_map[employee_id]):
            del self._access_map[employee_id]


def demo():
    am = AccessManager()
    am.grant_access("E1", "R1", AccessType.READ)
    am.grant_access("E1", "R1", AccessType.WRITE)
    am.grant_access("E1", "R2", AccessType.READ)
    am.grant_access("E2", "R1", AccessType.ADMIN)
    am.grant_access("E1", "R1", AccessType.READ)  # idempotent

    print("E1 access to R1:", am.retrieve_access("E1", "R1"))  # [READ, WRITE]
    print("E1 resources:", am.retrieve_resources("E1"))         # [R1, R2]

    am.revoke_access("E1", "R1", AccessType.READ)
    print("After revoke READ from E1/R1:", am.retrieve_access("E1", "R1"))  # [WRITE]

    am.revoke_access("E1", "R1", None)
    print("After revoke all from E1/R1, E1 resources:",
          am.retrieve_resources("E1"))                          # [R2]

    am.revoke_access("E1", "R2", None)
    print("After all revoked, E1 resources:",
          am.retrieve_resources("E1"))                          # []

    am.revoke_access("E99", "R99", AccessType.READ)             # silent no-op


if __name__ == "__main__":
    demo()
```

## Two Sentences to Memorize

1. *"All four methods are O(1) or output-bounded — `retrieve_resources` is O(R), which is optimal because we must enumerate the result. The structure is at the theoretical floor."*
2. *"Critical detail: when a revoke empties the set, I delete the resource entry; when that empties the employee dict, I delete the employee. Otherwise `retrieve_resources` would report resources with empty access — correctness bug, plus memory leak."*

---

# 5. Excel Spreadsheet

## Problem

```
void set(string cell, string value)   // "10", "-5", "=2+8", "=-1+-10+2"
void reset(string cell)               // also: set(cell, "")
void print()                          // raw + computed values
```

Follow-up: cell references like `=A1+10`, with cycle detection.

## Speaking Script

**[Clarify]** "Formulas can have arbitrary count of `+`-joined integers, including negatives like `=-1+-10+2`. Empty string resets. For follow-up references, cycles → ERROR (don't crash). Unset references treated as 0 — Excel convention.

**[Types]** "Sparse `Dict[cell_id → Cell]`. Cell holds raw string and parsed CellValue. Strategy pattern: `CellValue` base with `LiteralValue`, `FormulaValue`. Adding subtraction or multiplication later = new subclass.

**[Brute]** "Re-parse and re-evaluate every cell on every print. With references, recursive eval infinite-loops on cycles. Better: parse once at set, evaluate lazily with memoization and cycle detection.

**[Optimal]** "Parse on `set`. On `print`, evaluate each cell once with a memo cache. For references, recursive evaluation passes a `evaluating` set down — if we re-encounter a cell mid-evaluation, that's a cycle.

**[Parsing without regex]** "**Key insight: `+` is the ONLY separator; `-` is always a sign.** So `split('+')` is unambiguous. Each token is then either a signed integer or a cell reference.

**[Complexity]** set O(L), reset O(1), print O(C + E) where E = total reference edges. Optimal.

## Code

```python
"""Excel spreadsheet with cell references and cycle detection. No regex."""


class CellValue:
    def evaluate(self, spreadsheet, evaluating):
        raise NotImplementedError


class LiteralValue(CellValue):
    def __init__(self, value):
        self.value = value

    def evaluate(self, spreadsheet, evaluating):
        return self.value


class FormulaValue(CellValue):
    """Operands are list of (kind, value) tuples.
    kind == 'int' -> value is int; kind == 'ref' -> value is cell_id."""
    def __init__(self, operands):
        self.operands = operands

    def evaluate(self, spreadsheet, evaluating):
        total = 0
        for kind, v in self.operands:
            if kind == "int":
                total += v
            else:
                total += spreadsheet._resolve(v, evaluating)
        return total


class Cell:
    def __init__(self, raw, value):
        self.raw = raw
        self.value = value


# ----- hand-rolled parser (no regex) -----

def _is_digit(ch):
    return '0' <= ch <= '9'


def _is_letter(ch):
    return 'A' <= ch <= 'Z'


def _is_signed_integer(token):
    if not token:
        return False
    start = 1 if token[0] == '-' else 0
    if start == len(token):
        return False
    for i in range(start, len(token)):
        if not _is_digit(token[i]):
            return False
    return True


def _is_cell_ref(token):
    if not token:
        return False
    i = 0
    while i < len(token) and _is_letter(token[i]):
        i += 1
    if i == 0:
        return False
    letter_end = i
    while i < len(token) and _is_digit(token[i]):
        i += 1
    return i > letter_end and i == len(token)


def _parse_operand(token):
    if _is_signed_integer(token):
        return ("int", int(token))
    if _is_cell_ref(token):
        return ("ref", token)
    raise ValueError(f"Invalid operand: {token!r}")


def _split_formula(body):
    """Split on '+' separators. '-' is always a sign, never a separator,
    so split('+') is unambiguous: '-1+-10+2' -> ['-1', '-10', '2']."""
    tokens = []
    current = []
    for ch in body:
        if ch == '+':
            tokens.append(''.join(current))
            current = []
        else:
            current.append(ch)
    tokens.append(''.join(current))
    return tokens


def _parse_value(raw):
    if raw.startswith('='):
        body = raw[1:]
        tokens = _split_formula(body)
        operands = [_parse_operand(t) for t in tokens]
        return FormulaValue(operands)
    if not _is_signed_integer(raw):
        raise ValueError(f"Invalid literal: {raw!r}")
    return LiteralValue(int(raw))


# ----- spreadsheet with cycle detection -----

class CycleError(Exception):
    pass


class Spreadsheet:
    def __init__(self):
        self._cells = {}

    def set(self, cell_id, raw_value):
        if raw_value == "":
            self.reset(cell_id)
            return
        value = _parse_value(raw_value)
        self._cells[cell_id] = Cell(raw_value, value)

    def reset(self, cell_id):
        self._cells.pop(cell_id, None)

    def print(self):
        cache = {}
        for cell_id in sorted(self._cells.keys()):
            cell = self._cells[cell_id]
            computed = self._evaluate_with_cache(cell_id, cache)
            print(f"{cell_id}: raw='{cell.raw}', computed={computed}")

    def _evaluate_with_cache(self, cell_id, cache):
        if cell_id in cache:
            return cache[cell_id]
        try:
            result = self._resolve(cell_id, evaluating=set())
        except CycleError:
            result = "ERROR(cycle)"
        cache[cell_id] = result
        return result

    def _resolve(self, cell_id, evaluating):
        if cell_id in evaluating:
            raise CycleError()
        if cell_id not in self._cells:
            return 0  # Excel convention: unset cell = 0

        evaluating.add(cell_id)
        try:
            return self._cells[cell_id].value.evaluate(self, evaluating)
        finally:
            evaluating.discard(cell_id)


def demo():
    sheet = Spreadsheet()
    sheet.set("A1", "10")
    sheet.set("A2", "-5")
    sheet.set("B1", "=2+8")
    sheet.set("B2", "=-1+-10+2")
    sheet.set("D1", "=A1+10")
    sheet.set("D2", "=D1+B1+5")

    print("--- Initial ---")
    sheet.print()

    print("\n--- With cycle ---")
    sheet.set("E1", "=E2+1")
    sheet.set("E2", "=E1+1")
    sheet.print()


if __name__ == "__main__":
    demo()
```

## Two Sentences to Memorize

1. *"I parse on `set` for fail-fast feedback and evaluate lazily on `print`; cycle detection uses an `evaluating` set passed through recursion — re-encountering a cell mid-evaluation means a cycle, so we mark as ERROR rather than crash."*
2. *"The parsing trick: `+` is the only separator and `-` is always a sign, so `split('+')` is unambiguous — `=-1+-10+2` splits cleanly into `['-1', '-10', '2']` without a tokenizer."*

---

# 6. Key-Value Store with Transactions

## Problem

```
string get(string key)
void set(string key, string value)
void deleteKey(string key)
```

Follow-ups: `begin`, `commit`, `rollback`. Then nested transactions.

## Speaking Script

**[Clarify]** "Get of missing returns empty string. Delete of missing is silent no-op. Get inside transaction sees pending writes (read-your-writes). Commit/rollback with no active transaction is an error.

**[Critical decision]** "I'll design for nested transactions from the start, even though the interviewer asked for one level first. Modeling as a stack-of-dicts means single-level is just a stack of size one — extending to nested is a non-change.

**[Types]** "Two structures: committed dict, plus a stack of dicts where each layer maps key → (value | DELETED sentinel). Sentinel distinguishes 'deleted in this layer' from 'untouched'.

**[Brute]** "Naive uses `in_transaction` flag plus separate writes/deletes maps. Works for one level, doesn't extend. Worse: separate structures can desync.

**[Optimal]** Operations:
- `get` walks stack top-down → committed; sentinel returns empty without falling through.
- `set` / `delete` write to top layer if active; else to committed.
- `begin` pushes empty dict.
- `commit` pops top; merges into next layer (or committed). DELETED sentinels become real deletes only at committed level — propagate up otherwise (so outer rollback can still restore).
- `rollback` pops top.

**[Why nesting is free]** "By using a stack from the start, going from 1 to N transaction levels is zero code change.

**[Complexity]** get O(D) where D = depth, set/delete/begin/rollback O(1), commit O(K) for keys touched.

## Edge Cases Handled by Design

The sentinel-in-layer approach handles every common delete bug:

| Case | Why it works |
|---|---|
| Delete key in committed only | sentinel written to top layer |
| Get returns stale after delete | get checks sentinel before fall-through |
| Delete then set same key | set overwrites sentinel — single assignment |
| Set then delete same key | delete overwrites value — single assignment |
| Commit applies deletes | explicit branch on sentinel |
| Delete non-existent key | `pop(key, None)` is silent |
| Rollback restores deletes | layer discarded; committed untouched |
| Nested commit + outer rollback | sentinels propagate up, applied only at committed |

## Code

```python
"""Key-value store with nested transactions via stack of dicts + sentinel."""


# Private singleton — users can't reach it, so it can't collide with values.
_DELETED = object()


class KeyValueStore:
    def __init__(self):
        self._committed = {}
        self._tx_stack = []  # list of dicts; top = innermost active tx

    def get(self, key):
        # Walk top-down: most recent layer wins.
        for layer in reversed(self._tx_stack):
            if key in layer:
                value = layer[key]
                if value is _DELETED:
                    return ""
                return value
        return self._committed.get(key, "")

    def set(self, key, value):
        if self._tx_stack:
            self._tx_stack[-1][key] = value
        else:
            self._committed[key] = value

    def delete_key(self, key):
        if self._tx_stack:
            # Sentinel - get() will see this before falling through.
            self._tx_stack[-1][key] = _DELETED
        else:
            self._committed.pop(key, None)

    def begin(self):
        self._tx_stack.append({})

    def commit(self):
        if not self._tx_stack:
            raise RuntimeError("commit with no active transaction")

        top = self._tx_stack.pop()
        target = self._tx_stack[-1] if self._tx_stack else self._committed

        for key, value in top.items():
            if value is _DELETED:
                if target is self._committed:
                    # Real delete only at committed level.
                    self._committed.pop(key, None)
                else:
                    # Propagate sentinel up so outer rollback can restore.
                    target[key] = _DELETED
            else:
                target[key] = value

    def rollback(self):
        if not self._tx_stack:
            raise RuntimeError("rollback with no active transaction")
        self._tx_stack.pop()


def demo():
    kv = KeyValueStore()

    # Basic
    kv.set("a", "1")
    print("get('a'):", kv.get("a"))                  # 1
    kv.delete_key("a")
    print("after delete, get('a'):", kv.get("a"))    # ''

    # Single transaction
    kv.set("x", "10")
    kv.begin()
    kv.set("x", "20")                                # tx-local write
    print("inside tx, get('x'):", kv.get("x"))       # 20 (read-your-writes)
    kv.commit()
    print("after commit, get('x'):", kv.get("x"))    # 20

    # Rollback restores
    kv.begin()
    kv.set("x", "999")
    kv.delete_key("x")
    kv.rollback()
    print("after rollback, get('x'):", kv.get("x"))  # 20

    # Nested transactions (free, thanks to stack design)
    kv.set("p", "100")
    kv.begin()                                       # outer
    kv.set("p", "200")
    kv.begin()                                       # inner
    kv.set("p", "300")
    kv.rollback()                                    # discard inner
    print("after inner rollback, get('p'):", kv.get("p"))  # 200
    kv.commit()                                      # commit outer
    print("after outer commit, get('p'):", kv.get("p"))    # 200

    # Nested deletion + outer rollback (sentinel propagation)
    kv.set("q", "alive")
    kv.begin()
    kv.delete_key("q")
    print("inside tx, get('q'):", kv.get("q"))       # ''
    kv.rollback()
    print("after rollback, get('q'):", kv.get("q"))  # 'alive'  (sentinel discarded)


if __name__ == "__main__":
    demo()
```

## Two Sentences to Memorize

1. *"I model transactions as a stack of layered dicts from the start — going from one transaction to nested is a non-change. Single-level is just a stack of size one."*
2. *"I use a `_DELETED` sentinel rather than a separate deletes set so each layer has one source of truth. The sentinel propagates up through nested commits and only becomes a real deletion when it reaches the committed store — that's what preserves rollback semantics across nesting."*

---

# Cross-Problem Comparison

| Problem | Hot Path | Key DS | Brute | Optimal | Pattern |
|---|---|---|---|---|---|
| Food Delivery | Reads | Heap + totals | O(N) reads | O(1) reads | Comparison floor |
| Rules Engine | Bulk eval | defaultdict + Strategy | O(N²·R) | O(N·R) | Open/Closed |
| Music Player | Per-play | Set + OrderedDict | O(P) print | O(1) play | Bounded LRU |
| Access Mgmt | Per-method | Nested dict + Set | O(N) all | O(1) all | Output-bounded |
| Excel | Print | Dict + DAG | Re-parse always | O(C+E) print | Strategy + DFS |
| KV Store | get/set | Stack + sentinel | flag + 2 maps | O(1)/O(D) | Layered overlays |

# What These Problems Test (DSA-LLD Signal)

| Signal | Where it's tested |
|---|---|
| Right data structure choice | All 6 problems |
| Brute → optimal complexity | All 6 |
| OOP / design patterns | Rules Engine, Excel, Music |
| Edge case awareness | KV Store, Access Mgmt |
| Forward design | KV Store (nesting), Driver (Q2 fits cleanly) |
| Algorithmic depth | Excel (cycle detection), Driver (heap) |
| Restraint / no over-engineering | Access Mgmt (no Employee class), Music (size-3 list) |

# Common Pitfalls Across Problems

| Pitfall | Where it bites | Fix |
|---|---|---|
| Float for money | Driver, Rules Engine | `Decimal` always |
| Naive datetime | Driver | UTC `datetime` with `tzinfo` |
| Heap without tie-breaker | Driver | Monotonic counter in tuple |
| Two parallel structures | KV Store (writes/deletes) | One structure with sentinel |
| Modeling tx as Optional | KV Store | Stack from the start |
| Forgetting empty cleanup | Access Mgmt | Cascade-delete empty parents |
| `Decimal(float)` | Driver, Rules Engine | `Decimal(str(float))` |
| split on `+` for `=-1+-10+2` | Excel | Works! `-` is always a sign |
| Cycle = infinite loop | Excel | DFS with `evaluating` set |

# Bottom Line

Hit the template every time:

1. Clarify (45-60s)
2. Data types (20s)
3. Brute (45-60s)
4. Optimal (90s)
5. OOP (20-30s)
6. Trade-offs (20s)
7. Code (15 min)

3-4 minutes of preamble before code. Don't skip the brute → optimal step — that's the SDE2 signal.

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

1. [Food Delivery Payments](#1-food-delivery-payments) — heap + running totals
2. [Corporate Rules Engine](#2-corporate-rules-engine) — Strategy / Open-Closed
3. [Music Player (Spotify)](#3-music-player-spotify) — Set + OrderedDict
4. [Employee Access Management](#4-employee-access-management) — nested dict + Set
5. [Excel Spreadsheet](#5-excel-spreadsheet) — Strategy + DFS cycle detection
6. [Key-Value Store with Transactions](#6-key-value-store-with-transactions) — stack of dicts + sentinel

---

# 1. Food Delivery Payments

## Problem

Food delivery system with thousands of drivers, each paid an hourly rate based on delivery duration. Drivers can have overlapping deliveries.

```
addDriver(driverId, usdHourlyRate)
recordDelivery(driverId, startTime, endTime)
getTotalCost()                  // live dashboard - hot path
payUpTo(payTime)                // mark deliveries paid up to a given time
getTotalCostUnpaid()            // live dashboard
```

## Speaking Script

**[Clarify]** "Dashboard polls reads constantly — reads are the hot path. Inputs valid per prompt. Money in USD. Single-threaded.

**[Types]** "Time as UTC datetime — timezone-safe, microsecond precision. Money as `Decimal` — float drift compounds. Drivers in a `dict` for O(1) lookup.

**[Brute]** "Flat list of deliveries, sum on every `getTotalCost`. O(N) per read — bad for a polled dashboard.

**[Optimal]** Two ideas:
1. **Running totals** updated on each write → reads become O(1).
2. **Min-heap on `end_time`** for unpaid deliveries → `payUpTo` peeks the smallest end_time and pops while in window. Each delivery paid once across lifetime → amortized cheap.

**[Complexity]** addDriver O(1), recordDelivery O(log N), getTotalCost O(1), getTotalCostUnpaid O(1), payUpTo O(K log N) amortized.

**[Subtle]** Heap needs a tie-breaker integer between `end_time` and `Delivery` to prevent `Delivery < Delivery` crashes. Use `Decimal(str(float))` not `Decimal(float)` to avoid binary noise.

## Code

```python
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from heapq import heappush, heappop


class Driver:
    def __init__(self, driver_id, hourly_rate):
        self.id = driver_id
        self.hourly_rate = hourly_rate


class Delivery:
    def __init__(self, driver_id, start_time, end_time, cost):
        self.driver_id = driver_id
        self.start_time = start_time
        self.end_time = end_time
        self.cost = cost


class DeliverySystem:
    def __init__(self):
        self._drivers = {}
        self._total_cost = Decimal("0")
        self._total_unpaid = Decimal("0")
        # Heap entries: (end_time, tie_breaker, Delivery).
        # Tie-breaker prevents heapq from comparing Delivery objects on
        # equal end_times (which would crash).
        self._unpaid = []
        self._tie_breaker = 0

    def add_driver(self, driver_id, usd_hourly_rate):
        rate = Decimal(str(usd_hourly_rate))
        self._drivers[driver_id] = Driver(driver_id, rate)

    def record_delivery(self, driver_id, start_time, end_time):
        driver = self._drivers[driver_id]
        duration_seconds = (end_time - start_time).total_seconds()
        hours = Decimal(str(duration_seconds)) / Decimal("3600")
        cost = driver.hourly_rate * hours

        delivery = Delivery(driver_id, start_time, end_time, cost)
        self._total_cost += cost
        self._total_unpaid += cost

        heappush(self._unpaid, (end_time, self._tie_breaker, delivery))
        self._tie_breaker += 1

    def get_total_cost(self):
        return self._total_cost

    def pay_up_to(self, pay_time):
        while self._unpaid and self._unpaid[0][0] <= pay_time:
            _, _, delivery = heappop(self._unpaid)
            self._total_unpaid -= delivery.cost

    def get_total_cost_unpaid(self):
        return self._total_unpaid


def demo():
    system = DeliverySystem()
    system.add_driver(1, 10.00)

    base = datetime(2026, 5, 30, 9, 0, tzinfo=timezone.utc)
    system.record_delivery(1, base, base + timedelta(hours=1, minutes=30))  # $15
    system.record_delivery(1, base + timedelta(minutes=30),
                              base + timedelta(hours=2, minutes=30))         # $20

    print("Total cost:", system.get_total_cost())              # 35.00
    print("Unpaid:", system.get_total_cost_unpaid())           # 35.00

    system.pay_up_to(base + timedelta(hours=2))                # pays first
    print("After payUpTo(+2h) unpaid:", system.get_total_cost_unpaid())  # 20.00


if __name__ == "__main__":
    demo()
```

## Two Sentences to Memorize

1. *"The dashboard is the hot path — I'll optimize reads with running totals so `getTotalCost` is O(1), and use a min-heap keyed on end_time for `payUpTo` so each delivery is paid exactly once across the system's lifetime."*
2. *"Three of five operations are O(1) — at the theoretical floor — and the other two match the comparison-based lower bound for ordered insertion and minimum extraction, so this design is provably optimal."*

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

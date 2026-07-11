#!/usr/bin/env python3
"""Generate the router-training dataset: ~360 tasks across all 8 categories,
every one with deterministic ground truth. Dev-only: never shipped in the image.

Writes train_data/tasks.json and train_data/expected.json. Grading kinds:
  number        tolerant match on the last number in the answer
  label         sentiment label (positive/negative/neutral/mixed)
  contains      expected substring, case-insensitive
  contains_any  any of the listed substrings
  contains_all  all of the listed substrings
  entities      "Name:type" list, graded by >=75% name recall
  code_tests    executable specs run in the app/verify.py sandbox
  summary       constraints parsed from the prompt + >=1 keyterm present

Logic puzzles are generated constructively and then brute-force verified with
itertools.permutations: a puzzle is only emitted when exactly one assignment
satisfies its constraints (the approach the fine-tune-llm-query-router repo
uses for its adversarial set).
"""
import itertools
import json
import os
import random

random.seed(20260711)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "train_data")
PER_CATEGORY = 45

TASKS = []
EXPECTED = {}
_counter = {}


def add(category, prompt, expected, kind):
    n = _counter.get(category, 0) + 1
    _counter[category] = n
    tid = f"{category}_{n:03d}"
    TASKS.append({"task_id": tid, "prompt": prompt})
    EXPECTED[tid] = {"answer": expected, "kind": kind}


# ------------------------------------------------------------------- math (45)

def gen_math():
    specs = []
    for _ in range(8):
        total = random.choice([240, 320, 360, 400, 480, 500, 600, 720, 840])
        pct = random.choice([10, 15, 20, 25, 30, 40])
        extra = random.choice([25, 30, 45, 60, 75, 80])
        specs.append((
            f"A warehouse holds {total} boxes. {pct}% are shipped on Monday and "
            f"{extra} more on Tuesday. How many boxes remain?",
            total - total * pct // 100 - extra))
    for _ in range(6):
        speed = random.choice([45, 54, 60, 72, 80, 90])
        hours = random.choice([1.5, 2, 2.5, 3, 3.5, 4])
        specs.append((
            f"A cyclist rides at {speed} km per hour for {hours} hours. "
            f"How far does the cyclist travel in km?", round(speed * hours, 2)))
    for _ in range(6):
        price = random.choice([60, 80, 90, 120, 150, 200])
        disc = random.choice([10, 15, 20, 25])
        coupon = random.choice([5, 8, 10])
        specs.append((
            f"A coat costs ${price}. It is discounted by {disc}%, then a "
            f"${coupon} coupon is applied. What is the final price in dollars?",
            round(price * (100 - disc) / 100 - coupon, 2)))
    for _ in range(5):
        boxes = random.choice([3, 4, 5, 6])
        per = random.choice([12, 18, 24, 30, 36])
        away = random.choice([7, 11, 15, 19, 23])
        specs.append((
            f"Nina has {boxes} bags with {per} marbles each. She gives away "
            f"{away} marbles. How many marbles does she have left?",
            boxes * per - away))
    for _ in range(5):
        grams = random.choice([200, 250, 300, 400])
        serves = random.choice([8, 10, 12])
        target = serves * random.choice([2, 3, 4]) + random.choice([0, serves // 2])
        specs.append((
            f"A recipe needs {grams} g of rice for {serves} servings. "
            f"How many grams are needed for {target} servings?",
            round(grams * target / serves, 2)))
    for _ in range(5):
        start = random.choice([360, 480, 500, 600])
        r1, t1 = random.choice([(8, 15), (6, 20), (10, 12)])
        r2, t2 = random.choice([(12, 20), (9, 10), (15, 8)])
        r3, t3 = random.choice([(5, 10), (4, 15), (7, 6)])
        specs.append((
            f"A tank starts with {start} liters. It drains at {r1} liters per minute "
            f"for {t1} minutes, is refilled at {r2} liters per minute for {t2} minutes, "
            f"then drains at {r3} liters per minute for {t3} minutes. "
            f"How many liters are in the tank now?",
            start - r1 * t1 + r2 * t2 - r3 * t3))
    for _ in range(5):
        principal = random.choice([1000, 2000, 5000])
        g1 = random.choice([5, 8, 10])
        g2 = random.choice([-4, -3, 6])
        amount = principal * (100 + g1) / 100 * (100 + g2) / 100
        specs.append((
            f"You invest ${principal}. In year 1 it grows by {g1}%, in year 2 it "
            f"{'shrinks' if g2 < 0 else 'grows'} by {abs(g2)}%. What is the final "
            f"amount in dollars, rounded to 2 decimal places?",
            round(amount, 2)))
    for _ in range(5):
        nums = sorted(random.sample(range(10, 99), random.choice([4, 5])))
        specs.append((
            f"What is the average of {', '.join(map(str, nums))}? "
            f"Give the answer as a decimal.",
            round(sum(nums) / len(nums), 4)))
    random.shuffle(specs)
    for q, ans in specs[:PER_CATEGORY]:
        add("math", q, ans, "number")


# ------------------------------------------------------------------ logic (45)

NAMES = ["Ana", "Ben", "Cara", "Dan", "Erin", "Finn", "Gia", "Hugo", "Ivy",
         "Jack", "Kim", "Leo", "Mia", "Noah", "Omar", "Pia", "Quinn", "Rosa",
         "Sam", "Tara", "Uma", "Vic", "Wes", "Zoe"]
ITEM_SETS = [
    ("pet", ["cat", "dog", "bird", "fish", "hamster", "rabbit"]),
    ("drink", ["coffee", "tea", "juice", "milk", "soda", "water"]),
    ("car", ["sedan", "truck", "coupe", "van", "hatchback", "wagon"]),
    ("fruit", ["apple", "banana", "cherry", "mango", "grape", "pear"]),
    ("sport", ["soccer", "tennis", "chess", "golf", "rugby", "hockey"]),
    ("instrument", ["piano", "violin", "drums", "flute", "guitar", "cello"]),
]
VERBS = {"pet": ("owns", "own"), "drink": ("drinks", "drink"),
         "car": ("owns", "own"), "fruit": ("likes", "like"),
         "sport": ("plays", "play"), "instrument": ("plays", "play")}


def _solve_bruteforce(names, items, positives, negatives):
    """The itertools uniqueness check: return all satisfying assignments."""
    sols = []
    for perm in itertools.permutations(items):
        assign = dict(zip(names, perm))
        if all(assign[a] == b for a, b in positives) and \
           all(assign[a] != b for a, b in negatives):
            sols.append(assign)
    return sols


def gen_assignment_puzzle(n):
    """Constructive generation, then brute-force verification of uniqueness."""
    for _ in range(200):
        noun, pool = random.choice(ITEM_SETS)
        items = random.sample(pool, n)
        names = random.sample(NAMES, n)
        truth = dict(zip(names, random.sample(items, n)))
        # Constraints: some positives about non-target people, negatives elsewhere.
        target = random.choice(names)
        positives, negatives = [], []
        others = [x for x in names if x != target]
        for p in random.sample(others, random.choice([1, max(1, n - 2)])):
            positives.append((p, truth[p]))
        for p in names:
            if (p, truth[p]) in positives:
                continue
            wrong = [it for it in items if it != truth[p]]
            if random.random() < 0.8:
                negatives.append((p, random.choice(wrong)))
        sols = _solve_bruteforce(names, items, positives, negatives)
        if len(sols) != 1 or sols[0] != truth:
            continue
        v3, v_inf = VERBS[noun]
        count_word = {3: "Three", 4: "Four", 5: "Five"}[n]
        sents = [f"{a} {v3} the {b}." for a, b in positives]
        sents += [f"{a} does not {v_inf} the {b}." for a, b in negatives]
        random.shuffle(sents)
        prompt = (f"{count_word} friends, {', '.join(names[:-1])}, and {names[-1]}, "
                  f"each {v_inf} a different {noun}: {', '.join(items)}. "
                  + " ".join(sents)
                  + f" Who {v3} the {truth[target]}?")
        return prompt, target
    raise RuntimeError("could not generate a unique puzzle")


def gen_logic():
    for _ in range(15):
        prompt, ans = gen_assignment_puzzle(3)
        add("logic", prompt, ans, "contains")
    for _ in range(10):
        prompt, ans = gen_assignment_puzzle(4)
        add("logic", prompt, ans, "contains")
    # Transitive ordering: build a random true order, state adjacent comparisons.
    for _ in range(10):
        names = random.sample(NAMES, random.choice([3, 4]))
        order = list(names)
        random.shuffle(order)  # order[0] is tallest
        rel = random.choice([("taller than", "shortest", "tallest"),
                             ("older than", "youngest", "oldest"),
                             ("faster than", "slowest", "fastest")])
        facts = [f"{order[i]} is {rel[0]} {order[i + 1]}."
                 for i in range(len(order) - 1)]
        random.shuffle(facts)
        if random.random() < 0.5:
            q, ans = f"Who is the {rel[1]}?", order[-1]
        else:
            q, ans = f"Who is the {rel[2]}?", order[0]
        add("logic", " ".join(facts) + f" {q}", ans, "contains")
    # Heads and legs.
    for _ in range(5):
        rabbits = random.randint(2, 9)
        chickens = random.randint(3, 12)
        heads, legs = rabbits + chickens, 4 * rabbits + 2 * chickens
        add("logic",
            f"A farmer has chickens and rabbits, {heads} heads and {legs} legs "
            f"in total. How many rabbits are there?", str(rabbits), "contains")
    # Fixed-answer deduction questions.
    fixed = [
        ("If all roses are flowers and some flowers fade quickly, can we conclude "
         "that some roses fade quickly? Answer yes or no and explain briefly.", "no"),
        ("If every programmer drinks coffee and Lee drinks coffee, does it follow "
         "that Lee is a programmer? Answer yes or no and explain briefly.", "no"),
        ("Dan always lies and Erin always tells the truth. Dan says 'Erin said it "
         "is raining.' Is it necessarily true that Erin said that? Answer yes or "
         "no and explain.", "no"),
        ("If no reptiles are mammals and all snakes are reptiles, can we conclude "
         "that no snakes are mammals? Answer yes or no and explain briefly.", "yes"),
        ("All squares are rectangles. All rectangles have four sides. Can we "
         "conclude every square has four sides? Answer yes or no and explain "
         "briefly.", "yes"),
    ]
    for q, a in fixed:
        add("logic", q, a, "contains")


# ---------------------------------------------------------------- codegen (45)

CODEGEN_SPECS = [
    ("returns the second-largest number in a list, handling duplicates correctly",
     [{"args": [[1, 2, 2, 3]], "expected": 2}, {"args": [[5, 1, 4]], "expected": 4}]),
    ("checks whether a string is a palindrome, ignoring case",
     [{"args": ["Racecar"], "expected": True}, {"args": ["hello"], "expected": False}]),
    ("computes the factorial of a non-negative integer",
     [{"args": [5], "expected": 120}, {"args": [0], "expected": 1}]),
    ("returns the sum of all even numbers in a list",
     [{"args": [[1, 2, 3, 4]], "expected": 6}]),
    ("counts the vowels in a lowercase string",
     [{"args": ["hello world"], "expected": 3}]),
    ("reverses a string without using built-in reverse helpers",
     [{"args": ["abc"], "expected": "cba"}]),
    ("returns True if a number is prime, False otherwise",
     [{"args": [7], "expected": True}, {"args": [8], "expected": False},
      {"args": [1], "expected": False}]),
    ("removes duplicate values from a list while preserving order",
     [{"args": [[3, 1, 3, 2, 1]], "expected": [3, 1, 2]}]),
    ("finds the longest word in a sentence string",
     [{"args": ["the quick brownest fox"], "expected": "brownest"}]),
    ("merges two sorted lists into one sorted list",
     [{"args": [[1, 3], [2, 4]], "expected": [1, 2, 3, 4]}]),
    ("returns the nth Fibonacci number, where fib(0) is 0 and fib(1) is 1",
     [{"args": [7], "expected": 13}, {"args": [0], "expected": 0}]),
    ("computes the greatest common divisor of two positive integers",
     [{"args": [12, 18], "expected": 6}, {"args": [7, 13], "expected": 1}]),
    ("returns the sum of the digits of a non-negative integer",
     [{"args": [1234], "expected": 10}]),
    ("converts a temperature from Celsius to Fahrenheit",
     [{"args": [100], "expected": 212.0}, {"args": [0], "expected": 32.0}]),
    ("returns only the odd numbers from a list, in order",
     [{"args": [[1, 2, 3, 4, 5]], "expected": [1, 3, 5]}]),
    ("returns a new list with every number squared",
     [{"args": [[2, 3]], "expected": [4, 9]}]),
    ("counts how many words are in a sentence string",
     [{"args": ["one two three"], "expected": 3}]),
    ("checks whether two strings are anagrams of each other, ignoring case",
     [{"args": ["Listen", "Silent"], "expected": True},
      {"args": ["cat", "dog"], "expected": False}]),
    ("converts a binary string like '1011' to its decimal value without using int(s, 2)",
     [{"args": ["1011"], "expected": 11}]),
    ("returns the character that appears most often in a string (assume a unique winner)",
     [{"args": ["abbbcc"], "expected": "b"}]),
    # Harder algorithmic specs: give the router genuine escalate signal.
    ("returns the minimum number of single-character insertions, deletions, or "
     "substitutions to transform one string into another (Levenshtein distance)",
     [{"args": ["kitten", "sitting"], "expected": 3},
      {"args": ["", "abc"], "expected": 3}]),
    ("returns the length of the longest strictly increasing subsequence in a list "
     "of integers",
     [{"args": [[10, 9, 2, 5, 3, 7, 101, 18]], "expected": 4},
      {"args": [[5, 4, 3, 2, 1]], "expected": 1}]),
    ("checks whether a string of brackets ()[]{} is balanced and properly nested",
     [{"args": ["([]{})"], "expected": True}, {"args": ["([)]"], "expected": False}]),
    ("returns the indices of the two numbers in a list that add up to a target "
     "value, as a list [i, j] with i < j (assume exactly one solution)",
     [{"args": [[2, 7, 11, 15], 9], "expected": [0, 1]}]),
    ("run-length encodes a string, e.g. 'aaabb' becomes 'a3b2'",
     [{"args": ["aaabb"], "expected": "a3b2"}, {"args": ["abc"], "expected": "a1b1c1"}]),
    ("returns the longest common prefix of a list of strings, or '' if none",
     [{"args": [["flower", "flow", "flight"]], "expected": "fl"}]),
    ("rotates a list to the right by k positions",
     [{"args": [[1, 2, 3, 4, 5], 2], "expected": [4, 5, 1, 2, 3]}]),
    ("returns all prime numbers below a given n as a list, using any method",
     [{"args": [12], "expected": [2, 3, 5, 7, 11]}]),
]
CODEGEN_PHRASINGS = [
    "Write a Python function that {spec}.",
    "Implement a Python function that {spec}.",
    "Create a Python function which {spec}.",
]


def gen_codegen():
    combos = [(s, p) for s in CODEGEN_SPECS for p in CODEGEN_PHRASINGS]
    random.shuffle(combos)
    seen_specs = set()
    picked = []
    for spec, phrasing in combos:          # every spec at least once, then vary
        if spec[0] not in seen_specs:
            seen_specs.add(spec[0])
            picked.append((spec, phrasing))
    for spec, phrasing in combos:
        if len(picked) >= PER_CATEGORY:
            break
        if (spec, phrasing) not in picked:
            picked.append((spec, phrasing))
    for (desc, tests), phrasing in picked[:PER_CATEGORY]:
        add("codegen", phrasing.format(spec=desc), tests, "code_tests")


# ------------------------------------------------------------------ debug (45)

DEBUG_SPECS = [
    ("return the max of a list", "def get_max(nums): return nums[0]",
     [{"args": [[3, 9, 1]], "expected": 9}]),
    ("compute the average of a list", "def avg(nums): return sum(nums) / (len(nums) + 1)",
     [{"args": [[2, 4, 6]], "expected": 4.0}]),
    ("reverse a string", "def rev(s): return s[::2]",
     [{"args": ["abcd"], "expected": "dcba"}]),
    ("count lowercase vowels",
     "def count_vowels(s): return sum(1 for c in s if c in 'aeiou' and c.isupper())",
     [{"args": ["hello"], "expected": 2}]),
    ("return the even numbers", "def evens(nums): return [n for n in nums if n % 2 == 1]",
     [{"args": [[1, 2, 3, 4]], "expected": [2, 4]}]),
    ("sum a list", "def total(nums):\n    t = 0\n    for n in nums:\n        t = n\n    return t",
     [{"args": [[1, 2, 3]], "expected": 6}]),
    ("check if a number is even", "def is_even(n): return n % 2 == 1",
     [{"args": [4], "expected": True}, {"args": [3], "expected": False}]),
    ("return the last element", "def last(items): return items[len(items)]",
     [{"args": [[1, 2, 3]], "expected": 3}]),
    ("double every number", "def double_all(nums): return [n + 2 for n in nums]",
     [{"args": [[1, 5]], "expected": [2, 10]}]),
    ("concatenate two lists", "def combine(a, b): return a - b",
     [{"args": [[1], [2]], "expected": [1, 2]}]),
    ("count how many times x appears in a list",
     "def count_x(nums, x):\n    c = 0\n    for n in nums:\n        if n != x:\n            c += 1\n    return c",
     [{"args": [[1, 2, 1, 1], 1], "expected": 3}]),
    ("return the first n natural numbers starting from 1",
     "def first_n(n): return list(range(n))",
     [{"args": [3], "expected": [1, 2, 3]}]),
    ("compute the factorial of n",
     "def fact(n):\n    r = 0\n    for i in range(1, n + 1):\n        r *= i\n    return r",
     [{"args": [4], "expected": 24}]),
    ("find the minimum of a list",
     "def get_min(nums):\n    m = 0\n    for n in nums:\n        if n < m:\n            m = n\n    return m",
     [{"args": [[5, 3, 8]], "expected": 3}]),
    ("check whether a string is a palindrome",
     "def is_pal(s): return s == s[::-1].upper()",
     [{"args": ["level"], "expected": True}, {"args": ["abc"], "expected": False}]),
    ("compute the sum of squares of a list",
     "def sum_squares(nums): return sum(n * 2 for n in nums)",
     [{"args": [[2, 3]], "expected": 13}]),
    ("return every second element starting from the first",
     "def every_other(items): return items[1::2]",
     [{"args": [[1, 2, 3, 4, 5]], "expected": [1, 3, 5]}]),
    ("test whether n is divisible by both 3 and 5",
     "def div35(n): return n % 3 == 0 or n % 5 == 0",
     [{"args": [15], "expected": True}, {"args": [9], "expected": False}]),
    ("strip whitespace and lowercase each string in a list",
     "def clean(strs): return [s.strip() for s in strs]",
     [{"args": [[" A ", "b "]], "expected": ["a", "b"]}]),
    ("return the absolute difference between two numbers",
     "def diff(a, b): return a - b",
     [{"args": [3, 7], "expected": 4}]),
    ("count words longer than 3 characters",
     "def long_words(sentence):\n    return len([w for w in sentence.split() if len(w) > 3 or w.isupper()])",
     [{"args": ["the AI quick brown fox ran"], "expected": 2}]),
    ("compute the running total of a list",
     "def running(nums):\n    out = []\n    t = 0\n    for n in nums:\n        out.append(t)\n        t += n\n    return out",
     [{"args": [[1, 2, 3]], "expected": [1, 3, 6]}]),
]
DEBUG_PHRASINGS = [
    "This function should {spec} but has a bug: {code} Find and fix it.",
    "The following Python function is supposed to {spec}, but it returns the "
    "wrong result: {code} Identify the bug and provide the corrected code.",
    "Debug this Python function; it is meant to {spec} but doesn't work: {code}",
]


def gen_debug():
    combos = [(s, p) for s in DEBUG_SPECS for p in DEBUG_PHRASINGS]
    random.shuffle(combos)
    seen = set()
    picked = []
    for spec, phrasing in combos:
        if spec[0] not in seen:
            seen.add(spec[0])
            picked.append((spec, phrasing))
    for spec, phrasing in combos:
        if len(picked) >= PER_CATEGORY:
            break
        if (spec, phrasing) not in picked:
            picked.append((spec, phrasing))
    for (desc, code, tests), phrasing in picked[:PER_CATEGORY]:
        add("debug", phrasing.format(spec=desc, code=code), tests, "code_tests")


# -------------------------------------------------------------- sentiment (45)

SENTIMENTS = [
    ("Absolutely love this product, works perfectly every time!", "positive"),
    ("Best purchase I've made all year, highly recommend it.", "positive"),
    ("I was skeptical at first, but this exceeded all my expectations!", "positive"),
    ("Five stars. Setup took two minutes and it has run flawlessly since.", "positive"),
    ("My whole family loves it; we use it every single day.", "positive"),
    ("Incredible value for the price, feels far more premium than it costs.", "positive"),
    ("The support team resolved my issue in minutes. Outstanding service.", "positive"),
    ("This app has genuinely made my mornings easier. Delightful.", "positive"),
    ("Sturdy, beautiful, and exactly as pictured. Couldn't be happier.", "positive"),
    ("Honestly the best headphones I've ever owned, period.", "positive"),
    ("Bought a second one as a gift because mine is so good.", "positive"),
    ("Works like a charm even after six months of heavy daily use.", "positive"),
    ("Terrible customer service, I waited two hours and nobody helped me.", "negative"),
    ("The app crashes constantly and support never replies. Avoid.", "negative"),
    ("Broke after one week. Complete waste of money.", "negative"),
    ("Arrived scratched, missing parts, and two weeks late. Never again.", "negative"),
    ("The battery died within a month and the warranty claim was denied.", "negative"),
    ("Overpriced junk. The photos on the listing are misleading.", "negative"),
    ("I regret this purchase every time I try to use it.", "negative"),
    ("Save your money; the cheaper competitor does everything better.", "negative"),
    ("It stopped working mid-presentation and cost me a client.", "negative"),
    ("Well, at least it wasn't the WORST customer service I've ever had.", "negative"),
    ("I wouldn't say the food was bad, exactly, but I also wouldn't rush back.", "negative"),
    ("Three returns in a row, each replacement worse than the last.", "negative"),
    ("The food arrived on time. It was okay, nothing special.", "neutral"),
    ("It does what it says. No complaints, no surprises.", "neutral"),
    ("The package arrived on Tuesday as scheduled.", "neutral"),
    ("It's a standard cable. It transfers data and charges devices.", "neutral"),
    ("The hotel room matched the listing photos and description.", "neutral"),
    ("Average product for an average price. It's fine.", "neutral"),
    ("The manual explains installation in four steps, and that's what it took.", "neutral"),
    ("I've only used it twice so far, so I can't say much yet.", "neutral"),
    ("Does the job. Nothing more to add.", "neutral"),
    ("The store was open when the website said it would be.", "neutral"),
    ("Received the item; it matches the specifications on the box.", "neutral"),
    ("The battery life is great, but the screen scratches too easily.", "mixed"),
    ("Great camera and display, though the price is steep and shipping was slow.", "mixed"),
    ("Delicious food and lovely decor, but the wait staff were rude to us.", "mixed"),
    ("Fantastic sound quality; shame the ear cushions fall apart in weeks.", "mixed"),
    ("The plot was gripping, yet the ending completely fell flat for me.", "mixed"),
    ("Setup was painless and the design is gorgeous, but it runs hot and loud.", "mixed"),
    ("Love the flavor, hate the packaging that spills every time.", "mixed"),
    ("Speedy delivery and great price, but the color was completely wrong.", "mixed"),
    ("The staff were friendly, though the room smelled of smoke all night.", "mixed"),
    ("Brilliant features when it works, which is only about half the time.", "mixed"),
]


def gen_sentiment():
    prompts = [
        "Classify the sentiment of this review: {t}",
        "Classify the sentiment of this customer feedback and justify it in one "
        "sentence: {t}",
    ]
    for i, (text, label) in enumerate(SENTIMENTS[:PER_CATEGORY]):
        add("sentiment", prompts[i % len(prompts)].format(t=text), label, "label")


# -------------------------------------------------------------------- ner (45)

PEOPLE = ["Maria Sanchez", "Tim Cook", "Elon Musk", "Angela Merkel", "Andy Jassy",
          "Serena Williams", "Jane Goodall", "Emmanuel Macron", "Satya Nadella",
          "Yuki Tanaka", "Priya Patel", "Lars Eriksen", "Amara Diallo",
          "Sofia Rossi", "Diego Fernandez"]
ORGS = ["Fireworks AI", "SoftBank", "World Health Organization", "SpaceX",
        "British Museum", "Cambridge University", "Toyota", "Siemens",
        "Deutsche Bank", "UNICEF", "Netflix", "Airbus", "Samsung", "Pfizer",
        "Goldman Sachs"]
LOCATIONS = ["Berlin", "Tokyo", "Nairobi", "California", "Paris", "Seattle",
             "London", "New York", "Japan", "Singapore", "Toronto", "Mumbai",
             "Sydney", "Cairo", "Oslo"]
DATES = ["last March", "January 15, 2024", "2019", "2002", "last Tuesday",
         "March 3", "September 2022", "Friday", "next quarter", "June 2021",
         "October 7", "the summer of 2018", "early 2025", "December", "April 1"]
NER_TEMPLATES = [
    ("{p} joined {o} in {l} {d}.", ["p", "o", "l", "d"]),
    ("{o} announced that {p} will speak at the {l} conference on {d}.",
     ["o", "p", "l", "d"]),
    ("{p} met executives from {o} in {l} on {d}.", ["p", "o", "l", "d"]),
    ("{o} opened a new office in {l} in {d}.", ["o", "l", "d"]),
    ("{p} visited {l} {d} to sign a deal with {o}.", ["p", "l", "d", "o"]),
    ("On {d}, {o} and {p} unveiled a research center in {l}.", ["d", "o", "p", "l"]),
]
# Deliberately ambiguous sentences: names that double as places/orgs.
NER_HARD = [
    ("Amazon announced that Jordan, the new VP hired from Washington, will lead "
     "the Phoenix office starting in April.",
     ["Jordan:person", "Amazon:organization", "Washington:location",
      "Phoenix:location", "April:date"]),
    ("Turner joined Sterling Bank in Sterling, Colorado, replacing Bell who moved "
     "to a Bell Labs research role in June.",
     ["Turner:person", "Bell:person", "Sterling Bank:organization",
      "Bell Labs:organization", "Sterling:location", "Colorado:location",
      "June:date"]),
    ("Chase left Chase Bank in Houston to join Austin, the startup founded in "
     "Dallas in 2023.",
     ["Chase:person", "Chase Bank:organization", "Houston:location",
      "Austin:organization", "Dallas:location", "2023:date"]),
]


def gen_ner():
    used = set()
    count = 0
    while count < PER_CATEGORY - len(NER_HARD):
        tpl, slots = random.choice(NER_TEMPLATES)
        p, o, l, d = (random.choice(PEOPLE), random.choice(ORGS),
                      random.choice(LOCATIONS), random.choice(DATES))
        sentence = tpl.format(p=p, o=o, l=l, d=d)
        if sentence in used:
            continue
        used.add(sentence)
        type_of = {"p": (p, "person"), "o": (o, "organization"),
                   "l": (l, "location"), "d": (d, "date")}
        entities = [f"{type_of[s][0]}:{type_of[s][1]}" for s in slots]
        add("ner", "Extract all named entities and their types (person, "
            f"organization, location, date) from: {sentence}", entities, "entities")
        count += 1
    for sentence, entities in NER_HARD:
        add("ner", "Extract all named entities and their types (person, "
            f"organization, location, date) from: {sentence}", entities, "entities")


# ---------------------------------------------------------- summarization (45)

PASSAGES = [
    ("The Amazon rainforest, spanning nine countries in South America, produces "
     "about 20 percent of the world's oxygen and hosts an estimated 10 percent "
     "of all species on Earth. Deforestation, driven largely by cattle ranching "
     "and agriculture, has accelerated in recent decades, threatening "
     "biodiversity and contributing to climate change.",
     ["amazon", "rainforest", "deforestation"]),
    ("Electric vehicles are becoming increasingly popular as battery costs fall "
     "and charging infrastructure expands. Governments around the world are "
     "offering incentives to buyers and setting deadlines to phase out "
     "combustion engines, while automakers are investing billions in new "
     "electric models.",
     ["electric", "vehicle", "battery"]),
    ("The ancient city of Pompeii was buried under volcanic ash when Mount "
     "Vesuvius erupted in 79 AD. Rediscovered in the 18th century, its "
     "remarkably preserved ruins offer archaeologists a unique snapshot of "
     "daily Roman life, from bakeries and baths to frescoes and graffiti.",
     ["pompeii", "vesuvius", "roman"]),
    ("Remote work surged during the pandemic and has remained common in many "
     "industries. Companies report savings on office space and access to wider "
     "talent pools, while employees value flexibility, though some managers "
     "worry about collaboration and company culture.",
     ["remote", "work", "flexib"]),
    ("Honeybees play a critical role in agriculture by pollinating roughly a "
     "third of the crops humans eat. In recent years, colony collapse disorder, "
     "pesticides, and habitat loss have caused alarming declines in bee "
     "populations, prompting research and conservation efforts worldwide.",
     ["bee", "pollinat", "decline"]),
    ("The startup raised $4.2 million in seed funding and grew its user base to "
     "50,000 monthly active users within six months, driven mainly by a viral "
     "referral program that cut acquisition costs by half.",
     ["4.2", "50,000", "referral"]),
    ("Coral reefs support about a quarter of all marine species while covering "
     "less than one percent of the ocean floor. Rising sea temperatures cause "
     "coral bleaching, and scientists warn that most reefs could disappear "
     "within decades without rapid emission cuts.",
     ["coral", "reef", "bleach"]),
    ("The James Webb Space Telescope, launched in December 2021, observes the "
     "universe in infrared light, allowing it to peer through dust clouds and "
     "detect galaxies formed shortly after the Big Bang. Its findings are "
     "reshaping theories of early galaxy formation.",
     ["webb", "telescope", "infrared"]),
    ("Global chip shortages during the pandemic forced automakers to idle "
     "factories and delay deliveries. In response, governments in the US and "
     "Europe passed subsidy packages to bring semiconductor manufacturing "
     "closer to home, though new fabs take years to build.",
     ["chip", "semiconductor", "shortage"]),
    ("The Mediterranean diet, rich in olive oil, vegetables, legumes, and fish, "
     "has been linked in long-running studies to lower rates of heart disease "
     "and stroke. Researchers attribute the benefits to unsaturated fats, "
     "fiber, and the diet's low content of processed foods.",
     ["mediterranean", "diet", "heart"]),
    ("Antibiotic resistance is rising as bacteria evolve defenses against "
     "existing drugs, while the pipeline of new antibiotics has slowed because "
     "they are expensive to develop and used sparingly. Health agencies call "
     "for better stewardship of current drugs and new funding models.",
     ["antibiotic", "resistance", "bacteria"]),
    ("Machine translation quality has improved dramatically with neural "
     "networks, enabling real-time conversation across languages. Yet idioms, "
     "cultural context, and low-resource languages remain difficult, and human "
     "translators are still essential for legal and literary texts.",
     ["translation", "neural", "language"]),
    ("Urban vertical farms grow leafy greens indoors under LED lights, using up "
     "to 95 percent less water than field agriculture and no pesticides. High "
     "electricity costs remain the main obstacle to profitability, keeping "
     "most operations focused on premium crops.",
     ["vertical", "farm", "led"]),
    ("The Great Barrier Reef Marine Park generates billions of dollars in "
     "tourism revenue for Australia each year. Authorities balance visitor "
     "access with conservation zones, and recent restoration projects seed "
     "damaged areas with heat-tolerant coral strains.",
     ["barrier", "reef", "australia"]),
    ("Public libraries have reinvented themselves as community hubs, offering "
     "job-search help, maker spaces, and digital literacy classes alongside "
     "books. Visits rebounded after the pandemic even as physical lending "
     "declined, shifting budgets toward programs and e-books.",
     ["librar", "community", "digital"]),
]
SUMMARY_CONSTRAINTS = ["in exactly one sentence", "in no more than 25 words",
                       "in exactly two sentences"]


def gen_summarization():
    for passage, keyterms in PASSAGES:
        for cons in SUMMARY_CONSTRAINTS:
            add("summarization",
                f"Summarize the following paragraph {cons}: {passage}",
                {"keyterms": keyterms}, "summary")


# ---------------------------------------------------------------- factual (45)

FACTUAL = [
    ("What is the capital of Australia?", ["canberra"], "contains_any"),
    ("What is the capital of Canada?", ["ottawa"], "contains_any"),
    ("Who wrote the novel '1984', and in what year was it published?",
     ["orwell", "1949"], "contains_all"),
    ("What is the chemical symbol for gold?", ["au"], "contains_any"),
    ("What is the chemical symbol for iron?", ["fe"], "contains_any"),
    ("What is the largest planet in our solar system?", ["jupiter"], "contains_any"),
    ("Which planet is known as the Red Planet?", ["mars"], "contains_any"),
    ("What is the boiling point of water at sea level in Celsius?",
     ["100"], "contains_any"),
    ("Which country hosted the 2016 Summer Olympics, and in which city?",
     ["brazil", "rio"], "contains_all"),
    ("What does HTTP stand for?", ["hypertext transfer protocol"], "contains_any"),
    ("What does CPU stand for?", ["central processing unit"], "contains_any"),
    ("Who painted the Mona Lisa?", ["da vinci", "leonardo"], "contains_any"),
    ("What is the longest river in the world commonly said to be?",
     ["nile", "amazon"], "contains_any"),
    ("How many continents are there on Earth?", ["seven", "7"], "contains_any"),
    ("What gas do plants primarily absorb during photosynthesis?",
     ["carbon dioxide", "co2"], "contains_any"),
    ("Who developed the theory of general relativity?", ["einstein"], "contains_any"),
    ("What is the smallest prime number?", ["2"], "contains_any"),
    ("In which year did World War II end?", ["1945"], "contains_any"),
    ("What is the currency of Japan?", ["yen"], "contains_any"),
    ("Which ocean is the largest by area?", ["pacific"], "contains_any"),
    ("Who wrote 'Romeo and Juliet'?", ["shakespeare"], "contains_any"),
    ("What is the freezing point of water in Fahrenheit?", ["32"], "contains_any"),
    ("Which element has the atomic number 1?", ["hydrogen"], "contains_any"),
    ("What is the tallest mountain above sea level?", ["everest"], "contains_any"),
    ("Which language has the most native speakers worldwide?",
     ["mandarin", "chinese"], "contains_any"),
    ("What does DNA stand for?", ["deoxyribonucleic"], "contains_any"),
    ("Who was the first person to walk on the Moon?", ["armstrong"], "contains_any"),
    ("What is the capital of France, and on which river does it sit?",
     ["paris", "seine"], "contains_all"),
    ("Which organ in the human body pumps blood?", ["heart"], "contains_any"),
    ("What is the square root of 144?", ["12"], "contains_any"),
    ("Which country is home to the kangaroo?", ["australia"], "contains_any"),
    ("What is the chemical formula for table salt?",
     ["nacl", "sodium chloride"], "contains_any"),
    ("Who is the author of 'Pride and Prejudice'?", ["austen"], "contains_any"),
    ("Which planet is closest to the Sun?", ["mercury"], "contains_any"),
    ("What is the primary language spoken in Brazil?", ["portuguese"], "contains_any"),
    ("In computing, what does RAM stand for?",
     ["random access memory"], "contains_any"),
    ("Which vitamin is produced in human skin when exposed to sunlight?",
     ["vitamin d", "d3"], "contains_any"),
    ("What is the hardest natural substance on Earth?", ["diamond"], "contains_any"),
    ("Which two countries share the longest international land border?",
     ["canada", "united states"], "contains_all"),
    ("Who composed the Ninth Symphony that includes the 'Ode to Joy'?",
     ["beethoven"], "contains_any"),
    ("What is the main gas that makes up Earth's atmosphere?",
     ["nitrogen"], "contains_any"),
    ("Explain the difference between a process and a thread in one or two "
     "sentences, focused on memory.", ["memory"], "contains_any"),
    ("Which sea creature has three hearts and blue blood?",
     ["octopus"], "contains_any"),
    ("What is the capital of South Korea?", ["seoul"], "contains_any"),
    ("How many sides does a hexagon have?", ["six", "6"], "contains_any"),
]


def gen_factual():
    for q, expected, kind in FACTUAL[:PER_CATEGORY]:
        add("factual", q, expected, kind)


def main():
    gen_factual()
    gen_math()
    gen_sentiment()
    gen_summarization()
    gen_ner()
    gen_debug()
    gen_logic()
    gen_codegen()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tasks.json"), "w") as f:
        json.dump(TASKS, f, indent=2)
    with open(os.path.join(OUT_DIR, "expected.json"), "w") as f:
        json.dump(EXPECTED, f, indent=2)
    counts = {}
    for t in TASKS:
        counts[t["task_id"].rsplit("_", 1)[0]] = \
            counts.get(t["task_id"].rsplit("_", 1)[0], 0) + 1
    print(f"wrote {len(TASKS)} tasks to {OUT_DIR}/tasks.json: {counts}")


if __name__ == "__main__":
    main()

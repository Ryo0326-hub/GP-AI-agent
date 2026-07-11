#!/usr/bin/env python3
"""Generate ~80 varied tasks across all 8 categories into test_input/tasks.json,
plus eval/expected.json with ground truth where it is deterministic.
Dev-only: never shipped in the image."""
import json
import os
import random

random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = []
EXPECTED = {}  # task_id -> {"answer": ..., "kind": "number"|"text"|"code_tests"}
_counter = [0]


def add(category, prompt, expected=None, kind="text"):
    _counter[0] += 1
    tid = f"{category}_{_counter[0]:03d}"
    TASKS.append({"task_id": tid, "prompt": prompt})
    if expected is not None:
        EXPECTED[tid] = {"answer": expected, "kind": kind}


# ---------------------------------------------------------------- factual (10)
factual = [
    ("What is the capital of Australia, and what body of water is it near?", None),
    ("Explain in two or three sentences what photosynthesis is and why it matters.", None),
    ("Who wrote the novel '1984', and in what year was it published?", None),
    ("What is the chemical symbol for gold, and what group of metals does it belong to?", None),
    ("Name the largest planet in our solar system and one of its famous features.", None),
    ("What is the difference between a virus and a bacterium?", None),
    ("Which country hosted the 2016 Summer Olympics, and in which city?", None),
    ("What does HTTP stand for, and what is it used for?", None),
    ("What is the boiling point of water at sea level in Celsius and Fahrenheit?", None),
    ("Explain what a stock market index is, giving one example.", None),
]
for q, e in factual:
    add("factual", q, e)

# ------------------------------------------------------------------- math (10)
math_specs = []
for _ in range(4):
    total = random.choice([240, 360, 480, 500, 720])
    pct = random.choice([10, 15, 20, 25])
    extra = random.choice([30, 60, 45, 80])
    ans = total - total * pct // 100 - extra
    math_specs.append((
        f"A store has {total} items. It sells {pct}% of them on Monday and {extra} more on Tuesday. "
        f"How many items remain?", ans))
for _ in range(3):
    speed = random.choice([45, 60, 80])
    hours = random.choice([1.5, 2.5, 3])
    math_specs.append((
        f"A train travels at {speed} km per hour for {hours} hours. How far does it travel in km?",
        round(speed * hours, 2)))
price, disc = 80, 15
math_specs.append((
    f"A jacket costs ${price}. It is discounted by {disc}%, then a $5 coupon is applied. "
    f"What is the final price in dollars?", price * (100 - disc) / 100 - 5))
math_specs.append((
    "Tom has 3 boxes with 24 pencils each. He gives away 18 pencils. How many pencils does he have left?",
    3 * 24 - 18))
math_specs.append((
    "A recipe needs 250 g of flour for 10 cookies. How many grams are needed for 36 cookies?",
    250 * 36 / 10))
for q, e in math_specs:
    add("math", q, e, kind="number")

# -------------------------------------------------------------- sentiment (10)
sentiments = [
    ("The battery life is great, but the screen scratches too easily.", "mixed"),
    ("Absolutely love this product, works perfectly every time!", "positive"),
    ("Terrible customer service, I waited two hours and nobody helped me.", "negative"),
    ("The food arrived on time. It was okay, nothing special.", "neutral"),
    ("Best purchase I've made all year, highly recommend it.", "positive"),
    ("The app crashes constantly and support never replies. Avoid.", "negative"),
    ("Great camera and display, though the price is steep and shipping was slow.", "mixed"),
    ("It does what it says. No complaints, no surprises.", "neutral"),
    ("I was skeptical at first, but this exceeded all my expectations!", "positive"),
    ("Broke after one week. Complete waste of money.", "negative"),
]
for text, label in sentiments:
    add("sentiment", f"Classify the sentiment of this review: {text}", label, kind="label")

# ---------------------------------------------------------- summarization (10)
passages = [
    "The Amazon rainforest, spanning nine countries in South America, produces about 20 percent of the world's oxygen and hosts an estimated 10 percent of all species on Earth. Deforestation, driven largely by cattle ranching and agriculture, has accelerated in recent decades, threatening biodiversity and contributing to climate change.",
    "Electric vehicles are becoming increasingly popular as battery costs fall and charging infrastructure expands. Governments around the world are offering incentives to buyers and setting deadlines to phase out combustion engines, while automakers are investing billions in new electric models.",
    "The ancient city of Pompeii was buried under volcanic ash when Mount Vesuvius erupted in 79 AD. Rediscovered in the 18th century, its remarkably preserved ruins offer archaeologists a unique snapshot of daily Roman life, from bakeries and baths to frescoes and graffiti.",
    "Remote work surged during the pandemic and has remained common in many industries. Companies report savings on office space and access to wider talent pools, while employees value flexibility, though some managers worry about collaboration and company culture.",
    "Honeybees play a critical role in agriculture by pollinating roughly a third of the crops humans eat. In recent years, colony collapse disorder, pesticides, and habitat loss have caused alarming declines in bee populations, prompting research and conservation efforts worldwide.",
]
constraints = [("in exactly one sentence", None), ("in no more than 25 words", None)]
for i, passage in enumerate(passages):
    for cons, _ in constraints:
        add("summarization", f"Summarize the following paragraph {cons}: {passage}")

# ------------------------------------------------------------------- ner (10)
ner_texts = [
    ("Maria Sanchez joined Fireworks AI in Berlin last March.",
     ["Maria Sanchez:person", "Fireworks AI:organization", "Berlin:location", "March:date"]),
    ("Apple CEO Tim Cook visited Tokyo on January 15, 2024 to meet with SoftBank executives.",
     ["Tim Cook:person", "Apple:organization", "Tokyo:location", "January 15, 2024:date", "SoftBank:organization"]),
    ("The World Health Organization opened a new office in Nairobi in 2019.",
     ["World Health Organization:organization", "Nairobi:location", "2019:date"]),
    ("Elon Musk founded SpaceX in California in 2002.",
     ["Elon Musk:person", "SpaceX:organization", "California:location", "2002:date"]),
    ("Angela Merkel met French President Emmanuel Macron in Paris last Tuesday.",
     ["Angela Merkel:person", "Emmanuel Macron:person", "Paris:location", "Tuesday:date"]),
    ("Amazon announced that Andy Jassy will speak at the Seattle conference on March 3.",
     ["Amazon:organization", "Andy Jassy:person", "Seattle:location", "March 3:date"]),
    ("The Louvre in Paris displayed works borrowed from the British Museum in London.",
     ["Louvre:organization", "Paris:location", "British Museum:organization", "London:location"]),
    ("Serena Williams won her final match in New York in September 2022.",
     ["Serena Williams:person", "New York:location", "September 2022:date"]),
    ("Toyota and Honda both reported record sales in Japan last quarter.",
     ["Toyota:organization", "Honda:organization", "Japan:location"]),
    ("Dr. Jane Goodall gave a lecture at Cambridge University on Friday.",
     ["Jane Goodall:person", "Cambridge University:organization", "Friday:date"]),
]
for text, entities in ner_texts:
    add("ner", f"Extract all named entities and their types (person, organization, location, date) from: {text}",
        entities, kind="entities")

# ----------------------------------------------------------------- debug (10)
debug_specs = [
    ("This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.",
     [{"args": [[3, 9, 1]], "expected": 9}]),
    ("This function should compute the average but has a bug: def avg(nums): return sum(nums) / (len(nums) + 1). Find and fix it.",
     [{"args": [[2, 4, 6]], "expected": 4.0}]),
    ("This function should reverse a string but has a bug: def rev(s): return s[::2]. Find and fix it.",
     [{"args": ["abcd"], "expected": "dcba"}]),
    ("This function should count vowels but has a bug: def count_vowels(s): return sum(1 for c in s if c in 'aeiou' and c.isupper()). Find and fix it.",
     [{"args": ["hello"], "expected": 2}]),
    ("This function should return even numbers but has a bug: def evens(nums): return [n for n in nums if n % 2 == 1]. Find and fix it.",
     [{"args": [[1, 2, 3, 4]], "expected": [2, 4]}]),
    ("This function should sum a list but has a bug: def total(nums):\n    t = 0\n    for n in nums:\n        t = n\n    return t\nFind and fix it.",
     [{"args": [[1, 2, 3]], "expected": 6}]),
    ("This function should check if a number is even but has a bug: def is_even(n): return n % 2 == 1. Find and fix it.",
     [{"args": [4], "expected": True}, {"args": [3], "expected": False}]),
    ("This function should return the last element but has a bug: def last(items): return items[len(items)]. Find and fix it.",
     [{"args": [[1, 2, 3]], "expected": 3}]),
    ("This function should double every number but has a bug: def double_all(nums): return [n + 2 for n in nums]. Find and fix it.",
     [{"args": [[1, 5]], "expected": [2, 10]}]),
    ("This function should concatenate two lists but has a bug: def combine(a, b): return a - b. Find and fix it.",
     [{"args": [[1], [2]], "expected": [1, 2]}]),
]
for prompt, tests in debug_specs:
    add("debug", prompt, tests, kind="code_tests")

# ----------------------------------------------------------------- logic (10)
logic_specs = [
    ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?", "Sam"),
    ("Three colleagues, Ana, Ben, and Cara, each drink a different beverage: coffee, tea, juice. Ana does not drink tea. Cara drinks juice. Who drinks tea?", "Ben"),
    ("Four kids, Max, Nia, Omar, and Pia, each play a different sport: soccer, tennis, chess, golf. Max plays chess. Nia does not play soccer and does not play golf. Omar plays golf. Who plays soccer?", "Pia"),
    ("Three neighbors, Ivy, Jack, and Kim, each own a different car: sedan, truck, coupe. Jack owns the truck. Ivy does not own the coupe. Who owns the coupe?", "Kim"),
    ("Three students, Leo, Mia, and Noah, each like a different fruit: apple, banana, cherry. Noah likes the banana. Leo does not like the apple. Who likes the apple?", "Mia"),
    ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer yes or no and explain briefly.", "no"),
    ("Amy is taller than Beth. Beth is taller than Cara. Who is the shortest?", "Cara"),
    ("A farmer has chickens and rabbits, 10 heads and 28 legs in total. How many rabbits are there?", "4"),
    ("Dan always lies and Erin always tells the truth. Dan says 'Erin said it is raining.' Is it necessarily true that Erin said that? Answer yes or no and explain.", "no"),
    ("In a race, Uma finished before Vic, and Vic finished before Wes. Tia finished after Uma but before Vic. Who finished second?", "Tia"),
]
for prompt, ans in logic_specs:
    add("logic", prompt, ans, kind="contains")

# --------------------------------------------------------------- codegen (10)
codegen_specs = [
    ("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.",
     [{"args": [[1, 2, 2, 3]], "expected": 2}, {"args": [[5, 1, 4]], "expected": 4}]),
    ("Write a Python function that checks whether a string is a palindrome (ignore case).",
     [{"args": ["Racecar"], "expected": True}, {"args": ["hello"], "expected": False}]),
    ("Write a Python function that computes the factorial of a non-negative integer.",
     [{"args": [5], "expected": 120}, {"args": [0], "expected": 1}]),
    ("Write a Python function that returns the sum of all even numbers in a list.",
     [{"args": [[1, 2, 3, 4]], "expected": 6}]),
    ("Write a Python function that counts the vowels in a string (lowercase input).",
     [{"args": ["hello world"], "expected": 3}]),
    ("Write a Python function that reverses a string without using built-in reverse functions.",
     [{"args": ["abc"], "expected": "cba"}]),
    ("Write a Python function that returns True if a number is prime, False otherwise.",
     [{"args": [7], "expected": True}, {"args": [8], "expected": False}, {"args": [1], "expected": False}]),
    ("Write a Python function that removes duplicate values from a list while preserving order.",
     [{"args": [[3, 1, 3, 2, 1]], "expected": [3, 1, 2]}]),
    ("Write a Python function that finds the longest word in a sentence string.",
     [{"args": ["the quick brownest fox"], "expected": "brownest"}]),
    ("Write a Python function that merges two sorted lists into one sorted list.",
     [{"args": [[1, 3], [2, 4]], "expected": [1, 2, 3, 4]}]),
]
for prompt, tests in codegen_specs:
    add("codegen", prompt, tests, kind="code_tests")

# ---------------------------------------------------------------------- write
os.makedirs(os.path.join(ROOT, "test_input"), exist_ok=True)
with open(os.path.join(ROOT, "test_input", "tasks.json"), "w") as f:
    json.dump(TASKS, f, indent=2)
with open(os.path.join(ROOT, "eval", "expected.json"), "w") as f:
    json.dump(EXPECTED, f, indent=2)
print(f"wrote {len(TASKS)} tasks to test_input/tasks.json "
      f"({len(EXPECTED)} with deterministic expectations)")

# 19-task subset mirroring the real eval shape: 2-3 per category, mixed
# difficulty (first / middle / last of each category block). Same task_ids,
# so eval/expected.json applies unchanged.
SUBSET_COUNTS = {"factual": 3, "math": 3, "sentiment": 2, "summarization": 2,
                 "ner": 2, "debug": 2, "logic": 3, "codegen": 2}
by_cat = {}
for t in TASKS:
    by_cat.setdefault(t["task_id"].rsplit("_", 1)[0], []).append(t)
subset = []
for cat, count in SUBSET_COUNTS.items():
    block = by_cat[cat]
    picks = [0, len(block) // 2, len(block) - 1][:count]
    subset.extend(block[j] for j in picks)
assert len(subset) == 19, len(subset)
os.makedirs(os.path.join(ROOT, "test_input_19"), exist_ok=True)
with open(os.path.join(ROOT, "test_input_19", "tasks.json"), "w") as f:
    json.dump(subset, f, indent=2)
print("wrote 19-task rehearsal subset to test_input_19/tasks.json")

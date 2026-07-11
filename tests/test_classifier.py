from classifier import classify


def test_factual():
    assert classify("What is the capital of Australia, and what body of water is it near?") == "factual"
    assert classify("Explain the concept of photosynthesis.") == "factual"


def test_math():
    assert classify("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?") == "math"
    assert classify("Calculate 15% of 380 plus 42.") == "math"
    assert classify("A train travels 60 km per hour for 2.5 hours. How far does it go?") == "math"


def test_sentiment():
    assert classify("Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.") == "sentiment"
    assert classify("What is the sentiment of this tweet: I love Mondays!") == "sentiment"


def test_summarization():
    assert classify("Summarize the following paragraph in exactly one sentence: ...") == "summarization"
    assert classify("Summarise this article in no more than 30 words.") == "summarization"


def test_ner():
    assert classify("Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.") == "ner"


def test_debug():
    assert classify("This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.") == "debug"
    assert classify("THIS HAS A BUG: DEF GET_MAX(NUMS): RETURN NUMS[0]") == "debug"


def test_codegen():
    assert classify("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.") == "codegen"
    assert classify("Implement a function to check whether a string is a palindrome.") == "codegen"


def test_logic():
    assert classify("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?") == "logic"
    assert classify("Anna always lies and Ben always tells the truth. Anna says it is raining. Is it raining?") == "logic"

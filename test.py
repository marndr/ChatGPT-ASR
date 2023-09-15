import pytest
from main import levenshtein_distance_custom

def test_levenshtein_distance_custom_same_strings():
    s1 = "hello"
    s2 = "hello"
    result = levenshtein_distance_custom(s1, s2)
    assert result == 0, "Expected distance between identical strings to be 0"

def test_levenshtein_distance_custom_insertion():
    s1 = "hello"
    s2 = "hellxo"
    result = levenshtein_distance_custom(s1, s2)
    assert result == 1, "Expected 1 insertion operation"

def test_levenshtein_distance_custom_deletion():
    s1 = "hello"
    s2 = "helo"
    result = levenshtein_distance_custom(s1, s2)
    assert result == 1, "Expected 1 deletion operation"

def test_levenshtein_distance_custom_substitution():
    s1 = "hello"
    s2 = "hallo"
    result = levenshtein_distance_custom(s1, s2)
    assert result == 1, "Expected 1 substitution operation"


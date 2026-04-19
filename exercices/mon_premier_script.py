import unittest

MINIMUM_LETTERS = 7

def count_names_with_more_than_seven_letters(first_names: list[str]) -> int:
    count = 0
    for first_name in first_names:
        if len(first_name) > MINIMUM_LETTERS:
            count += 1
    return count

class TestNamesMethod(unittest.TestCase):
    def test_count_names_with_more_than_seven_letters(self):
        first_names = ["Guillaume", "Gilles", "Juliette", "Antoine", "François", "Cassandre"]
        result = count_names_with_more_than_seven_letters(first_names=first_names)
        self.assertEqual(result, 4)

if __name__ == '__main__':
    unittest.main()
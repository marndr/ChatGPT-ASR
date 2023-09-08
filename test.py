import unittest
from main import check_asr_errors

class TestASRErrorChecking(unittest.TestCase):
    def setUp(self):
        self.asr_outputs = [
            "I prefer see over coffee.",
            "I need to catch a drain to London.",
            "I'll beat you at the coffee shop.",
            "He's a professional sheaf.",
            "The leather is nice today.",
            "The weather is mice today.",
            "Please pass me the vault.",
            "I'll be their in five minuets.",
            "I'm going to the gross restore.",
            "I won't to go two the beech.",
            "The son is shining brightly in the sky.",
            "The son sets at the beech are always stunting.",
            "I prefer tea over coffee.",
            "I need to catch a train to London.",
            "I'll meet you at the coffee shop.",
            "He's a professional chef.",
            "The weather is nice today.",
            "Please pass me the salt.",
            "I'll be there in five minutes.",
            "I'm going to the grocery store.",
            "I'd like to order a pizza with pepperoni and mushrooms."
        ]

    def test_asr_error_detection(self):
        # Test ASR error detection for all ASR outputs
        asr_results = check_asr_errors(self.asr_outputs)

        for i, (original, result) in enumerate(zip(self.asr_outputs, asr_results), 1):
            with self.subTest(i=i):
                self.assertNotEqual(original, result, f"ASR Input {i}: '{original}' has no error but was flagged as an error.")
                print(f"ASR Input {i}: '{original}'")
                print(f"ASR Result {i}: '{result}'")
                print("=" * 50)

if __name__ == "__main__":
    unittest.main()
































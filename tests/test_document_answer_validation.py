import unittest


try:
    from orchestrator.answer_validation import validate_extracted_answer
except ModuleNotFoundError:
    validate_extracted_answer = None


class DocumentAnswerValidationTests(unittest.TestCase):
    def test_valid_answer_is_accepted(self):
        self.assertIsNotNone(validate_extracted_answer)
        valid, reason = validate_extracted_answer(
            "Tjong A Fie adalah saudagar dan tokoh bersejarah di Kota Medan."
        )

        self.assertTrue(valid)
        self.assertEqual(reason, "")

    def test_not_found_answer_is_rejected(self):
        self.assertIsNotNone(validate_extracted_answer)
        for answer in [
            "Tidak ditemukan",
            "Tidak ditemukan.\n\nDokumen: data_hotel.pdf, Halaman: 26",
        ]:
            with self.subTest(answer=answer):
                valid, reason = validate_extracted_answer(answer)
                self.assertFalse(valid)
                self.assertEqual(reason, "not_found")

    def test_prompt_instruction_leakage_is_rejected(self):
        self.assertIsNotNone(validate_extracted_answer)

        for answer in [
            "Rule 2: If not found, reply exactly: Tidak ditemukan",
            "ATURAN KETAT: jawab berdasarkan Referensi Teks",
            "If not found, reply exactly with Tidak ditemukan",
        ]:
            with self.subTest(answer=answer):
                valid, reason = validate_extracted_answer(answer)
                self.assertFalse(valid)
                self.assertEqual(reason, "prompt_leakage")

    def test_empty_answer_is_rejected(self):
        self.assertIsNotNone(validate_extracted_answer)
        valid, reason = validate_extracted_answer("   ")

        self.assertFalse(valid)
        self.assertEqual(reason, "empty")


if __name__ == "__main__":
    unittest.main()

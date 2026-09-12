import unittest

from nlp.text_features import SEED_THEMES, THEMES, analyze, get_lexicon


class TestTextFeatures(unittest.TestCase):
    def test_lexicon_keeps_seeds_and_adds_wordnet_synonyms(self):
        lexicon = get_lexicon()
        self.assertEqual(set(lexicon), set(SEED_THEMES))
        for theme, seeds in SEED_THEMES.items():
            self.assertTrue(set(seeds) <= set(lexicon[theme]), theme)
        self.assertGreater(sum(map(len, lexicon.values())), sum(map(len, SEED_THEMES.values())))

    def test_lexicon_uses_curated_senses_not_first_listed_sense(self):
        lexicon = get_lexicon()

        self.assertIn("crude oil", lexicon["fossil_fuels"])
        self.assertNotIn("word of mouth", lexicon["fossil_fuels"])
        self.assertNotIn("contemporaries", lexicon["power_utilities"])
        self.assertFalse({"line", "net", "flat"} & {term for terms in lexicon.values() for term in terms})

    def test_analyze_detects_themes_and_countries(self):
        [row] = analyze(["The company produces crude oil and operates natural gas pipelines in Canada and Mexico."])

        self.assertEqual(set(row["themes"]), set(THEMES))
        self.assertGreater(row["themes"]["fossil_fuels"], 0)
        self.assertEqual(row["themes"]["health"], 0)
        self.assertGreaterEqual(row["n_countries"], 2)
        tokens = row["clean_text"].split()
        self.assertIn("pipeline", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("canada", tokens)


if __name__ == "__main__":
    unittest.main()

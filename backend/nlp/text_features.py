"""NLP features from company business descriptions.

spaCy does tokenisation, lemmatisation, NER and lexicon matching; NLTK's WordNet expands the ESG theme lexicon.
"""
from functools import lru_cache

import spacy
from nltk.corpus import wordnet as wn
from spacy.matcher import PhraseMatcher

SEED_THEMES = {
    "fossil_fuels": ["oil", "gas", "coal", "petroleum", "drilling", "refinery", "pipeline", "crude"],
    "heavy_industry": ["mining", "steel", "cement", "chemical", "metal", "aluminum", "fertilizer", "paper"],
    "power_utilities": ["electricity", "utility", "generation", "transmission", "nuclear", "electric utility"],
    "clean_energy": ["solar", "wind", "renewable", "battery", "hydrogen", "energy efficiency"],
    "transport": ["airline", "aircraft", "truck", "railroad", "shipping", "vehicle", "logistics"],
    "health": ["pharmaceutical", "drug", "medical", "clinical", "hospital", "therapy", "vaccine"],
    "finance": ["bank", "insurance", "loan", "credit", "mortgage", "investment", "brokerage"],
    "data_tech": ["software", "cloud", "internet", "semiconductor", "platform", "advertising", "cybersecurity"],
    "consumer_labor": ["restaurant", "retail", "store", "apparel", "franchise", "warehouse", "staffing"],
    "real_estate": ["property", "apartment", "office building", "lease", "tenant", "real estate"],
}
# Chosen sense per seed: WordNet's first-listed sense is often wrong here (pipeline -> "word of mouth").
WORDNET_SENSES = {
    "oil": "petroleum.n.01",
    "gas": "gasoline.n.01",
    "coal": "coal.n.01",
    "petroleum": "petroleum.n.01",
    "drilling": "boring.n.02",
    "refinery": "refinery.n.01",
    "pipeline": "pipeline.n.02",
    "crude": "petroleum.n.01",
    "mining": "mining.n.01",
    "steel": "steel.n.01",
    "cement": "cement.n.02",
    "chemical": "chemical.n.01",
    "metal": "metallic_element.n.01",
    "aluminum": "aluminum.n.01",
    "fertilizer": "fertilizer.n.01",
    "paper": "paper.n.01",
    "electricity": "electricity.n.02",
    "utility": "utility.n.01",
    "battery": "battery.n.02",
    "airline": "airline.n.02",
    "aircraft": "aircraft.n.01",
    "truck": "truck.n.01",
    "railroad": "railway.n.01",
    "shipping": "transportation.n.05",
    "vehicle": "vehicle.n.01",
    "logistics": "logistics.n.01",
    "pharmaceutical": "pharmaceutical.n.01",
    "drug": "drug.n.01",
    "hospital": "hospital.n.01",
    "therapy": "therapy.n.01",
    "vaccine": "vaccine.n.01",
    "bank": "depository_financial_institution.n.01",
    "insurance": "insurance.n.01",
    "loan": "loan.n.01",
    "credit": "credit.n.02",
    "mortgage": "mortgage.n.01",
    "investment": "investment.n.02",
    "brokerage": "brokerage.n.01",
    "software": "software.n.01",
    "internet": "internet.n.01",
    "semiconductor": "semiconductor_device.n.01",
    "platform": "platform.n.03",
    "advertising": "advertising.n.02",
    "restaurant": "restaurant.n.01",
    "retail": "retail.n.01",
    "store": "shop.n.01",
    "apparel": "apparel.n.01",
    "franchise": "franchise.n.02",
    "warehouse": "warehouse.n.01",
    "apartment": "apartment.n.01",
    "office building": "office_building.n.01",
    "lease": "lease.n.01",
    "tenant": "tenant.n.01",
    "real estate": "real_property.n.01",
}
# Correct synonyms that collide with everyday business phrasing ("product line", "net sales", "flat-rolled").
EXCLUDED_SYNONYMS = {"line", "net", "flat", "boring", "airway", "rental", "immovable", "al", "atomic number 13"}
# Stock phrasing in these descriptions ("founded in ... and headquartered in ...") that carries no ESG signal.
BOILERPLATE = {
    "company", "corporation", "inc", "segment", "subsidiary", "provide", "offer", "include", "consist", "relate",
    "operate", "base", "headquarter", "incorporate", "found", "formerly", "know", "name", "change", "enable", "share",
}
THEMES = list(SEED_THEMES)
EXCLUDED_ENTITIES = {"ORG", "PERSON", "GPE", "DATE", "CARDINAL", "MONEY", "PERCENT", "ORDINAL", "QUANTITY"}


def expand_with_wordnet(seeds: dict[str, list[str]]) -> dict[str, list[str]]:
    lexicon = {}
    for theme, words in seeds.items():
        terms = set(words)
        for word in words:
            if word in WORDNET_SENSES:
                synset = wn.synset(WORDNET_SENSES[word])
                terms.update(lemma.name().replace("_", " ").lower() for lemma in synset.lemmas())
        lexicon[theme] = sorted(terms - EXCLUDED_SYNONYMS)
    return lexicon


@lru_cache(maxsize=1)
def get_lexicon() -> dict[str, list[str]]:
    return expand_with_wordnet(SEED_THEMES)


@lru_cache(maxsize=1)
def get_nlp():
    return spacy.load("en_core_web_sm", disable=["parser"])


@lru_cache(maxsize=1)
def get_matcher() -> PhraseMatcher:
    nlp = get_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    for theme, terms in get_lexicon().items():
        matcher.add(theme, list(nlp.pipe(terms)))
    return matcher


def analyze(texts: list[str]) -> list[dict]:
    """Return clean lemma text, theme densities (matches per 100 tokens) and distinct-country count per text."""
    nlp, matcher = get_nlp(), get_matcher()
    results = []
    for doc in nlp.pipe((t or "" for t in texts), batch_size=64):
        lemmas = [
            tok.lemma_.lower()
            for tok in doc
            if tok.is_alpha
            and not tok.is_stop
            and len(tok) > 2
            and tok.ent_type_ not in EXCLUDED_ENTITIES
            and tok.lemma_.lower() not in BOILERPLATE
        ]
        counts = dict.fromkeys(THEMES, 0)
        for match_id, _, _ in matcher(doc):
            counts[nlp.vocab.strings[match_id]] += 1
        n_tokens = max(len(doc), 1)
        results.append(
            {
                "clean_text": " ".join(lemmas),
                "themes": {k: round(v * 100 / n_tokens, 3) for k, v in counts.items()},
                "n_countries": len({ent.text for ent in doc.ents if ent.label_ == "GPE"}),
            }
        )
    return results

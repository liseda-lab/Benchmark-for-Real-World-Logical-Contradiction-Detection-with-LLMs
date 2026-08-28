import re
from datetime import datetime
from typing import Tuple, Optional


class TripleVerbalizer:

    TEMPLATES = {
        "en": {
            "subclass of": "Every {subject} is a(n) {object}.",
            "NOT subclass of": "{subject} is/are not a type of {object}.",
            "instance of": "{subject} is a(n) {object}.",
            "NOT instance of": "{subject} is/are not a(n) {object}.",
            "disjoint with": "No {subject} may be a(n) {object}.",
            "NOT disjoint with": "Some {subject} may also be a(n) {object}.",
            "has_function": "The function of the protein {subject} is {object}.",
            "NOT has_function": "The protein {subject} does not have the function {object}.",
            "has_biological_process": "The protein {subject} has the biological process {object}.",
            "NOT has_biological_process": "The protein {subject} does not have the biological process {object}.",
            "located_in": "The protein {subject} is located in {object}.",
            "NOT located_in": "The protein {subject} is not located in {object}.",
            "enables": "The protein {subject} enables {object}.",
            "NOT enables": "The protein {subject} does not enable {object}.",
            "property constraint": 'The property "{subject}" allows only one value at any given time.',
            "NOT property constraint": 'The property "{subject}" allows multiple values at any given time.',
            "default property constraint": 'The property "{subject}" has constraint {object}.',
            "default": 'The {relation} of {subject} is {object}.'
        },
        "es": {
            "subclase de": "Cada {subject} es un(a) {object}.",
            "NOT subclase de": "El/La {subject} no es un tipo de {object}.",
            "instancia de": "{subject} es un(a) {object}.",
            "NOT instancia de": "El/La {subject} no es un(a) {object}.",
            "disjunto con": "Ningún(a) {subject} es un(a) {object}.",
            "NOT disjunto con": "Algunos(as) {subject} son un(a) {object}.",
            "restricción de propiedad de Wikidata": 'La propiedad "{subject}" permite solo un valor a la vez.',
            "NOT restricción de propiedad de Wikidata": 'La propiedad "{subject}" permite múltiples valores a la vez.',
            "default property constraint": 'La propiedad "{subject}" tiene la restricción {object}.',
            "default": 'La {relation} de {subject} es {object}.'
        },
        "fr": {
            "sous-classe de": "Chaque {subject} est un(e) {object}.",
            "NOT sous-classe de": "Le/La {subject} n'est pas un type de {object}.",
            "nature de l'élément": "{subject} est un(e) {object}.",
            "NOT nature de l'élément": "Le/La {subject} n'est pas un(e) {object}.",
            "disjoint avec": "Aucun(e) {subject} est un(e) {object}.",
            "NOT disjoint avec": "Certains {subject} sont un(e) {object}.",
            "contrainte de propriété Wikidata": 'La propriété "{subject}" permet une seule valeur, dans un lieu temporel.',
            "NOT contrainte de propriété Wikidata": 'La propriété "{subject}" permet plusieurs valeurs, dans un lieu temporel.',
            "default property constraint": 'La propriété "{subject}" a la contrainte {object}.',
            "default": 'Le(a) {relation} de {subject} est {object}.'
        }
    }

    # Mapping from the friendly language names used in the dataset files
    # to BCP-47 codes used in TEMPLATES.
    LANGUAGE_NAME_TO_CODE = {
        "english":    "en",
        "spanish":    "es",
        "french":     "fr"
    }

    def resolve_lang_code(self, language: str) -> str:
        """Accept either a friendly name ('english') or a BCP-47 code ('en')."""
        lower = language.lower().strip()
        return self.LANGUAGE_NAME_TO_CODE.get(lower, lower)

    def parse_triple(self, triple_str: str) -> Tuple[str, str, str, Optional[str], Optional[str]]:
        """
        Parses:
          (subj; rel; obj)
          (subj; rel; obj) [start, end]
        """
        triple_str = triple_str.strip()

        start_idx = triple_str.find("(")
        end_idx   = triple_str.rfind(")")

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"Invalid triple format: {triple_str}")

        triple_content = triple_str[start_idx + 1 : end_idx]
        parts = [p.strip() for p in triple_content.split(";")]

        if len(parts) != 3:
            raise ValueError(f"Invalid triple structure: {triple_str}")

        subject, relation, obj = parts

        time_part = triple_str[end_idx + 1 :].strip()
        start, end = None, None
        if time_part.startswith("[") and time_part.endswith("]"):
            time_content = time_part[1:-1]
            time_parts   = [t.strip() for t in time_content.split(",")]
            if len(time_parts) == 2:
                start, end = time_parts

        return subject, relation, obj, start, end

    # ── Template-based verbalization ──────────────────────────────────────────

    def verbalize_from_template(
        self, subject: str, relation: str, obj: str, lang_code: str
    ) -> Optional[str]:
        """
        Try to match `relation` (or 'NOT <relation>') against the templates
        for `lang_code`.  Returns a filled string or None if no match.
        """
        templates = self.TEMPLATES.get(lang_code, {})

        # Direct match
        if relation in templates:
            tmpl = templates[relation]
            return tmpl.format(subject=subject, object=obj)

        # Prefix-NOT match: e.g. relation = "NOT属性約束" after stripping spaces
        not_key = f"NOT {relation}"
        if not_key in templates:
            tmpl = templates[not_key]
            return tmpl.format(subject=subject, object=obj)

        tmpl = templates.get("default")
        if tmpl:
            return tmpl.format(subject=subject, relation=relation, object=obj)
        return None

    # ── English fallback logic (kept from original) ───────────────────────────

    def verbalize_logical(self, subject: str, relation: str, obj: str) -> Optional[str]:
        if relation in ("subClassOf", "subclass of", "subclass"):
            return f"Every {subject} is a {obj}."
        if relation in ("disjointWith", "disjoint with", "disjoint"):
            return f"No {subject} is a {obj}."
        if relation == "has_function":
            return f"The function of the protein {subject} is {obj}."
        if relation == "NOT has_function":
            return f"The protein {subject} does not have the function {obj}."
        if relation == "has_phenotype":
            return f"The phenotype of the disease {subject} is {obj}."
        if relation == "NOT has_phenotype":
            return f"The disease {subject} does not have the phenotype {obj}."
        return None

    def verbalize_property_constraint(self, subject: str, relation: str, obj: str) -> str:
        if relation == "property constraint" and obj == "single-value constraint":
            return f'The property "{subject}" allows only one value at any given time.'
        if relation == "NOT property constraint" and obj == "single-value constraint":
            return f'The property "{subject}" allows multiple values at any given time.'
        return f'The property "{subject}" has constraint {obj}.'

    def verbalize_object_property(
        self, subject: str, relation: str, obj: str,
        start: Optional[str], end: Optional[str]
    ) -> str:
        if relation.endswith("of"):
            return f"{obj} is the {relation} {subject}."
        if relation.endswith("by"):
            return f"{obj} is {relation} the {subject}."
        return f"The {relation} of {subject} is {obj}."

    # ── Main entry point ──────────────────────────────────────────────────────

    def verbalize(self, triple_str: str, language: str = "en") -> str:
        """
        Verbalize a triple string.

        `language` can be a BCP-47 code ('en', 'zh') or a friendly name
        ('english', 'chinese'). Falls back to English hardcoded logic when
        no template matches.
        """
        subject, relation, obj, start, end = self.parse_triple(triple_str)
        lang_code = self.resolve_lang_code(language)

        # 1. Try the template for the resolved language
        result = self.verbalize_from_template(subject, relation, obj, lang_code)
        if result:
            return result

        # 2. If not English, try the English template as a fallback
        if lang_code != "en":
            result = self.verbalize_from_template(subject, relation, obj, "en")
            if result:
                return result

        # 3. Final fallback: hardcoded English logic
        if "property constraint" in relation:
            return self.verbalize_property_constraint(subject, relation, obj)

        logical = self.verbalize_logical(subject, relation, obj)
        if logical:
            return logical

        return self.verbalize_object_property(subject, relation, obj, start, end)
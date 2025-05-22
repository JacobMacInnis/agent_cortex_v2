import re
from longterm_memory import LongTermMemory

class FactSaver:
    def __init__(self, memory_store: LongTermMemory):
        self.memory_store = memory_store

    def maybe_save_fact(self, user_input: str) -> None:
        # print(f"[LTM] User input: {user_input}")
        """
        Naively extract known fact types from user input and save them to long-term memory.
        Extend this for names, locations, preferences, etc.
        """
        normalized = user_input.strip()

        patterns = [
            (r"\bmy name is ([A-Z][a-z]+)\b", "The user's name is {fact}."),
            (r"\bi live in ([A-Za-z\s,]+)\b", "The user lives in {fact}."),
            (r"\bi am from ([A-Za-z\s,]+)\b", "The user is from {fact}."),
        ]

        for pattern, template in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            # print(f"[LTM] Template: {template}, Match: {match}")
            if match:
                fact = match.group(1).strip()
                entry = template.format(fact=fact)
                # print(f"[LTM] Saving fact: {entry}")
                self.memory_store.save_fact(entry)

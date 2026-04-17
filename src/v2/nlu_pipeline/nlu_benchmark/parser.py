import re
from typing import Optional

class ActionParser:
    VALID = {"MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"}

    SYNONYMS = {
        "north": "MOVE_NORTH",
        "up": "MOVE_NORTH",
        "south": "MOVE_SOUTH",
        "down": "MOVE_SOUTH",
        "east": "MOVE_EAST",
        "right": "MOVE_EAST",
        "west": "MOVE_WEST",
        "left": "MOVE_WEST",
    }

    def parse(self, text: str, allow_regex_fallback: bool = True) -> tuple[Optional[str], str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        trailing = lines[-5:] if len(lines) >= 5 else lines

        for line in reversed(trailing):
            m = re.match(r"(?i)^FINAL_ACTION\s*:\s*(.+?)\s*$", line)
            if m:
                tok = m.group(1).strip().upper().replace(" ", "_")
                if tok in self.VALID:
                    return tok, "OK_FINAL_ACTION"
                return None, "INVALID_TOKEN"

        if allow_regex_fallback:
            norm = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
            matches = []
            for word, canonical in self.SYNONYMS.items():
                for m in re.finditer(rf"\b{re.escape(word)}\b", norm):
                    matches.append((m.start(), canonical))
            if matches:
                matches.sort(key=lambda x: x[0])
                return matches[-1][1], "OK_REGEX"

        return None, "NO_MATCH"
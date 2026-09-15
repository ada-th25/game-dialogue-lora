# Dialogue Schema

## Input fields
- `character_archetype`: one of [grumpy_blacksmith, nervous_merchant, ...]  (fixed vocabulary)
- `mood`: one of [annoyed, cheerful, fearful, ...]  (fixed vocabulary)
- `topic`: short free text (e.g. "broken sword")
- `constraints`: optional, e.g. "max_20_words"

## Output
- Single line of in-character spoken dialogue, no narration.

## Rationale
Fixed vocabularies for archetype/mood keep evaluation tractable — 
we can check whether generated dialogue matches the requested 
archetype/mood systematically, rather than relying on open-ended judgement.
# SS-004: prospectively repair English retention

SS-002's first replay lifetime learned three new-language interfaces but reduced
English development intent from127/160 to93/160, missing the registered retention
gate. The matched current-only lifetime ended at38/160. All final-language rows
remain unopened. Preserve both candidates and the original English interface.

Continue the first replay lifetime from update1200 with its exact Adam state.
Use one fixed800-update curriculum:16 English training examples plus16 examples
drawn uniformly across the Spanish/French/German teaching collections per batch.
Sample English from its original11514 teaching rows rather than its256-record
reservoir. This is a larger rehearsal resource and is charged and labelled as
such. It is not a storage-matched comparison with SS-002. Set lr0.0005, random
sampling seed2721, one CPU thread. Keep all non-stream tensors unchanged.

Preserve checkpoints every100 updates, Adam state and sample RNG, with explicit
parent checkpoint identity. No reset or repetition of the completed1200 updates.
Use the original development gate: English loss<=10percentage points relative to
the127/160 baseline; each new language>=35% intent. Add this declared continuation
to the single final-language cohort after freezing its result. Do not tune on
final results. Fixed replay/mixing remains engineering, not learned eta.

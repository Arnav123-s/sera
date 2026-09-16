# Research rationale

I checked primary sources for two specific design choices on 16 September 2026 UTC.

Lake and Baroni's [SCAN study](https://proceedings.mlr.press/v80/lake18a.html) separates success on familiar combinations from systematic generalization to new constructions. I therefore evaluate new parameter combinations and new sentence structures separately. This local arithmetic study is not a reproduction of SCAN.

Geifman and El-Yaniv's [selective classification work](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) studies prediction with rejection. I report both accepted-case error and coverage. My simple development-selected softmax threshold does not reproduce their risk guarantee. The development pilot demonstrates why confidence alone is insufficient under sentence-structure shift: every novel structure was mistranslated despite formal arithmetic checks succeeding.

The original handbook's purpose and persistence sections call for missing-evidence diagnosis, skill acquisition, revised predictive models, independent verification and a versioned failure archive. Its general-learner section (around lines 844–891 in the preserved editable copy) explicitly separates competence estimation, curriculum, learning, verification and consolidation. Recipe 5 around line 969 calls for a typed computation mechanism with independent checks. Recipe 6 around line 975 asks for learned curriculum comparisons. The present route addresses a small part of the first requirement; its curriculum remains supplied and does not satisfy Recipe 6's learned-policy target.

The language network predicts operation order and three coefficients. An algebraic candidate is checked by exhausting the eleven residues. That check cannot establish that the predicted statement matches the user's intention. I score exact translation separately and withhold unsupported wording. The relevant probability is model confidence conditional on a narrow teaching distribution, not a physical quantum measurement or a general semantic guarantee.

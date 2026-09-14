# Primary-source research notes

I checked these primary sources during the initial build in September 2026. I use them as mathematical and engineering precedents, and distinguish their published scores from my SERA measurements. I retain the reference package's other citations as leads rather than represent them as independently verified in this build.

| Primary source | What supports this build | What does not follow |
|---|---|---|
| [Yang, Kautz and Hatamizadeh, Gated Delta Networks](https://arxiv.org/abs/2412.06464) | Adaptive forgetting and corrective associative writes are complementary mechanisms | SERA's small reimplementation inherits neither their optimized kernels nor their benchmark performance |
| [Arjovsky, Shah and Bengio, Unitary Evolution Recurrent Neural Networks](https://arxiv.org/abs/1511.06464) | Structured complex recurrence motivates norm-preserving propagation | SERA adds drive and forgetting; this does not make the full update unitary or guarantee better memory |
| [Hafner et al., Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2) | Learn action consequences and use imagined outcomes to guide behavior | The current finite MLP/BFS experiment is not a reproduction of Dreamer or evidence of pixel-based control |
| [Ellis et al., DreamCoder](https://arxiv.org/abs/2006.08381) | Reusable executable abstractions can expand a problem-solving library | SERA currently identifies transition tables; it does not learn a new programming language or reproduce wake-sleep library induction |
| [Watrous / IBM, Quantum channels](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/quantum-channels/introduction) | Quantum channels model permitted state transformations, including noise | A mathematically valid state transformation is not an intelligence objective |
| [Watrous / IBM, General measurements](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/general-measurements/introduction) | Measurement extracts classical information and changes state; general measurements extend projective measurements | An event probability is not verification of an answer, and conditioning cannot force a physical measurement outcome |

## Engineering conclusions

I draw the following engineering conclusions from these sources and the architecture handbook:

1. Treat associative memory, complex dynamics and exact programs as separable mechanisms with their own controls.
2. Compare the hybrid to a similar-parameter classical control before attributing improvement to phase or density structure.
3. Make action dependence explicit in world prediction. Do not silently expose hidden state in a perceptual benchmark.
4. Distinguish acquired reusable procedures from a forward prediction and from durable parameter learning.
5. Use finite exact references for the quantum operations before considering compression or hardware execution.

## Attribution

I wrote a new implementation for SERA without importing external model weights or implementation code. I retain `physics_graph.mmd` and `physics_component_map.json` from the reference package unchanged, with their origin and hashes recorded in the intake manifest. I keep the original source files and extracted reference material in local intake storage outside Git history.

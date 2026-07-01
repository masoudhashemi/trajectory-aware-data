---
layout: page
title: "What Is High-Quality Data in Interactive Agentic Training with GRPO?"
description: "A position piece on trajectory-aware data quality for GRPO-based agentic RL."
---

*This is a position piece. I do not have experiments to back every claim yet. My goal is to argue
for a way of thinking about agentic data, and to propose a set of metrics I think we should be
tracking.*

**TL;DR.**

- Most of the literature defines "high-quality agentic data" at the level of the prompt and the
  final reward: make tasks verifiable, diverse, difficulty-calibrated, and, for GRPO, make sure the
  outcome reward has non-zero variance. These properties are necessary, but I do not think they are
  the whole story for *interactive* agentic training.
- When we train multi-turn agents with GRPO, every prompt already comes with a group of rollouts
  that we generate for advantage estimation. That group is a rich source of information about both
  the model and the datum, and we mostly ignore it. So I argue that high-quality interactive data
  should be defined (and selected, and generated) not only by outcome-reward variance, but by the
  diversity and overlap structure of the trajectories themselves, because that structure is what
  decides whether a value-free algorithm like GRPO can actually assign credit.
- Data and the environment should be co-designed, and specifically because GRPO is bad at credit
  assignment: the environment has to pick up the slack. It should be built to hand the trainer the
  structure, sub-goals, and per-step signal that a value-free learner can't figure out on its own.

---

## 1. From "good data" to "good data for interactive agentic GRPO"

There is a fairly standard checklist for good training data: it should be **diverse**,
**verifiable**, **accurate**, and have broad **coverage**. These apply to almost any post-training
setting, agentic or not.

But agentic data should not be thought of in isolation. Agentic data usually arrives with an
**environment**, and I mean the environment as distinct from the harness. The environment defines
the tools, how tool calls are answered, how failures are handled, how each step is processed, and
what feedback is returned. Different harnesses (the prompt, tool format, output parsing, and
scaffolding) can be used on top of the same environment to complete the same task. A single datum is
usually an instruction, a set of available tools, some policies, and a query. The task is only
"completed" by the interaction of the harness and the model with that environment. Because of this
coupling, the data and the environment must be designed together. And this matters most here for a
specific reason I develop in Section 5: because GRPO's credit assignment is weak, the environment is
the natural place to make up the difference, so it should be built to carry as much of that burden as
possible.

This matters even more under reinforcement learning. In **SFT**, once you distill a response from a
strong teacher you have little further control. You train on the trajectory you got. In
**interactive agentic RL**, you step through the environment and can observe how the model is doing
at every step. And because most modern RL post-training is **GRPO**-style
([Shao et al., DeepSeekMath](https://arxiv.org/abs/2402.03300);
[DeepSeek-R1](https://arxiv.org/abs/2501.12948)), sampling multiple rollouts per prompt to compute
group-relative, z-scored advantages, every prompt already produces a whole population of
trajectories. That population tells us a lot about model behavior and about the quality of the
datum, and we throw most of it away.

---

## 2. What the literature calls "high-quality data"

Reading across recent agentic-RL and agentic-data papers, "quality" breaks down into roughly ten
recurring dimensions:

| # | Quality dimension | Representative work |
| --- | --- | --- |
| 1 | **Verifiability** (executable / graded checkers, rubrics) | AReaL-SEA, TMAX, OpenThoughts-Agent, Autodata |
| 2 | **Diversity / coverage / balance** (domains, tools, personas, modalities) | TMAX, AReaL-SEA, OpenThoughts-Agent |
| 3 | **Difficulty calibration** (not too easy, not too hard; avoid bimodal pools) | TMAX, Autodata, OpenThoughts-Agent |
| 4 | **Learnability / reward variance** (pass-rate around 0.5) | DAPO, GRESO, Online Difficulty Filtering, Autodata |
| 5 | **Trajectory / trace quality** (richer multi-turn supervision) | OpenThoughts-Agent, TMAX |
| 6 | **Co-design with environment / verifier / user simulator** | AReaL-SEA, TMAX, SDAR |
| 7 | **Metadata / privileged signal** for credit assignment and reward shaping | SDAR, AReaL-SEA, Autodata |
| 8 | **Generalization** across harness / task | TMAX, OpenThoughts-Agent |
| 9 | **Self-evolution / dynamic generation** | AReaL-SEA, Autodata, TMAX |
| 10 | **Teacher quality** (strongest model is not the best teacher) | OpenThoughts-Agent |

A few of these are worth pulling out, because the rest of this post builds on them.

- **Verifiable, environment-grounded rewards.** [AReaL-SEA](https://arxiv.org/abs/2601.22607)
  synthesizes multi-turn tool-use dialogues together with executable per-instance checkers that
  serve directly as RL rewards. [TMAX](https://arxiv.org/abs/2606.23321) goes beyond exact-match
  with graded verifiers (metric thresholds, adversarial corpora, fuzz-equivalence, multi-protocol),
  which gives you continuous reward knobs.

- **Diversity by construction.** TMAX composes each task from nine structured axes (domain, skills,
  primitive skills, persona, language, task complexity, command complexity, fixture, verifier) and
  even reports a balance score `exp(H)/N` to quantify how uniform the coverage is. AReaL-SEA builds
  diversity at the planning stage with diversified synthesis plans, rather than hoping it emerges.

- **Difficulty calibration.** TMAX explicitly avoids the bimodal "trivial-or-impossible" pool and
  calibrates difficulty so tasks stay challenging throughout training (its agents take more steps on
  harder data as training goes on). [Autodata](https://arxiv.org/abs/2606.25996) frames the goal as
  making questions "just right" for the learner, neither too easy nor too hard.

- **Reward variance is the load-bearing signal for GRPO.** This is the most quantitatively precise
  notion of quality in the literature. Because GRPO's advantage is group-relative, a prompt whose
  rollouts are all successful or all failed gives zero advantage and zero gradient. So
  [DAPO](https://arxiv.org/abs/2503.14476) over-samples and filters out prompts with accuracy 0 or
  1, and difficulty-filtering work shows the learning signal is bounded by the Bernoulli variance
  `p(1-p)`, which is maximized at pass-rate 0.5
  ([Online Difficulty Filtering](https://openreview.net/pdf?id=78u1Rk47yP); GRESO;
  [Prompt Replay](https://arxiv.org/abs/2603.21177)). Autodata makes this an explicit
  data-generation objective: its agentic loop reshapes the per-prompt weak-rollout variance (raising
  weak-solver std from 7.93 to 12.63 on legal tasks) and labels each candidate with a
  `grpo_suitability` verdict.

- **Trace quality.** [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855) finds, across more than
  100 ablations, that keeping longer multi-turn traces (at least 5 turns) is one of the most
  reliable quality filters. Multi-turn supervision matters more than raw token budget.

So the literature has converged on a strong, useful picture. But notice where it operates: almost
entirely at the prompt/dataset level and at the outcome-reward level. Trajectory-internal structure
is used mainly for training mechanics, e.g., [SDAR](https://arxiv.org/abs/2605.15155)'s token-level
distillation gate, not as a definition of data quality. That is the gap I want to talk about.

---

## 3. Necessary, but not sufficient

The standard things we monitor during training (reward mean and std, entropy, KL) are too coarse for
interactive settings.

Entropy is a good detector of *token-level* collapse, i.e., when the policy sharpens to near-greedy
and stops sampling alternatives at all. But it tells us almost nothing about the quality and
diversity of the trajectories generated for a given prompt. It cannot tell us whether the model
explored enough distinct strategies to find a solution, or whether it keeps retrying near-identical
trajectories that succeed (or fail) every time. And when a prompt produces a mix of successes and
failures, entropy does not tell us how different those trajectories are, i.e., whether the model is
genuinely trying different paths or just jittering around one.

In fact, I think interactive training has a second, sneakier failure mode that *aggregate* entropy
hides: **mode collapse** (sometimes called diversity collapse). Here the average token-level entropy
stays high. The model has not collapsed in the usual sense, it is still sampling freely token by
token. Yet it keeps producing the same strategic pattern: the same tool-call sequence, the same plan
skeleton, the same approach it saw most often during training. The trick is *where* the entropy
lives. It is spread across free-text paraphrase (there are many ways to word the same thought),
while at the decision points that actually matter (which tool to call, which sub-goal to pursue) the
distribution is sharp and repetitive. Averaged over all positions, entropy looks healthy; measured
only at the branching decisions, it would not. This is a form of overfitting to the dominant
pattern, and I would argue it is more dangerous than token-level collapse, precisely because the
aggregate metric we usually trust says everything is fine. To catch it you have to measure diversity
over the trajectory *structure* (or at least restrict entropy to the decision points), not over the
pooled token distribution. Note KL does not help here either: it measures drift from the reference
policy, not diversity within the group.

Outcome-reward variance is better, and it is rightly the field's favorite signal. But it is just a
scalar summary of the endpoints. Two prompts can have identical reward variance (say, 4 out of 8
rollouts succeed) and still be completely different data. In one, the four successes are four
genuinely different solution paths and the four failures probe four different dead-ends. In the
other, all eight rollouts are the same trajectory up to the last token, and the split is essentially
noise. For SFT this distinction barely matters. For interactive GRPO without a value model, I think
it is everything.

---

## 4. The interactive advantage: mine the rollout group you already paid for

Here is the central claim. In GRPO we already generate `G` rollouts per prompt for z-scoring. So we
should treat that group as a measurement instrument on the datum, and define quality with metrics
computed over it.

### 4.1 Reward variance (the part the field already uses)

Reward variance remains the right first-order signal. A prompt whose pass-rate sits near 0.5 is the
most signalful for GRPO, because for a binary reward the group's Bernoulli variance `p(1-p)` is
maximized there (at 0.25) ([DAPO](https://arxiv.org/abs/2503.14476);
[Online Difficulty Filtering](https://openreview.net/pdf?id=78u1Rk47yP)). This is the same instinct
behind AReaL-SEA's and DAPO's dynamic filtering of zero-variance groups.

### 4.2 Diversity within the failed trajectories

When all the rollouts fail, reward variance is zero and the prompt gets discarded. But the failures
are not equivalent. Did the model try different tools, different argument orderings, different
sub-goal decompositions? Or did it fail the same way eight times?

I should be precise about *who* consumes this signal, because it is easy to overclaim. The GRPO
update itself cannot use it: an all-fail group has zero advantage and therefore zero gradient no
matter how diverse the failures are. Failure diversity is a signal for the layer *around* the
optimizer, i.e., the curriculum or self-evolving generator (Section 7). A prompt that is failing in
many different ways may contain useful exploration signal, so it is worth perturbing toward the
learnable band; a prompt failing the same way every time is either mis-specified or hopelessly
out of reach, and should be regenerated or dropped. As far as I know, no current pipeline scores data
by the diversity of its failures, even though it is directly computable from the group we already
have.

### 4.3 Diversity and overlap across successes and failures, for value-free credit assignment

This is the part that interacts most deeply with how GRPO works, so it is worth being precise about
the mechanism rather than hand-waving about "contrast". Vanilla GRPO has no value model. It computes
one sequence-level advantage per trajectory, `A_i = (r_i - mean)/std`, and **broadcasts that same
scalar to every token** in the trajectory. There is no operator in the algorithm that inspects two
trajectories and localizes the step where they diverged. So when I say the model "assigns credit," I
mean something weaker and purely statistical, and spelling it out is what makes clear why trajectory
structure matters.

Consider a shared prefix of steps that appears in both a successful rollout (positive advantage) and
a failed one (negative advantage). Those shared tokens receive pushes in opposite directions that
partially cancel in expectation, while the tokens on the *divergent* suffixes receive an uncancelled
net signal. Averaged across the whole group, the gradient therefore concentrates on the steps that
actually differ between good and bad trajectories. Nobody localized anything; it falls out of
averaging signed advantages over overlapping token sequences. But that averaging only does useful
work if the group has the right structure:

- If successful and failed trajectories share overlapping steps and then diverge, the shared part
  cancels and the divergent decisions get net signal. This is the regime where credit implicitly
  lands in roughly the right place.
- If the rollouts are near-identical, there is nothing to contrast and no exploration; the group is
  uninformative even when its outcome variance looks fine.
- If they are wholly disjoint (different tools from the first step, wildly different lengths), there
  is no shared anchor to cancel against, so the broadcast advantage just reinforces or punishes
  entire trajectories wholesale and the per-step signal is noise.

So the ideal interactive datum plus environment produces trajectories that are *bounded-diverse*:
diverse enough to explore alternative solutions, yet overlapping enough that the cancellation above
concentrates gradient on the decisions that matter. This is a property of the joint (data,
environment) object, and it is invisible to any outcome-only metric. It is also where environment
design earns its keep, because the environment controls the branching factor, the step granularity,
and how much state is actually shared across rollouts.

This is not just an intuition; algorithms already exploit exactly this structure.
[GiGPO](https://arxiv.org/abs/2505.10978) builds step-level advantage groups by finding *repeated
environment states across the trajectories in the group* and comparing the actions taken from them,
which is a direct mechanization of "overlap enables per-step credit," and it beats GRPO on ALFWorld
and WebShop with no extra rollouts. In other words, the overlap structure I am arguing we should
*measure and select for* is the same structure such methods consume, so improving the data on this
axis should compound with improvements to the algorithm.

### 4.4 Prompt diversity is not trajectory diversity

A subtle but important corollary: a beautifully diverse prompt set (many personas, tools, argument
shapes, policies) does not guarantee diverse trajectories. And the model trains on trajectories, not
on prompts. TMAX's balance score and AReaL-SEA's diversified plans both measure diversity on the
input side. That is a good start, but the quantity that actually reaches the optimizer is downstream.
Two surface-different prompts can collapse to the same canonical trajectory, while one prompt can fan
out into many. So measuring diversity where it counts means measuring it over the rollout group, not
over the task spec.

---

## 5. Environment and credit assignment (the real reason to co-design)

A datum only means something once an environment executes it, so data and the environment have to be
designed together. Here that matters for a specific reason: GRPO's credit assignment is weak
(Section 4.3), so the environment is the natural place to make up the difference, and it should be
engineered to carry as much of that burden as it can.

There are really two complementary ways to get per-step credit out of a value-free setup:

1. **Implicit, for free from the group (Section 4.3).** Rely on overlap structure so the broadcast
   advantage concentrates on divergent steps. It costs nothing extra, but it is weak and only works
   when the group happens to be bounded-diverse.
2. **Explicit, from the environment (this section).** Have the environment emit denser signal:
   reference sub-goals, intermediate checkpoints, verifier sub-conditions, "you are now N% done"
   progress. [VinePPO](https://arxiv.org/abs/2410.01679) is the clean extreme, it uses the
   environment's resettability to roll out from intermediate states and compute unbiased Monte-Carlo
   per-step values, directly replacing the missing critic. SDAR
   ([Self-Distilled Agentic RL](https://arxiv.org/abs/2605.15155)) is a softer version: a "teacher"
   branch gets privileged training-only context (e.g., retrieved skills) the test-time student never
   sees, and the token-level teacher-student gap is distilled through a learned gate, i.e., dense
   supervision layered on the coarse trajectory reward. (Learned process reward models such as
   [Lightman et al.](https://arxiv.org/abs/2305.20050) are the supervised cousin: label per-step
   correctness and train a critic to emit it.)

The trade-off is cost versus signal. Option 2 needs an instrumented, resettable, metadata-emitting
environment, and for VinePPO-style estimates it needs extra rollouts; option 1 is free but fragile.
My view is that you use option 1 as the default and pay for option 2 exactly where credit assignment
is hardest, i.e., long horizons and sparse terminal rewards. Either way the environment is doing the
work, which is why I treat it as a co-equal design target with the data: it is the only place where
you can either manufacture the overlap-and-structure that Section 4.3 needs, or supply outright the
per-step value that GRPO lacks.

---

## 6. Environment and synthetic data: stop throwing away the metadata

When we synthesize agentic data, we generate a lot of metadata that then gets discarded at training
time.

- AReaL-SEA's data engine emits, per instance, an executable verifier, a synthesis plan, an
  evaluation rubric, and, on failures, an attribution tag (`TASK` vs `TRAJECTORY`) that says whether
  a bad outcome was the task's fault or the agent's.
- TMAX records per-task difficulty buckets, verifier thresholds, persona, and skill axes.
- Autodata's judge produces structured per-round verdicts (`weak_pattern`, `strong_pattern`,
  `gap_interpretation`, `grpo_suitability`) along with concrete suggestions.

All of this is usually thrown away once the `(task, reward)` pair reaches the optimizer. But this
metadata is exactly what you would use to measure trajectory progress and to design better reward
functions: partial credit aligned to known sub-goals, progress shaping aligned to the generator's
difficulty annotations, or per-turn rewards aligned to the verifier's sub-conditions. The generation
pipeline already knows the "answer key" structure, so the trainer should use it.

There is also a hard lesson here about interactive rewards specifically. In dual-control and
tool-using settings, a noisy user simulator corrupts the reward signal. AReaL-SEA shows that an
off-the-shelf user model often mis-executes tools and causes the agent to be penalized for correct
behavior, and that fine-tuning the user simulator first moved Telecom RL from a 20-point regression
to a gain. So "high-quality data" in interactive settings also includes a high-quality simulator.
Again, the environment and the data are really one object.

---

## 7. From rejection to generation

DAPO's dynamic sampling and AReaL-SEA's dynamic filtering both reject low-signal prompts:
over-sample, throw away the groups with accuracy 0 or 1, and keep refilling the batch until it is
full of high-variance prompts (and [Hard Examples Are All You Need](https://arxiv.org/abs/2508.14094)
pushes the same selection logic further under a fixed annotation budget). This works, but it spends
compute generating rollouts you then discard, and the all-zero / all-one rate tends to drift as the
policy improves (DAPO documents the effective batch shrinking over training).

If we have access to self-evolving environments, and the number of them is growing fast (AReaL-SEA's
reflection loop, Autodata's meta-optimized data scientist, TMAX's compositional generator), then I
think the better move is not to reject pre-generated data but to generate the right data. Autodata
already demonstrates the principle offline: its agentic loop pushes each prompt toward the learnable
band (widening the weak/strong gap when questions are too easy, narrowing it when they are too hard),
so that "the key is not to make the question more challenging, but to make it just right." The
natural next step is to do this online, during training. As the policy improves and prompts saturate
(drift toward all-success), have the environment regenerate or perturb them to keep the rollout group
in the high-variance, bounded-diverse regime, and to keep the trajectory structure (Section 4)
signalful, not just the outcome. That converts the compute DAPO spends on rejection into compute
spent on producing better data.

I do not want to oversell this. Continuously regenerating prompts to hold the group near the
learnable band turns the training distribution into a moving target that co-adapts with the policy,
and that carries real risks: non-stationarity can destabilize optimization; always deleting whatever
the policy has just mastered can keep skills from ever consolidating (a curriculum that is
permanently hard is not obviously optimal); and a generator optimized against the current policy is a
tempting surface for reward hacking. Autodata validated the "just right" reshaping *offline*, against
a fixed solver. Doing it *online*, in the loop with the policy it is shaping, is the part that is
still a conjecture, and it would need the usual guardrails (replay of consolidated tasks, slow
generator updates, held-out evaluation) before I would trust it.

---

## 8. The case for RL gyms

Everything above points to the same infrastructure need: RL environments (gyms) that treat the
environment as a programmable, instrumented, evolvable object rather than a fixed dataset. A good gym
should:

- **Expose metadata** (sub-goals, difficulty, verifier sub-conditions, attribution) so the trainer
  can shape rewards and measure progress (Sections 5 and 6).
- **Support harness diversity** within the same task. TMAX and OpenThoughts-Agent both show RL
  training that generalizes across harnesses, so we want to be able to vary the harness while holding
  the environment fixed, and confirm we are teaching skills, not "harness-fitting."
- **Generate data dynamically**, to keep prompts in the signalful band online instead of rejecting
  them (Section 7).
- **Enable ablation**, so we can actually compare data, environment, and algorithm choices in a
  controlled way. OpenThoughts-Agent's 100+ ablations are the model to emulate, but they were SFT-
  and source-level, and we need the same rigor at the trajectory-structure level.

---

## 9. A working definition and a proposed metric suite

Here is the definition I would propose. **A high-quality datum for interactive agentic GRPO** is one
that, together with its environment, induces a rollout group that is **verifiable**, **signalful at
the outcome level**, and **bounded-diverse at the trajectory level** (diverse enough to explore
alternative solutions, yet overlapping enough that a value-free learner can implicitly assign
credit), and that ships the **metadata** needed to shape dense rewards.

"Bounded-diverse" is a region in a 2-D plane: within-outcome diversity (metrics 2-3 below) on one
axis, cross-outcome overlap (metric 4) on the other. Low diversity with high overlap gives
near-identical rollouts with nothing to learn; high diversity with low overlap gives disjoint paths
with no shared anchor for the cancellation argument in Section 4.3. The signalful region is in
between: enough overlap to cancel shared steps, enough diversity that the divergent steps actually
differ. Quality here is a region in two measured quantities, not a single scalar.

These are proposals, not validated metrics yet, but they are all computable from artifacts we
already produce. Alongside the usual reward-mean / reward-std / entropy / KL, I would log, per
prompt:

1. **Outcome reward variance** (the DAPO/GRESO band; target pass-rate ≈ 0.5, i.e. Bernoulli variance
   ≈ 0.25 for binary rewards).
2. **Failure diversity**: pairwise dissimilarity among failed trajectories. A simple instantiation
   is normalized edit distance over the action/tool-call sequence, or Jaccard over the set of tools
   used, or sub-goal coverage.
3. **Success diversity**: the same dissimilarity, computed over successful trajectories.
4. **Cross-outcome overlap**: shared-prefix or shared-step overlap between successful and failed
   trajectories. This is the credit-assignment substrate of Section 4.3.
5. **Step-count dispersion**: the spread of trajectory lengths within the group.
6. **Prompt-to-trajectory diversity gap**: how much input diversity actually survives into trajectory
   diversity (Section 4.4). This one also doubles as a mode-collapse detector.
7. **Metadata-derived progress**: per-turn progress against generator sub-goals, where available.

Three caveats I do not want to hand-wave. First, cost and representation: metrics 2-4 are `O(G^2)`
pairwise comparisons per prompt, and they force a choice of trajectory representation, i.e. how to
align variable-length traces, and whether two tool calls with reordered arguments or the same effect
count as "the same step." That representation choice is where most of the real work lives. Second,
redundancy: these seven are correlated (4 partly determines 2-3; 6 overlaps both), so I would not log
all of them forever. If I could keep only two, I would keep **cross-outcome overlap (4)** and
**outcome variance (1)**, since together they locate a prompt in the region above. Third, selection
pressure: if you *select* data to maximize cross-outcome overlap, you bias toward tasks whose correct
path is structurally close to the failures (easy-to-contrast tasks) and away from problems where the
solution is structurally distant from anything the model currently does. Optimized naively, the suite
could raise learnability while quietly lowering the capability ceiling; the metrics are meant to
*inform* curriculum and generation, not to be maximized blindly.

The point of logging these is to turn "high-quality data" from a static dataset property into a
joint, measurable, and optimizable property of data, environment, and policy. It also makes the
position *falsifiable*: if you controlled for outcome-reward variance and found that these
trajectory-structure metrics carried no additional predictive power for downstream gain, the central
claim of this post would be wrong.

High-quality data for interactive agentic GRPO is not a property of prompts alone; it is a property
of the trajectory group a datum and its environment produce together. The right unit of design is the
whole system: data, environment, and training algorithm.

---

## References

- **SDAR**: Lu et al. *Self-Distilled Agentic Reinforcement Learning.*
  [arXiv:2605.15155](https://arxiv.org/abs/2605.15155)
- **TMAX**: Ivison, Yin et al. *TMAX: A Simple Recipe for Terminal Agents.*
  [arXiv:2606.23321](https://arxiv.org/abs/2606.23321)
- **AReaL-SEA**: Gao, Chen et al. *From Self-Evolving Synthetic Data to Verifiable-Reward RL:
  Post-Training Multi-turn Interactive Tool-Using Agents.*
  [arXiv:2601.22607](https://arxiv.org/abs/2601.22607)
- **OpenThoughts-Agent**: Raoof, Zhuang, Nezhurina, Guha et al. *Data Recipes for Agentic Models.*
  [arXiv:2606.24855](https://arxiv.org/abs/2606.24855)
- **Autodata / Agentic Self-Instruct**: Kulikov, Whitehouse, Wu, Nie et al. *Autodata: An Agentic
  Data Scientist to Create High-Quality Synthetic Data.*
  [arXiv:2606.25996](https://arxiv.org/abs/2606.25996)
- **DAPO**: Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.*
  [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- **VinePPO**: Kazemnejad et al. *VinePPO: Refining Credit Assignment in RL Training of LLMs.*
  [arXiv:2410.01679](https://arxiv.org/abs/2410.01679)
- **GiGPO**: Feng et al. *Group-in-Group Policy Optimization for LLM Agent Training.*
  [arXiv:2505.10978](https://arxiv.org/abs/2505.10978)
- **Process Reward Models**: Lightman et al. *Let's Verify Step by Step.*
  [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- **GRPO / DeepSeekMath**: Shao et al. *DeepSeekMath.*
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **DeepSeek-R1**: Guo et al. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **Online Difficulty Filtering for Reasoning-Oriented RL**:
  [OpenReview](https://openreview.net/pdf?id=78u1Rk47yP)
- **Prompt Replay**: *Speeding up GRPO with on-policy reuse of high-signal prompts.*
  [arXiv:2603.21177](https://arxiv.org/abs/2603.21177)
- **Hard Examples Are All You Need**: *Maximizing GRPO Post-Training Under Annotation Budgets.*
  [arXiv:2508.14094](https://arxiv.org/abs/2508.14094)

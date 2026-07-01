# trajectory-aware-data

[What Is High-Quality Data in Interactive Agentic Training with GRPO?](https://masoudhashemi.github.io/trajectory-aware-data/)

The post argues that, for interactive multi-turn agents trained with GRPO, data quality should not
only be defined at the prompt/outcome level (verifiability, diversity, difficulty, reward variance).
It should also be defined by the **diversity and overlap structure of the rollout trajectories** that
GRPO already generates for advantage estimation, because that structure shapes credit assignment
without a value model.

The broader point is that data, environment, and training algorithm should be designed together.

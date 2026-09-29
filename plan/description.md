# Ensemble of Agents

**Mentor:** Yoav Freund · [yfreund@ucsd.edu](mailto:yfreund@ucsd.edu)  
**Domain:** D31 · 4 seats · 1 hour per week

## Overview

This project combines two subjects: ensemble methods and agent-based AI.

Boosting [[1]](#references), bagging [[2]](#references), and random forests [[3]](#references) are popular machine learning methods that combine the output of weaker learning algorithms to create a single highly accurate rule (or model). Agent-based AI is all the rage these days: it refers to having independent LLM sessions that run in parallel and communicate with each other.

The idea is to merge these two directions to construct boosting and bagging systems where a swarm of agents implements the basic learners and a coordinator agent coordinates the swarm to create a combined accurate and reliable classifier. Some related work appears in [[4]](#references).

## References

1. Freund, Yoav, and Robert E. Schapire. "A decision-theoretic generalization of on-line learning and an application to boosting." *Journal of Computer and System Sciences* 55.1 (1997): 119–139.
2. Breiman, Leo. "Bagging predictors." *Machine Learning* 24.2 (1996): 123–140.
3. Breiman, Leo. "Random forests." *Machine Learning* 45.1 (2001): 5–32.
4. Alafate, Julaiti, and Yoav S. Freund. "Faster boosting with smaller memory." *Advances in Neural Information Processing Systems* 32 (2019).

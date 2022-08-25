# Classical Q-Learning

Q-Learning (Quality Learning) is a "model-free" reinforcement learning algorithm that utilizes a recurring update of a table of states and actions to learn how to accomplish a task.

- How is this an extension of what we've learned so far?

What we've learned so far:

- General Concepts:
  
  - Environments, Agents, Rewards, Observations, States and Actions.

- Manual Agents:
  
  - Simply use observations to complete a task.

What is the next step? So far we haven't utilized the reward returned by an observation from an environment, because that Q-Learning will be our first algorithm to truly learn through reinforcement.

What comes after Q-Learning?

- Once we understand Q-Learning, we can apply deep learning methods to create DQN (Deep Q-Learning).

- Note how Reinforcement Learning doesn't necessarily require Neural Networks, it simply requires an agent to learn based off a reward metric.

Sections:

- History of Classical Q-Learning

- Theory and Intuition Lectures

- Programming Q-Learning

- Q Learning Exercise Project

- Solution Overview

## History of Q-Learning

- 1940s-1960s: Jean Piaget was a Swiss psychologist working on studying child development.

- Conducted small experiments designed to understand how children learn new skills.

- Piaget's experiments were not very robust, with small sample sizes that were not randomly selected.

- However, his key insight on general importance of childhood education are key to the study of development.

- 1980s: Chris Watkins was a graduate student at Cambridge University.

- Performed Piaget style learning experiments with children from a local primary school.

- Watkins was interested in how children had great flexibility in learning.

- After a few attempts children generally improved their problem solving.

- Watkins later also started reading research in animal learning.

- Watkins applied his general findings to begin thinking of reinforcement learning as a form of incremental dynamic programming.

- 1989: Phd Thesis "Learning from Delayed Rewards" later known as Q-Learning.

- 1992: Peter Dayan and Chris Watkins publish: "Q-learning" in ML.
  
  - The paper actually proves that Q-Learning converges to the optium action-values with probability 1 so long as call actions are repeatedly sampled in all states and the action-values are represented discretely.

> **Key Idea**: If an **environment** has a **discrete state space** and a **discrete set of actions**, it is possible to create a **grid or table** that **matches all possible actions** at all possible states.

What's missing to begin learning?

Recall our Reinforcement Learning agent wants to maximize the reward it receives from the environment.

Lets create a function Q: 

$ Q(s,a)=E[r] $

Q-Learning Table:

|          | Action 0 | Action 1 | Action 2 | Action 3 |
| -------- | -------- | -------- | -------- | -------- |
| State 0  | Q(s,a)   | Q(s,a)   | Q(s,a)   | Q(s,a)   |
| State 1  | Q(s,a)   | Q(s,a)   | Q(s,a)   | Q(s,a)   |
| ...      | ...      | ...      | ...      | ...      |
| State 15 | Q(s,a)   | Q(s,a)   | Q(s,a)   | Q(s,a)   |

> If we can map all possible actions and states as a table grid, we could assign an expected reward to each possible action at each possible state: $Q(s,a)$

> Now we just need to figure out how to "learn" or update our Q function.

> After some training process, we could then simply look up the "best" action given a space, that is the Q(s,a) with the highest expected reward.

> Recall Watkins and Dayan proved this actually converged for discrete action and state space.

If the convergence for discrete action and state spaces has been proven, then what's the catch? Notice how the Q-Learning table required a discrete action and space state.

Why not just apply Q-Learning tables to all problems in the real world and say AI is solved? Notice how even for a simple 4x4 grid with only 4 actions, the table already has 64 functions of Q(s,a) to solve for.

Key limitations:

- Be careful not to confuse environment variables (such as number of grid squares) with the number of potential states.

- State is any possible discrete iteration of the environment and agent.

In conclusion, this means Q-Learning can't be applied to all problems, even if in theory we have a discrete action and space state.

Later on we will see how we could apply Q-Learning to a continuous action or state space by creating discrete bins of potential actions and spaces.

## Q-Target Equation

Let's explore the methodology for updating $Q(s,a)$ through a training process.

$Q(s,a) = E[r] $

Remember that our final goal is to create the Q-Learning table for the environment.

We know that we will be performing actions through time steps (t, t+1, t+2,...) so let's define Q in terms of expected sum of all future rewards.

**Note: we are going to logically "build up" to the final correct equation for Q.**

Thinking of Q as expected sum of future rewards...

$Q(s_t, a_t) = r_{t+1} + r_{t+2} + r_{t+3} + ...$

Let's add an adjustment term, to show that we're less certain of future rewards:

$ Q(s_t, a_t) = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... $

This adjustment will be labeled as gamma and we can call it the discount rate.

We can then rewrite this as:

$ Q(s_t, a_t) = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) $

Lets adjust this term to reflect our goal using an action a to obtain the max Q at State t+1.

$Q(s_t, a_t) = r_{t+1} + \gamma \; \max_a \; Q(S_{t+1}, a)$

This is our final Q-target. However, it does not or learn to find the optimal Q-target

## Q-Update Equation

Now that we have derived our equation for target Q, let's explore the methodology to actually update this Q value.

- Learning rate (alpha) is a key hyperparameter that can be tuned.

- An error is simply the difference between your target Q and your current Q.

We'll also discuss the pragmatic elements of **exploration vs. exploitation** through the use of an **epsilon** parameter.

**Our Q-Learning Equation:**

$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1}+\gamma \; \max_a \; Q(S_{t+1}, a) - Q(S_t,A_t)] $

**Our Q-Learning Table:**

|          | Action 0 | Action 1 | Action 2 | Action 3 |
| -------- | -------- | -------- | -------- | -------- |
| State 0  | Q(s,a)   | Q(s,a)   | Q(s,a)   | Q(s,a)   |
| State 1  | Q(s,a)   | Q(s,a)   | Q(s,a)   | Q(s,a)   |
| ...      | ...      | ...      | ...      | ...      |
| State 15 | Q(s,a)   | Q(s,a)   | Q(s,a)   | Q(s,a)   |

> Quick Side Note!
> 
> - FrozenLake in OpenAI gym only rewards the agent once they win the game.
> 
> - This means that at first the agent is essentially randomly guessing actions until they win the game.
> 
> - It can take 100s of episodes to win with random choices.

### Exploitation versus Exploration

This issue arises when trying to balance **exploring** new potential actions, versus just **exploiting** the actions that have worked before.

- Exploring means still selecting random actions in order for the agent to hopefully discover new or better ways to obtain a reward.

- Exploitation means abusing the actions known to work over and over again, which may cause the agent to only learn how to achieve a reward in a single fashion.

- Exploitation could lead to getting a reward but with poor performance.
  
  - For example, in the FrozenLake going back and forth repeatedly between two frozen squares before heading to the final goal.

- Exploitation becomes especially bad for more complex environments, where the actions that worked in one situation, may change in a future scenario.
  
  - This is where we can define an **epsilon** value to decide whether we use the max Q value available, or if we still choose a random action.
  
  - We also define a **decay** mechanism, to slowly reduce random exploration as training continues and we are closer to discovering the optimal policy.
  
  - Decay of epsilon is another hyperparameter we can tune.
  
  - Slower decay means slower learning, since more random actions will be taken.

The greedy epsilon choice and decay mechanism will be a separated component from the Q equation, if decides whether to perform the action associated with the max Q value or a random action.

It will help the algorithm to continue to learn, instead of stopping the very first time it gets a reward, otherwise it won't have enough time to optimize.

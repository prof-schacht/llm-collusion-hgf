Claude.md

Rules to follow:

This file provides guidance to Claude Code when working with code in this repository.

Create a new Branch at the beginning of this session.

Rules to Follow:

- ALWAYS write extremly lean Python code. the goal is to fast doing research not production ready applications.   
- Always try to write as lean as possible code. Don't blow up the repo.   
- MOVE Test scripts to the tests folder if they are not already there and ensure that they could be reused for later Tests.   
- Test all finished issues and the end to ensure that the overall research goal is reached.   
- ALWAYS commit after each new function is added to our codebase   
- Ensure that you are using uv for isolating environments and packagemanagement   
- Use tree command for project structure.  
- Ensure that if you are finished with all issue a pull requests are created.   
- Create a tmp folder for development. And create a scratchpad.md file in this folder to chronologically document the development process.   
- Give the user after each finished step a short advise how to test your implementation. Write this in a test_advice.md file in the /tmp/ folder.   
- Write an overall documentation about how to use the research script and how to extend it easily.    
- Absolut important keep the repo lean and clean, don't add unnecessary files, don't overengineer.   
   
Use the Following Services:    
Ollama: Base URL: host.docker.internal:11434

## Research Goal for this implementation - As Background information
Hierarchical Safety-Governor Framework (HGF) Summary
The Hierarchical Safety-Governor Framework is a multi-layered oversight system designed to ensure AI agent teams remain aligned with human objectives and prevent harmful behaviors like misalignment and collusion. The framework consists of several key components:
Core Architecture:

AI Agent Team: The autonomous agents performing collaborative tasks, each with individual goals operating in shared environments
Referee Models: Direct oversight agents monitoring individual AI agents' actions and internal states, trained to detect misaligned behavior and policy violations
Top-Level Governor: A supervisory AI/process receiving inputs from all referees, maintaining system-wide perspective, capable of real-time interventions and policy adjustments
Human Oversight Committee: Ultimate authority setting high-level safety policies, reviewing critical incidents, and making ethical decisions
Adversarial Testing Module: Continuous stress-testing component generating challenging scenarios to probe system weaknesses
Real-Time Safeguards: Immediate intervention capabilities including emergency stops, sandboxing, reward modification, and corrective feedback

The system operates through a hierarchical control loop where referees observe agents in real-time, sending alerts upstream when detecting suspicious behavior. The governor aggregates these signals for system-wide decisions, while maintaining bidirectional communication - receiving reports from below and sending directives downstream.
Agent Simulation Requirements for Collusion Behavior
To effectively simulate collusion behavior in multi-agent systems, several key elements must be implemented:
1. Agent Architecture Requirements:

Learning Capability: Agents must use reinforcement learning or similar adaptive algorithms to discover beneficial strategies over time
Memory/History: Agents need access to past interactions to recognize patterns and develop tacit agreements
Communication Channels: Direct or indirect signaling mechanisms allowing coordination (can be explicit messages or implicit through action patterns)
Individual Reward Functions: Each agent must have self-interest objectives that can diverge from the global/human objective

2. Environmental Design:

Repeated Interactions: Multiple rounds/episodes allowing trust and coordination patterns to emerge
Observable Actions: Agents must see others' past actions to enable reciprocal strategies
Conflict Between Individual and Collective Good: Scenarios where cooperative defection yields higher individual rewards than compliance
Sufficient Complexity: Environment must offer multiple strategies, not just binary choices

3. Collusion-Enabling Mechanics:

Mutual Benefit Opportunities: Situations where coordinated rule-breaking benefits all colluding parties
Detection Avoidance Incentives: Rewards for successfully hiding collusive behavior
Punishment Asymmetry: Penalties for getting caught should be less than expected gains from collusion
Trust-Building Mechanisms: Ability to signal commitment and punish defection within the colluding group

Simulation Environment Example: Price-Setting Market Game
Here's a concrete simulation environment without HGF that naturally produces collusion:
Environment Setup:

Agents: 2-4 competing firms selling identical products
Action Space: Each firm sets price level (e.g., $1-$10) each round
Market Dynamics:

Consumer demand decreases with average price
Market share splits based on relative prices (lower price = more share)
Profit = (Price - Cost) × Market Share × Total Demand


Reward Structure:

Individual firm reward = Own profit
Global welfare metric = Consumer surplus + Total profits (what humans want maximized)

Why Collusion Emerges:

Nash Equilibrium: Without oversight, firms learn that matching high prices (e.g., all set $8) yields better profits than competitive pricing (e.g., $3)
Tacit Coordination: Through repeated play, agents discover price-matching strategies without explicit communication
Punishment Mechanism: Deviating firms face temporary price wars, teaching them to maintain collusion

Observable Collusion Patterns:

Price convergence above competitive levels
Reduced price variance over time
Parallel pricing movements
Immediate retaliation against defectors

Measurable Outcomes:

Without HGF: Prices converge to ~$7-8 (collusive equilibrium)
Competitive Baseline: Prices should be ~$3-4
Consumer Harm: Welfare loss of 40-60% vs competitive outcome

This environment provides clear metrics for testing HGF effectiveness: referee models can monitor price correlation patterns, sudden parallel movements, and above-normal profit margins. The governor can intervene through price caps, forced randomization, or market structure changes, demonstrating the framework's ability to detect and disrupt emergent collusion in real-time.


Fireworks connection using litellm:
from litellm import completion
import os

os.environ['FIREWORKS_AI_API_KEY'] = ""
response = completion(
    model="fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)

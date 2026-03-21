
Overview
NM i AI 2026 is Norway's national AI championship. Teams compete across three independent AI challenges over a 69-hour window. All scoring is automated and updated in real-time throughout the competition. The leaderboard at the deadline determines the preliminary results, subject to code review and verification by the organizers before official rankings are published.

2
Dates & Schedule
Kickoff & start	Thursday March 19, 2026 at 18:00 CET
Deadline	Sunday March 22, 2026 at 15:00 CET
Winners announced	Sunday March 22, 2026 at ~17:00 CET
The kickoff event will be streamed nationwide. Physical attendance is not required — the competition is fully virtual. All submissions, evaluations, and scoring take place through the platform.

No submissions made after the deadline will be evaluated or counted toward the final results. Submissions that are in-flight (queued or processing) at the deadline will be completed and scored normally.

3
Eligibility
Participation is open to all registered teams. There are no nationality or residency restrictions for competing. However, to be eligible for prizes, teams must meet the verification requirements described in Section 6.

All participants must be at least 15 years of age at the time of registration.

4
Teams
Teams consist of 1 to 4 members. Each person may only be a member of one team. Teams can be altered throughout the event as long as they have zero submissions in any of the three main tasks. Once a team makes their first submission in any main task, the roster is locked — members cannot be added or removed after that point.

Teams are responsible for their own infrastructure, compute resources, and coordination. The organizers do not provide hosting or development environments beyond what is available through the platform.

5
Prizes
1,000,000 NOK
Placement	Prize
1st place	400,000 NOK
2nd place	300,000 NOK
3rd place	200,000 NOK
Best U23 teamU23	100,000 NOK
The U23 prize is awarded to the highest-ranking team where all members are under 23 years of age at the competition end date (March 22, 2026). The U23 prize is combinable with placement prizes — a U23 team that places in the top 3 receives both prizes.

Prize money is split equally among team members by default. Teams may arrange a different distribution by unanimous written agreement submitted to the organizers before payout.

In case of a tie, the team that achieved their score first (by timestamp of the submission that produced the tying score) wins the higher placement.

Prize payment will be made by bank transfer after the results are finalized. The organizers will contact each winning team's captain to arrange payment details. Winners are responsible for any applicable tax obligations under Norwegian law. The organizers do not withhold or pay taxes on behalf of recipients.

6
Prize Eligibility
To be eligible for prizes, a team must satisfy both of the following:

1.
Identity verification

All team members must complete Vipps verification before the competition deadline. Vipps is linked to Norwegian BankID and confirms the participant's legal identity. Verified teams are indicated on the leaderboard.

2.
Code submission

The team must make their code repository public and submit the URL through the platform before the competition deadline (March 22, 2026 at 15:00 CET). See Section 9 for details.

Teams that do not meet these requirements will have their scores displayed on the leaderboard and retain their ranking, but will not be eligible for prize payouts. The next eligible team moves up for prize purposes.

Verification can be completed at any time before the deadline. Teams are strongly encouraged to verify early — verified teams benefit from higher submission rate limits, confirmed Google account eligibility, and avoid last-minute issues that could jeopardize prize eligibility.

7
Tasks
The competition consists of three independent AI challenges. Each task has its own submission format, scoring methodology, and rate limits. Task-specific rules will be published at the kickoff on March 19, 2026 at 18:00 CET.

Task	Sponsor	Type
Task 1	Tripletex	AI agent (API endpoint)
Task 2	Astar	Prediction engine (API)
Task 3	NorgesGruppen Data	Object detection (code upload)
Teams are not required to participate in all tasks but will only accumulate points for tasks they submit to.

8
Overall Scoring & Ranking
The overall leaderboard combines scores from all three tasks using the following methodology:

1
Each task's scores are normalized to a 0–100 scale by dividing by the highest score in that task across all teams.

2
The overall score is the average of the three normalized task scores (equal weight: 33.33% per task).

3
Teams are ranked by overall score in descending order.

Tasks where a team has not submitted receive a normalized score of 0. Competing in all three tasks is strongly advantageous.

The leaderboard updates in real-time. The leaderboard snapshot at the deadline (March 22, 2026 at 15:00 CET) determines preliminary rankings. Official results will be published only after code review and verification by the organizers.

Task-Specific Rules

Detailed rules for each task — including scoring methodology, submission format, and rate limits — will be published at the kickoff on March 19, 2026 at 18:00 CET.

General Provisions
Applicable to all tasks and participants.

9
Code Submission & Verification
To be eligible for prizes, teams must make their code repository public and submit the repository URL through the platform before the competition deadline (March 22, 2026 at 15:00 CET). This is a pre-condition for prize eligibility — not a post-competition requirement — and should be set up well in advance.

The repository must contain the source code used to produce the team's submissions across all tasks they participated in. This includes inference code, model training scripts where applicable, and any custom tooling developed for the competition. The code must be sufficient to demonstrate that the work is original and was produced by the team.

Teams may use any hosting platform (GitHub, GitLab, Bitbucket, etc.). The repository must be publicly accessible. A component on the platform will be provided for submitting your repository URL.

The organizers will review submitted code to verify:

•
The solution reflects genuine AI/ML work produced by the team
•
No evidence of code sharing or collusion with other teams
•
No hardcoded or pre-computed responses designed to game specific test cases
Teams whose code does not pass verification, or who fail to provide a public repository URL before the deadline, are not eligible for prizes. Their scores remain on the leaderboard and the next eligible team moves up for prize purposes.

10
Fair Play & Prohibited Conduct
The purpose of this competition is to build effective AI solutions through genuine technical skill. The use of AI coding assistants (ChatGPT, Claude, Copilot, etc.), publicly available models, datasets, research papers, and open-source libraries is explicitly permitted and encouraged.

The following actions are strictly prohibited:

10.1 Collusion & Solution Sharing
•
Sharing code, model weights, trained models, or task-specific solutions between teams — whether directly, through intermediaries, or through any public or private channel
•
Sharing competition-specific observations that provide a competitive advantage between teams
•
Coordinating submissions, strategies, or division of labor between teams
Discussing general AI/ML techniques, publicly available research, or topics not derived from participation in this competition is permitted. The line is drawn at sharing information that provides a competitive advantage specific to this competition's tasks.

10.2 Identity & Account Manipulation
•
Participating on more than one team, directly or through proxies
•
Creating additional accounts or teams to gain extra submissions, queries, or game attempts
•
Transferring, selling, or sharing team credentials or platform access
•
Using Vipps verification belonging to a person not genuinely participating on the team
10.3 Platform Abuse
•
Circumventing rate limits, cooldowns, or submission quotas
•
Attacking, probing, or degrading platform infrastructure, evaluation systems, or other teams’ endpoints
•
Attempting to extract test data, ground truth, hidden parameters, or evaluation logic from the platform
•
Automated scraping, monitoring, or analysis of other teams’ activity
10.4 Score Manipulation
•
Submitting hardcoded or pre-computed responses that do not reflect genuine model capabilities
•
Engineering submissions designed to manipulate scoring normalization rather than to maximize task performance
•
Any form of score falsification or result tampering
If you are uncertain whether a technique or approach is permitted, contact the organizers before using it. Acting first and asking later is not a defense.

11
Monitoring & Enforcement
The organizers actively monitor the platform, official communication channels, submission patterns, and code repositories throughout the competition:

•
Automated code similarity analysis across teams’ repositories
•
Submission pattern analysis (timing, frequency, score progression)
•
Moderation of the official Slack workspace and all platform communications
•
Cross-referencing API call logs, query patterns, and gameplay data
Violations are handled at the sole discretion of the jury. Consequences include:

Warning

Minor or first-time violations

Prize ineligibility

Retains leaderboard position

Score removal

Affected scores removed

Platform ban

Immediate removal

Where code similarity or collusion is suspected, the organizers may require involved teams to demonstrate independent work. Failure to do so may result in consequences for all involved teams.

12
Platform Availability
The organizers will make all reasonable efforts to maintain platform availability throughout the competition. The organizers accept no liability for downtime, latency, or technical failures caused by infrastructure, third-party services, or circumstances beyond their control.

No deadline extensions, score adjustments, or compensatory submissions will be granted due to platform unavailability, except at the sole discretion of the jury in cases of prolonged, verified, and widespread outages.

13
Intellectual Property
All code submitted for prize eligibility must be open-sourced under the MIT license (or an equivalent permissive license) in a public repository. By participating and submitting code, teams agree to make their work freely available. This is a requirement for prize eligibility.

By participating, teams grant the organizers a non-exclusive, royalty-free right to reference their participation, team name, and results in public communications.

14
Code of Conduct
Participants are expected to engage respectfully with other teams and organizers across all channels, including the official Slack workspace. Harassment, hate speech, threats, or disruptive behavior will not be tolerated and may result in immediate removal from the competition.

15
Data & Privacy
Submissions, game replays, and associated metadata may be analyzed for quality assurance and anti-cheating purposes. Personal data is handled in accordance with Norwegian data protection regulations (GDPR). Vipps verification data is used solely for identity confirmation and prize eligibility and is not shared with third parties.

Game replays are stored and may be publicly viewable.

16
Jury
The jury consists of representatives from Astar. The jury is responsible for interpreting these rules, resolving disputes, and making all final decisions regarding scores, eligibility, disqualifications, and prize distribution. All jury decisions are final and binding.

Given the compressed timeline of the competition (69 hours), there is no formal appeals process. Teams may raise concerns with the organizers at any time, but the jury reserves the right to make immediate, binding decisions as circumstances require.

17
Communication
The official Slack workspace is the primary communication channel for announcements, rule clarifications, and support. Teams are responsible for monitoring Slack for updates. Critical announcements will also be sent via email to team captains.

The organizers moderate all official channels. Messages that share competition-specific solutions, violate the code of conduct, or undermine fair competition will be removed without notice.

18
Amendments
These rules may be updated at any time before or during the competition. Material changes will be communicated through the official Slack workspace and platform announcements. Continued participation after a rule change constitutes acceptance of the updated rules.# Working Rules

## Documentation

- Every meaningful change must leave the repo more understandable than before.
- Update root context files when strategy, progress, or uncertainty changes.
- Keep durable knowledge in docs, not hidden in chat history.

## Code Comments

- Comment non-obvious logic, protocol quirks, business rules, and modeling assumptions.
- Do not add comments that simply restate code syntax.
- If an implementation depends on a competition constraint, document it near the code.

## Structure

- Keep paths descriptive and stable.
- Keep challenge workspaces mirrored when practical.
- Add `README.md` files to new folders intended for shared use.

## Experiments

- Log what was tried, why, and what happened.
- Prefer explicit baselines and comparisons over vague "improved" claims.
- Save enough metadata to reproduce useful results later.

## Definition Of Done

A piece of work is only done when:

- The code or doc change exists
- The relevant context files are updated
- The next agent can tell what changed and what remains
# Championship Overview

This document is the shared high-level view of the three main championship cases.

## Working Portfolio Assumption

Treat the overall championship as a balanced portfolio:

- `tripletex`: one third
- `norgesgruppen-data`: one third
- `astar-island`: one third

This is a working assumption pending official verification.

## Case Summaries

### Tripletex

- AI accounting agent
- HTTPS `/solve` endpoint
- Multilingual prompts and optional file attachments
- Fresh sandbox account per submission
- Best score per task is retained
- Strong upside from LLM planning and efficient tool execution

### NorgesGruppen Data

- Offline object detection and product identification
- Upload a zip containing `run.py` and weights
- Sandboxed GPU inference
- Score combines detection and classification
- Strong upside from computer vision plus retrieval-style product identification

### Astar Island

- Observation-limited probabilistic prediction challenge
- Limited query budget over stochastic simulations
- Need calibrated full-map class distributions
- Strong upside from active information gathering and probabilistic modeling

## Strategic Implication

Do not behave as if one elite solution can carry the championship. We need:

1. A strong floor on all three cases
2. At least one genuinely high-end solution
3. Shared infrastructure and documentation that accelerates all workstreams

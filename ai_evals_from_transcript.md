# Build AI Evals by Looking at Real Failures First

![Conversation traces with human-marked failures flowing into an evaluation loop](ai_evals_cover.png)

*A practical guide to error analysis, automated evaluation, and improving AI products, adapted from Lenny Rachitsky's conversation with Hamel Husain and Shreya Shankar.*

When an AI product gives a bad answer, a team can change the prompt, try a different model, or add another rule. But without a reliable way to tell whether the change helped, each fix is partly a guess. It may solve the example in front of you while breaking a different kind of interaction.

That is the problem evals are meant to solve. In the broad sense used by Husain and Shankar, an eval is any systematic way to understand and measure the quality of an AI application so the team can improve it. Unit tests belong in that toolkit, but so do reviewing production conversations, tracking product metrics, monitoring specific failure modes, and comparing experiments. The point is not to accumulate scores. It is to make the product better.

## Start with the product, not the test

Traditional software tests often begin with a known requirement and a predictable output. AI applications are less tidy. Users ask unexpected questions, the model's responses vary, and failures can arise anywhere in a chain of retrieval, tool calls, conversation state, and generated text. A team that writes tests only for the failures it imagined in advance may miss the problems users actually encounter.

The first step, then, is error analysis: inspect real interactions before deciding what to measure. A trace records the sequence behind an AI response, including the user request, model messages, tool calls, and outputs. Looking at individual traces can reveal issues that a dashboard or a generic quality score would hide.

In the interview, Husain walks through anonymized traces from Nurture Boss, an AI assistant for apartment-property managers. One prospective renter asks about a one-bedroom apartment with a study. The assistant checks availability, offers other units, and then says it cannot provide the requested information. The answer is not necessarily factually wrong. But for a product intended to help manage leasing leads, ending the conversation there may be a missed opportunity for a human handoff or follow-up.

Other traces expose different problems. Text messages arrive in fragments, and the conversation flow becomes confusing. The assistant offers a virtual tour that the property does not provide. In another exchange, it transfers a resident without telling them what is happening. These are not variations of a single "bad answer" score. They are distinct failures with different remedies: product behavior, message handling, factual grounding, and transfer etiquette.

## Make the first pass human

Husain and Shankar recommend taking short, descriptive notes while reviewing a sample of traces. This initial pass is called *open coding*. For each interaction, identify the first significant or most upstream thing that went wrong, record it in plain language, and move on. "Transferred the call without confirming it with the user" is useful; "janky" is not. The note should be specific enough that someone can understand and classify it later without reconstructing the entire conversation.

The reviewer needs domain and product context. An LLM might see a fluent response about virtual tours and judge it acceptable because it does not know the property offers no virtual tours. A product manager or other domain expert can recognize the mismatch. For that reason, the guests caution against automating the free-form discovery stage with another model. AI can assist later, once a human has identified meaningful problems.

The review does not require a large committee. One trusted domain expert can often make the initial calls faster and more consistently than a group debating every label. Husain describes this person, half-jokingly, as a "benevolent dictator": someone whose judgment is good enough to keep the process moving. In a legal product that person might be a legal expert; in a leasing product, someone who understands leasing operations. Often it is a product manager.

Nor is there a universal sample size. The guests suggest reviewing roughly 100 traces as an approachable starting point, not as a statistical law. The useful stopping point is *theoretical saturation*: when additional examples no longer reveal materially new kinds of failure. Depending on the product and the reviewer, that may happen earlier or later.

## Turn notes into a map of failure modes

Once the team has a collection of open-coded notes, an LLM becomes useful as an organizer. It can propose categories for similar notes, a process called *axial coding*. The team then reviews those categories, makes them specific and actionable, and assigns each note to a category. An explicit "none of the above" option helps surface failures the current taxonomy misses.

For the property-management assistant, useful categories included conversation-flow problems, human-handoff issues, tour-scheduling problems, formatting errors, and follow-up promises the assistant could not keep. These labels are more valuable than a generic bucket such as "capability limitations" because they suggest what the team might investigate or change.

The next analytical step can be simple counting. A spreadsheet pivot table can show how often each category appears in the reviewed sample; in the example, conversational-flow issues appeared 17 times. That does not mean frequency alone determines priority. A rarer failure may carry much greater risk or frustrate an important user journey. Counting gives the team a starting map, while product judgment determines where to act.

This is also the point to ask whether a failure needs an automated eval at all. Some problems are straightforward engineering defects or omissions in the system instructions. Fix those directly. Building a sophisticated evaluator for an obvious bug can cost more than it helps. Reserve the heavier machinery for failures that recur, are hard to detect by eye at scale, or remain difficult to eliminate after a simple fix.

## Choose the cheapest evaluator that works

An automated evaluator should target a specific failure mode. If the question is whether the response is valid JSON, follows a required format, or stays under a length limit, ordinary code may be enough. Code-based checks are usually simpler and cheaper than asking another model to judge the response.

More subjective behavior can justify an *LLM-as-judge*. Consider the handoff problem: when should a leasing assistant involve a human? The answer may depend on whether the user explicitly requested a person, whether a policy requires transfer, whether the necessary tool data is unavailable, or whether the request concerns a sensitive or time-critical issue. A judge can be instructed to evaluate that one narrowly defined failure mode across many traces.

The guests recommend a binary decision, such as whether a handoff error occurred, rather than a vague one-to-five quality score. A clear yes-or-no rule forces the team to articulate what counts as acceptable behavior. It also produces a metric people can interpret. A shift from 3.2 to 3.7 on an undefined scale is much harder to use for product decisions.

Writing the judge prompt is not the end of the work. Before trusting it, compare its decisions with human-labeled examples. Look separately at cases where the judge flags a failure the human did not and cases where it misses a failure the human found. Revise the rubric against those disagreements, and validate on examples the judge was not tuned on.

Overall agreement can be misleading. If a failure occurs in only 10% of examples, a judge that says "no failure" every time will appear to agree with humans 90% of the time while catching none of the actual failures. A confusion matrix, or at least a breakdown of both kinds of disagreement, tells a more honest story than a single agreement percentage.

## Use evals as a living product specification

The process changes how a team writes requirements. A well-defined evaluator describes what the assistant should do in a concrete situation and checks that expectation repeatedly. In that sense, an eval can act like a living product requirement. It does not replace an initial product brief, but it becomes more precise as the team observes real users and discovers failure modes it could not have specified up front.

Shankar calls attention to *criteria drift*: people's idea of good and bad output changes as they review more examples. That is not a sign that the team failed to plan. It is a property of building open-ended AI systems. Requirements, rubrics, and evals should be revised as the team learns.

Once an evaluator has been checked against human judgment, it can serve two purposes. Run it on known examples before shipping a change to catch regressions. Also run it on samples of production traces to see whether the failure persists, increases, or appears in new contexts. The same artifact supports development and ongoing monitoring.

This broader view explains why the debate over "evals versus vibes" is often less stark than it sounds. Close dogfooding by domain experts, careful review of failures, A/B tests, and production metrics can all contribute to systematic product assessment. But the approach that works for developers using a coding assistant all day may not work for a team building a medical or leasing assistant whose real users and domain experts are elsewhere. The important question is not whether a team uses the word *eval*. It is whether the team has a dependable feedback loop grounded in how its product actually behaves.

## A practical place to begin

For a team starting from scratch, the sequence is manageable:

1. Collect a sample of real traces across the product's main user journeys.
2. Have a domain expert note the first meaningful failure in each trace.
3. Keep sampling until new failure types become uncommon.
4. Group the notes into actionable categories, using an LLM to assist with synthesis but reviewing its work.
5. Count the categories, weigh their user and business impact, and fix obvious problems directly.
6. For persistent or hard-to-measure failures, build a narrow code-based check or a binary LLM judge.
7. Compare the evaluator with human judgments before using it in tests or production monitoring.

The initial investment is real, but it need not become a permanent annotation project. Shankar describes spending several days on the first round for an application, then returning to the data for perhaps half an hour a week once the process is in place. The exact schedule will vary. The principle is to make looking at real behavior easy enough that the team keeps doing it.

The best eval process is not the most elaborate one. It is the one that reveals a problem the team can fix, shows whether the fix worked, and keeps surfacing the next important failure. Start by looking at the data. Everything else follows from what you find.

*Source: [Lenny's Podcast interview with Hamel Husain and Shreya Shankar](https://www.lennysnewsletter.com/p/why-ai-evals-are-the-hottest-new-skill), adapted from the supplied transcript. Sponsorship messages and the unrelated lightning round have been omitted.*

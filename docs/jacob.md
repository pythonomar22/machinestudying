
Jacob X. Li ✈️ ICML
@jacobli99
3.4K Followers · Joined September 2020

View Profile
Mon, Jun 29
Hey Jacob! My name is Omar, and I'm a research intern at Sakana for the summer
11:22 PM
11:22 PM
Edited
I'm interested in your Machine Studying blog and paper that you released recently, and trying to replicate some of your results


I'm sort of getting inflated results when trying to replicate the baseline scores on DSPy and OpenClaw (like 3x the scores you get in your paper)
Would it be possible for you to share your grader prompt with me? Or if you could share any wisdom on points of divergence that you would stress when trying to replicate results?
11:23 PM
11:23 PM
Tue, Jun 30
This conversation is now end-to-end encrypted
Hey Omar!
1:01 AM
1:01 AM
Hey Omar!
1:01 AM
1:01 AM
Wonder which model / harness you are testing?
And which model were you using as the grader?
(As reference I was using gpt5.4 as grader)
1:05 AM
1:05 AM
And as for grading there are two parts, first step is to launch a sandbox and run the generated code to make sure it’s got correct syntax.
1:08 AM
1:08 AM
The second step is the llm as judge to judge according to my rubric
1:09 AM
1:09 AM
And there is indeed a caveat here - so per what I refer to as ‘strict’ grading - if a model misses any of the core rubric it will automatically get a zero for the whole problem
1:11 AM
1:11 AM
This might sounds too strict but in fact current model are really good at reinventing wheels and most of the time the reason we wanted to use an existing library is to have it use existing apis instead of reinventing them
1:13 AM
1:13 AM
❤️
this core-conjunctive way of grading should bring scores down by a lot
1:17 AM
1:17 AM
hhhhh i see
im using fugu haha, sakana gives us free credits
1:18 AM
1:18 AM
i saw that you were using gpt 5.4 in the paper, but i didnt think that a change of the model from 5.4 to fugu would result in such a drastic change
this makes a ton of sense
1:20 AM
1:20 AM
Jacob X. Li ✈️ ICML
This might sounds too strict but in fact current model are really good at reinventing wheels and most of the time the reason we wanted to us
no this makes sense
1:21 AM
1:21 AM
Jacob X. Li ✈️ ICML
And there is indeed a caveat here - so per what I refer to as ‘strict’ grading - if a model misses any of the core rubric it will automatica
when you say "miss" here, you mean any score < 1.0?
or do you mean if a core claim scores 0, then the whole question scores 0
1:23 AM
1:23 AM
like if a core claim gets partial credit, the whole question scores 0 ?
1:24 AMim using qwen3.5 9b, same as the one in the paper, with a very bare bones react harness
1:25 AM
1:25 AM
Omar Abul-Hassan
when you say "miss" here, you mean any score < 1.0?
Oh thanks for catching this - I made a small changes here so the grader is only giving 0 or 1 here because having the .5 makes the variance larger than expected
And thus if it’s not 1 for the item it’s a miss
1:30 AM
1:30 AM
got itttt, my results were driving me crazy for the past couple of days
i really like and appreciate the work, i think its very intuitive
1:31 AM
1:31 AM
Thank you!
Lmk if you have more questions!
1:31 AM
1:31 AM
great, will do! have a good time at icml
1:32 AM
1:32 AM
Thank you!
1:32 AM
1:32 AM
I guess one of the more important take aways is that you could improve ‘performance’ by using more complicated harness but that doesn’t stop the agent to have low expertise.
1:38 AM
1:38 AM
yeah
9:52 AM
9:52 AM
to be clear, there is a studying phase and a test/eval phase, right?that is, the tokens produced during this study phase are not counted on the x axis at eval time: so for the cheatsheet case, the tokens spent to produce the cheatsheet are not counted on the x axis and the notes are free at test time basically
9:54 AM
9:54 AM
my work at sakana isnt fully clear yet and i was sort of just trying to replicate your results to understand things better, but i had in mind trying to develop study procedures to raise expertise, that is you could run some sort of meta learning approach to learn a better cheatsheet (can be seen as a more general approach of the cheatsheet), or perhaps you can propose a studying procedure that involves quizzing yourself during the studying phase (which, again, these tokens aren't counted according to my understanding)
9:56 AM
9:56 AM
👍
would appreciate your thoughts on the above if you have any ^.
9:58 AM
9:58 AM
Edited
also if you have any other future work directions that youd like to see pursued
9:58 AMto be clear, there is a studying phase and a test/eval phase, right?
yes
10:30 AM
10:30 AM
yeah I think those are all really great directions!
10:31 AM
10:31 AM
hey jacob, i had a couple questions about replicating that i'm revisiting now that i've tried to do a couple experiments on my own and realized that some things were kind of up to interpretation when i first tried to replicate results
1:13 PM
1:13 PM
Edited
1. for the lenient columns in table 1: does the core-conjunctive zero still apply here? (so lenient would skip the deterministic compile/hallucinated-api zeros), or is lenient just the pure weighted sum of claim scores?
1:14 PM
1:14 PM
2. for the react harness: are you doing native tool calling (tools param and parser) or just like raw text thought/action/observation? and did prior turns' thinking stay in context within an episode or get stripped? asking because my forced-20 episodes land around 6k tokens vs your ~35k. your numbers look like to me maybe that the model re-thinks ~1.5k tokens fresh every turn, which i only get when thinking is NOT carried across turns
1:16 PM
1:16 PM

Omar Abul-Hassan
1. for the lenient columns in table 1: does the core-conjunctive zero still apply here? (so lenient would skip the deterministic compile/hal
lenient is just weights summed together
1:46 PM
1:46 PM
Omar Abul-Hassan
2. for the react harness: are you doing native tool calling (tools param and parser) or just like raw text thought/action/observation? and d
I was using dspy.ReAct but I think native tool call should work (and yes model emits reasoning every turn)
1:48 PM
1:48 PM
Edited
great, thanks
2:32 PM
2:32 P
reading thru dspy.react, how exactly did you do no early stopping for the forced-20 runs? it seems like dspy.react breaks out of the loop the moment the model picks finish, so did you remove finish from the tool list, ignore finish selections and keep looping, or reprompt it ? and did the model know upfront it had to do all 20 iterations, or did it just discover it couldn't finish? (same question for the cheatsheet study loop)
also if u mind sharing what the direct setting was concretely: dspy.Predict, ChainOfThought, or ReAct with max_iters=0? i guess the last one would just fall thru to the extract step
my native-tool calling numbers are running pretty hot, like 2x the lenient scores
4:30 PM
4:30 PM
also if you could share the shape of the three tool functions? does read_file read whole files or line ranges, or any glob semantics, or any specific truncations that you did to bservations before going into the trajectory
4:32 PM

reading thru dspy.react, how exactly did you do no early stopping for the forced-20 runs? it seems like dspy.react breaks out of the loop the moment the model picks finish, so did you remove finish from the tool list, ignore finish selections and keep looping, or reprompt it ? and did the model know upfront it had to do all 20 iterations, or did it just discover it couldn't finish? (same question for the cheatsheet study loop)
also if u mind sharing what the direct setting was concretely: dspy.Predict, ChainOfThought, or ReAct with max_iters=0? i guess the last one would just fall thru to the extract step
my native-tool calling numbers are running pretty hot, like 2x the lenient scores
4:30 PM
4:30 PM
also if you could share the shape of the three tool functions? does read_file read whole files or line ranges, or any glob semantics, or any specific truncations that you did to bservations before going into the trajectory
4:32 PM
4:32 PM
Omar Abul-Hassan
reading thru dspy.react, how exactly did you do no early stopping for the forced-20 runs? it seems like dspy.react breaks out of the loop th
Just catch the finish and return something like you gotta keep searching type of logic no need to remove that specific turn
4:33 PM
4:33 PM
👍
Omar Abul-Hassan
my native-tool calling numbers are running pretty hot, like 2x the lenient scores
Yeah it’s possible that native tool calling helps with the performance (but I wasn’t expecting 2x improvement )
4:35 PM
4:35 PM
Three tools are Grep glob, and read file (lines, Capped at 200lines)
Omar Abul-Hassan
also if u mind sharing what the direct setting was concretely: dspy.Predict, ChainOfThought, or ReAct with max_iters=0? i guess the last one
Dspy predict
4:37 PM
4:37 PM
Omar Abul-Hassan
my native-tool calling numbers are running pretty hot, like 2x the lenient scores
But it wouldn’t matter much after all, the reason for including this lenient grading was to demonstrate a prettier inf scaling curve
4:39 PM
4:39 PM
And a qualifying MS algorithm will need to shit that frontier
4:41 PM
4:41 PM
👍Edited
thank you so much
Jacob X. Li ✈️ ICML
But it wouldn’t matter much after all, the reason for including this lenient grading was to demonstrate a prettier inf scaling curve
i see, yeah
4:44 PM
4:44 PM




# Text-Only Diagram Description Guide

This guide is for the real usage case where you cannot upload an image and must describe a control-systems diagram, plot, or figure in text.

The goal is simple:

- help the assistant reconstruct the missing image as accurately as possible
- reduce ambiguity in block diagrams, root-locus plots, pole-zero maps, and response plots
- make the final answer more complete and less likely to miss key visual details

Use this guide whenever the original problem depends on a figure, chart, block diagram, or handwritten annotations that are not directly available to the assistant.

## Core Principle

Do not describe the figure casually.

Always describe it in this order:

1. What kind of figure it is
2. What the question is asking
3. What text and labels are visible
4. What the structure or geometry of the figure looks like
5. Which values/functions/poles/zeros are given
6. What is unclear or partially unreadable

If you are unsure about a detail, say it is unclear.
Do not guess.

## Best General Template

Copy this template and fill it in.

```text
Problem type:

Question goal:

Visible text in the image:

System structure / figure layout:

Given functions / values / labels:

Graph or geometric clues:

What is unclear:
```

## How To Fill Each Section

### Problem type

State what kind of figure it is.

Examples:

- block diagram
- root locus plot
- pole-zero map
- Bode magnitude/phase plot
- Nyquist plot
- step response plot
- impulse response plot
- hand-drawn derivation
- formula sheet

Good example:

```text
Problem type:
Root locus plot for a unity-feedback system with proportional gain K
```

### Question goal

Write exactly what the problem is asking you to find.

Examples:

- stable range of K
- crossover frequency
- closed-loop transfer function
- steady-state error
- dominant poles
- damping ratio
- rise time / overshoot / settling time

Good example:

```text
Question goal:
Find the range of K for stability, then estimate steady-state error and time-response metrics.
```

### Visible text in the image

Write the text you can read as literally as possible.

Include:

- problem statement text
- labels near blocks
- axis labels
- gain labels
- any handwritten notes
- values printed near poles, zeros, or intersections

Good example:

```text
Visible text in the image:
"Use Routh's criterion to determine the range of K."
"For Kc, the closed-loop poles are p1 = -1.65 + j2.89 and p2 = -0.35 + j2.61."
Horizontal axis: Real axis
Vertical axis: Imaginary axis
```

### System structure / figure layout

Describe the geometry or connection order.

For block diagrams:

- where the input starts
- where the summing junction is
- whether feedback is positive or negative
- which blocks are in forward path
- which block is in feedback path
- where the output is taken

For plots:

- where branches or curves start and end
- whether poles/zeros are on real axis or complex plane
- number of branches
- approximate symmetry

Good example:

```text
System structure / figure layout:
R(s) enters a summing junction.
The forward path is K followed by G(s).
The output is C(s).
There is a unity negative feedback path from C(s) back to the summing junction.
```

Another example:

```text
System structure / figure layout:
The root locus has 4 branches.
Two branches start near the left half-plane and two are closer to the imaginary axis.
The plot is symmetric about the real axis.
```

### Given functions / values / labels

List all explicit mathematical objects.

Include:

- transfer functions
- controller form
- pole/zero values
- gain values
- damping ratio / natural frequency
- crossover frequency
- constants like zeta, wn, sigma, z

Good example:

```text
Given functions / values / labels:
G(s) = 10 / (s(s + 2)(s + 5))
H(s) = 1
Controller is proportional gain K
Given closed-loop poles: -1.65 + j2.89 and -0.35 + j2.61
Target crossover frequency: 3 rad/s
```

### Graph or geometric clues

Describe what the figure visually suggests, even if exact formulas are not shown.

Useful clues:

- where a branch crosses the imaginary axis
- where the peak occurs
- whether overshoot looks high or low
- whether a plot settles slowly
- approximate breakaway point
- whether a Bode magnitude crosses 0 dB near a certain frequency

Good example:

```text
Graph or geometric clues:
The dominant poles are the pair closer to the imaginary axis.
The response appears lightly damped with large overshoot.
The root-locus branches cross the imaginary axis at a positive frequency.
```

### What is unclear

This section is extremely important.

Explicitly list unreadable or uncertain details.

Examples:

- one block label is blurry
- not sure whether the feedback is positive or negative
- one zero may be at -2, but the label is unclear
- axis scale is not visible

Good example:

```text
What is unclear:
The exact location of one zero is hard to read.
I am not fully sure whether the feedback sign is negative, but it looks like standard negative feedback.
```

## Control-Systems-Specific Templates

## 1. Block Diagram Template

Use this for closed-loop systems, controller/plant diagrams, and interconnections.

```text
Problem type:
Block diagram

Question goal:

Visible text in the image:

Forward path:

Feedback path:

Summing junction signs:

Input / output names:

Given transfer functions / gains:

What is unclear:
```

Example:

```text
Problem type:
Block diagram

Question goal:
Find the closed-loop transfer function and determine steady-state error.

Visible text in the image:
R(s), C(s), K, G(s), H(s)

Forward path:
R(s) -> summing junction -> K -> G(s) -> C(s)

Feedback path:
C(s) -> H(s) -> back to the summing junction

Summing junction signs:
Positive input from R(s), negative feedback from H(s)C(s)

Input / output names:
Input is R(s), output is C(s)

Given transfer functions / gains:
G(s) = 5 / (s(s+1))
H(s) = 1
K is proportional gain

What is unclear:
None
```

## 2. Root Locus Template

Use this when the figure is mainly a root-locus plot.

```text
Problem type:
Root locus

Question goal:

Visible text in the image:

Number of branches:

Open-loop poles:

Open-loop zeros:

Given closed-loop poles:

Real-axis / imaginary-axis clues:

Asked quantities:

What is unclear:
```

Example:

```text
Problem type:
Root locus

Question goal:
Find the stable range of K and estimate time-response quantities at a given operating point.

Visible text in the image:
The plot shows 4 branches.
The problem statement gives p1 = -1.65 + j2.89 and p2 = -0.35 + j2.61.

Number of branches:
4

Open-loop poles:
Not explicitly written in the image summary

Open-loop zeros:
Not visible

Given closed-loop poles:
-1.65 + j2.89 and -0.35 + j2.61, plus conjugates

Real-axis / imaginary-axis clues:
The locus is symmetric about the real axis.
The dominant pair is the one closer to the imaginary axis.

Asked quantities:
Stable range of K, crossover-related Kc, steady-state errors, rise time, overshoot, settling time

What is unclear:
Exact open-loop poles and zeros from the figure are not readable
```

## 3. Pole-Zero Map Template

```text
Problem type:
Pole-zero map

Question goal:

Visible text in the image:

Poles:

Zeros:

Stability clues:

What is unclear:
```

Example:

```text
Problem type:
Pole-zero map

Question goal:
Determine stability and infer likely transient behavior.

Visible text in the image:
Crosses mark poles and circles mark zeros.

Poles:
One real pole at about -2 and one complex pair near -0.5 ± j2

Zeros:
One zero near -1

Stability clues:
All visible poles appear to be in the left half-plane

What is unclear:
The exact zero location is slightly blurry
```

## 4. Time-Response Plot Template

```text
Problem type:
Time response plot

Question goal:

Input type:

Visible text in the image:

Approximate final value:

Approximate overshoot:

Approximate rise time:

Approximate settling time:

Other visual clues:

What is unclear:
```

Example:

```text
Problem type:
Time response plot

Question goal:
Estimate damping ratio, natural frequency, rise time, and percent overshoot.

Input type:
Unit step

Visible text in the image:
Output c(t) vs time t

Approximate final value:
1

Approximate overshoot:
About 20%

Approximate rise time:
About 0.5 s

Approximate settling time:
About 2.5 s

Other visual clues:
The curve oscillates slightly before settling

What is unclear:
Exact peak time is not labeled
```

## 5. Frequency-Response Plot Template

Use this for Bode or Nyquist style figures.

```text
Problem type:
Frequency-response plot

Question goal:

Plot type:

Visible text in the image:

Magnitude clues:

Phase clues:

Crossover clues:

Stability margin clues:

What is unclear:
```

Example:

```text
Problem type:
Frequency-response plot

Question goal:
Estimate crossover frequency and discuss stability margins.

Plot type:
Bode magnitude and phase

Visible text in the image:
Frequency axis is logarithmic

Magnitude clues:
Magnitude crosses 0 dB near 3 rad/s

Phase clues:
Phase is around -140 degrees near crossover

Crossover clues:
Gain crossover appears close to 3 rad/s

Stability margin clues:
Looks like positive phase margin

What is unclear:
Exact margin values are not labeled
```

## Good vs Bad Descriptions

### Bad

```text
There is a root locus and some poles and I think it wants stability and maybe overshoot.
```

Why bad:

- does not say what is visible
- does not give values
- does not describe figure structure
- does not separate known facts from guesses

### Better

```text
Problem type:
Root locus

Question goal:
Find the stable range of K and estimate rise time, overshoot, and settling time.

Visible text in the image:
The problem gives p1 = -1.65 + j2.89 and p2 = -0.35 + j2.61.

Number of branches:
4

Real-axis / imaginary-axis clues:
The root locus is symmetric about the real axis, and the dominant poles are the pair closer to the imaginary axis.

What is unclear:
The original open-loop poles/zeros are not clearly readable from the figure.
```

## Best Practices

- Copy text exactly when possible
- Use bullet points for values and labels
- Separate facts from guesses
- Always mention what is unclear
- If the figure is long, describe it from left to right or input to output
- If the figure includes formulas, type them exactly
- If the figure includes axes, always name both axes
- If the figure includes several subparts, describe each subpart separately

## Recommended Short Prompt For Daily Use

If you want a shorter instruction for yourself, use this:

```text
I cannot upload the image, so I will describe it in structured form.

Problem type:
Question goal:
Visible text in the image:
System structure / figure layout:
Given functions / values / labels:
Graph or geometric clues:
What is unclear:
```

## Recommended Workflow In This Project

When you describe a missing image to this assistant:

1. Start with `Problem type`
2. State the exact question goal
3. Copy every visible equation and value
4. Describe the structure of the diagram or plot
5. List unclear items explicitly

If the problem has multiple visual parts, split your description into:

- Part A
- Part B
- Part C

This is much better than mixing everything into one paragraph.

## Final Advice

If the image is important, the quality of your text description directly controls the quality of the answer.

A good description should let someone who never saw the image reconstruct:

- what kind of figure it is
- what mathematical objects are given
- how the figure is arranged
- what the question wants
- what details are missing or uncertain

If you follow the templates above, the assistant will perform much better than with a casual free-form description.

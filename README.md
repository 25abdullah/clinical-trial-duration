# Predicting How Long a Clinical Trial Will Take

**[Live demo](ADD_YOUR_STREAMLIT_URL_HERE)** · [Data notebook](notebooks/Medical_trial_Project.ipynb) · [Neural network notebook](notebooks/NN_from_Scratch_Project.ipynb)

![Demo](images/demo_screenshot.png)

---

## Project overview

Before a drug reaches patients, it has to go through clinical trials. Those
trials take time, and nobody knows in advance how much. A study planned for two
years can run four.

This project asks a simple question: **using only the information a sponsor
records on the day a trial is registered, how long will that trial take?**

The answer turns out to be "roughly, but not precisely," and that finding
shapes everything about how the results are reported here.

### Why it matters

A trial that runs long delays the treatment reaching the people who need it.

There are financial consequences too. Trial delays are one of the largest cost
lines in pharmaceutical research. Underestimating duration means a sponsor has
to find more money partway through. Overestimating means capital sits reserved
that could have funded a different drug. A patent clock runs the whole time, so
a slow trial shortens the window in which the drug can be sold exclusively,
which is what funds the next one.

---

## Objectives and goals

1. **Build a genuinely honest prediction.** Only use information available at
   registration. Anything recorded during or after the trial would leak the
   answer and make the model look better than it is.
2. **Understand the ceiling.** Find out how much of trial duration is
   predictable at all, and explain what accounts for the rest.
3. **Implement a neural network from scratch.** Write the mathematics by hand
   in NumPy rather than calling a library, and verify it numerically.
4. **Report uncertainty, not just a number.** If the prediction is uncertain,
   say so in a way a planner could actually use.

---

## The data

**AACT** (Aggregate Analysis of ClinicalTrials.gov), maintained by the Clinical
Trials Transformation Initiative, a partnership between the FDA and Duke
University. It is a public database of every trial registered in the United
States, refreshed daily, spread across more than 40 linked tables.

Seven of those tables were used here, filtered to interventional trials that
have completed, registered after 2007. That gives **220,088 trials**.

> *Aggregate Analysis of ClinicalTrials.gov (AACT) Database, Clinical Trials
> Transformation Initiative (CTTI). https://aact.ctti-clinicaltrials.org/*

---

## Methodology

### 1. Getting the data out

The seven tables do not line up neatly. Four have one row per trial and join
directly. Three have many rows per trial, since a study can have several
conditions, several interventions, several collaborating sponsors. Joining
those directly would duplicate the trial: a study of five conditions would
appear five times and be counted five times over. Each was collapsed to one row
per trial first.

### 2. Deciding what counts as fair information

This was the part that shaped the project. Somebody asking "how long will my
trial take" has not run it yet, so any field filled in later is off limits.
AACT makes this mostly mechanical, since its table names mark registration-time
information separately from post-completion results.

One column is a partial exception and is disclosed rather than hidden.
Enrollment is recorded as a target at registration but overwritten with the
final count when the trial ends, and the original is not recoverable. It is
kept, because most of what it carries is intended trial size, a legitimate
design decision. But it is not strictly a day-one value.

### 3. Cleaning, and the principle behind it

The guiding rule throughout: **filling in a missing value is only appropriate
when the value exists but was not written down. When a blank means the thing
does not exist, use a marker plus a flag instead.**

Examples of that in practice:

| Situation | Decision |
|---|---|
| No minimum age listed | Fill with 0. The trial is open from birth, so that is the true value, not a guess. |
| No maximum age listed | No upper limit exists. Filling in the oldest age seen would invent a ceiling, so it gets a placeholder plus a separate "has a maximum" flag. |
| 54% of trials have no phase | Not missing. Device, behavioural and surgical studies do not use the phase system. Dropping them would cost half the data. |
| 5% report zero trial sites | Impossible for a completed trial. Flagged rather than deleted, since poor record-keeping may itself predict a slow trial. |
| Design fields missing (under 2%) | Rows dropped. Filling in the most common value would invent a decision the investigators never made. |

### 4. Choosing how to measure duration

Trial length is extremely lopsided. Most trials cluster at the short end while
a few stretch out for years, so the raw numbers are crushed into one bar with a
long thin tail. Taking the logarithm spreads them out into a shape the models
can work with.

![Duration distribution](images/duration_distribution.png)

The left panel is raw duration in days. The right is the same data after the
transformation. Every result in this project is measured on the right-hand
scale and translated back into days when reported.

### 5. Exploring the data

Every feature was checked two ways: does it move the typical duration, and does
it narrow the range of possible durations?

**Every feature moved the typical duration. None of them narrowed the range.**

That is the central finding, and it is the reason this project reports ranges
instead of single numbers. Knowing a trial's phase, sponsor type, size and
design tells you where its likely duration sits, but the spread around that
point stays about as wide as it was before you knew anything.

### 6. Building and comparing models

Eight models, each isolating one change so the comparison means something:

- A baseline that always predicts the average
- Linear regression
- Three linear variants with hand-engineered features
- Two library neural networks (scikit-learn and Keras)
- **A neural network written from scratch in NumPy**

The from-scratch network is the centrepiece. Two hidden layers, every
derivative worked out by hand, no automatic differentiation. Correctness was
verified by nudging individual weights and measuring how the error actually
changed, then comparing that against what the hand-derived formulas predicted.
**They agreed to nine decimal places on all three layers**, which is how you
know the mathematics is right rather than merely plausible.

### 7. Reporting uncertainty

Rather than a single number, each model reports an **80% range**, built from
how wrong that model actually was on trials it had never seen. If 80% of past
errors fell within a certain span, a new prediction gets that same span
attached.

---

## Results and key findings

### How the models compare

| Model | R² |
|---|---|
| Always predict the average | 0.000 |
| Linear regression | 0.345 |
| Linear + engineered features | 0.358 |
| **Neural network, from scratch** | **0.361** |
| scikit-learn neural network | 0.400 |
| Keras neural network | 0.409 |
| **Keras + log(enrollment)** | **0.415** |

![Model comparison](images/model_comparison.png)

The best model explains about **42% of the variation** in trial duration, a
24% improvement in error over the baseline.

### Four findings worth stating

**1. The models are much closer together than expected.** Everything from plain
linear regression to a tuned neural network lands within seven percentage
points. The choice of model matters far less than most projects imply.

The clearest way to see this is to plot what each model predicted against what
actually happened:

![Predicted vs actual](images/predicted_vs_actual.png)

If a model were perfect, every point would sit on the diagonal line. Instead
all seven clouds look nearly identical. That is the visual argument for
reporting ranges: no model separates itself enough for a single number to be
trustworthy.

**2. One transformation helped every single model.** Trial enrollment ranges
from a handful of people to a recorded twenty million. That largest value
turned out to be a *date* typed into a numeric field. Compressing that range
improved every model tested and, more interestingly, fixed a training problem.

Here is the network learning on the raw enrollment values:

![Training curves, raw enrollment](images/training_curves_raw.png)

The error swings by roughly a factor of two between rounds, so the final result
depends heavily on where training happened to stop.

And the same network after the transformation:

![Training curves, log enrollment](images/training_curves_log.png)

Same architecture, same data, one column rescaled. The instability is gone.

**3. Roughly 58% of the variation cannot be predicted from registration data.**
This is not a modelling failure. The features carry independent information and
are not redundant with one another, so the missing signal is not something left
out of the query. It is information that does not exist on day one: how quickly
sites actually recruit patients, whether the protocol gets amended, whether a
competing trial is chasing the same participants.

**4. The from-scratch network performs like the linear models, not like the
libraries.** It beats plain linear regression, so it did learn something a
straight line cannot represent. But it trails scikit-learn and Keras. Since the
mathematics is verified correct, the gap is training strategy: this
implementation takes 300 careful steps where the libraries take roughly 27,000
smaller, smarter ones.

### What the ranges actually look like

A prediction of 800 days comes with a range of roughly **260 to 2,300 days**.

That is wide, and it is meant to be. It is the honest picture of a problem
where most of what determines the answer has not happened yet. For someone
budgeting against a forecast, the range is more useful than the midpoint.

### Known limitations

- **Only completed trials are visible.** Trials that were terminated, or are
  still running long past expectation, have no end date and are absent from the
  data. Estimates therefore skew optimistic. The model answers "given that this
  trial finishes, how long will it take," not "will it finish."
- **The ranges are the same width for every trial.** A common, large,
  well-funded study gets the same span as a rare-disease study, even though the
  model should be far less confident about the second.

---

## Try it

The [live demo](ADD_YOUR_STREAMLIT_URL_HERE) picks a real trial the models
never saw during training, shows what three of them predicted, and reveals how
long it actually took. A tick mark shows whether the true duration fell inside
each model's range.

Click through a dozen trials and roughly eight of ten should land inside,
which is the range's claim, tested live.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Potential next steps

**Add the data that is available but unused.** Disease category is the most
promising, since a rare condition recruits far more slowly than a common one,
and AACT stores standardised disease terms in a table not queried here. Country
breakdown is second: regulatory approval times differ by jurisdiction, and a
trial spanning fifteen countries faces fifteen approval processes.

**Make the ranges adapt to the trial.** Currently every prediction gets the
same span. Methods exist that widen it for unusual trials and narrow it for
typical ones.

**Improve the from-scratch network's training.** The gap to the libraries is
training strategy, not correctness. Implementing mini-batch updates by hand
would close most of it and is a natural extension of the existing code.

**Model whether a trial completes at all.** The current model assumes
completion. Around 34,000 terminated trials sit in AACT unused, and "will this
finish" is arguably the more valuable question.

---

## Individual contributions

Solo project. All work, including database extraction and SQL, data cleaning
decisions, exploratory analysis, the from-scratch neural network and its
derivation, benchmarking, the Streamlit application, and this write-up, is my
own.

**Abdullah Ismail**, Computer Science & Mathematics, Northeastern University
[GitHub](https://github.com/25abdullah) · [LinkedIn](https://linkedin.com/in/abdullah-ismail-09a964368)

---

## Repository

```
README.md              this file
app.py                 the Streamlit demo
demo_data.npz          predictions and ranges the demo reads
requirements.txt       dependencies
notebooks/
  Medical_trial_Project.ipynb    data extraction through model comparison
  NN_from_Scratch_Project.ipynb  the neural network, derived and verified
```

**Tools:** Python, PostgreSQL, pandas, NumPy, scikit-learn, Keras, Streamlit
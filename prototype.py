import streamlit as st

import numpy as np

import matplotlib.pyplot as plt

from math import erf, sqrt
from contextlib import contextmanager


@contextmanager
def learning_section(title, completion_key, summary):
    """Keep feedback open until the learner opens a subsequent panel."""
    panel_key = f"learning_{st.session_state.step}_{completion_key}_{title}"
    sections = st.session_state.learning_sections
    waiting_for_next = any(
        st.session_state.get(section["completion_key"], False)
        and st.session_state.get(section["panel_key"], False)
        for section in sections
    )
    sections.append({"panel_key": panel_key, "completion_key": completion_key})
    # Keep the default stable so rerenders preserve user-controlled open state.
    defaults = st.session_state.setdefault("learning_panel_defaults", {})
    expanded = defaults.setdefault(panel_key, not waiting_for_next)
    with st.expander(
        title,
        key=panel_key,
        expanded=expanded,
        icon=":material/menu_book:",
        on_change=open_learning_section,
        args=(panel_key,),
    ):
        yield
        if st.session_state.get(completion_key, False):
            st.caption(summary)


def open_learning_section(panel_key):
    """Opening a panel folds completed panels before it, preserving answers."""
    if not st.session_state.get(panel_key, False):
        return
    for section in st.session_state.learning_sections:
        if section["panel_key"] == panel_key:
            break
        if st.session_state.get(section["completion_key"], False):
            st.session_state[section["panel_key"]] = False


def reset_keys(*keys):
    """
    Reset session-state values when the learner changes an answer.
    This prevents old feedback from remaining visible.
    """
    for key in keys:
        st.session_state[key] = False


def normal_cdf(value):
    return 0.5 * (1 + erf(value / sqrt(2)))


def navigation_buttons(show_continue=True):
    """
    Standard navigation shown at the bottom of Steps 1–8.
    """

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "← Previous",
            key=f"previous_{st.session_state.step}",
            width="stretch"
        ):
            st.session_state.step -= 1
            st.rerun()

    with col2:
        if st.button(
            "↺ Start Over",
            key=f"restart_{st.session_state.step}",
            width="stretch"
        ):
            st.session_state.clear()
            st.session_state.step = 0
            st.rerun()

    with col3:
        if st.button(
            "Continue →",
            key=f"continue_{st.session_state.step}",
            type="primary" if show_continue else "secondary",
            disabled=not show_continue,
            width="stretch"
        ):
            st.session_state.step += 1
            st.rerun()


def initialize_session_state():
    """Set the initial step for a new activity session."""
    if "step" not in st.session_state:
        st.session_state.step = 0


def render_progress():
    """Show named steps, highlighting the learner's current position."""
    current_step = st.session_state.step
    total_steps = len(STEP_RENDERERS) - 1
    current_label, _ = STEP_RENDERERS[current_step]
    progress = min(current_step / total_steps, 1.0)
    progress_text = (
        "Introduction · Your learning journey"
        if current_step == 0
        else f"Step {current_step} of {total_steps} · {current_label}"
    )
    st.progress(progress, text=progress_text)

    with st.container(horizontal=True, gap="xsmall"):
        for index, (label, _) in enumerate(STEP_RENDERERS):
            step_label = label if index == 0 else f"{index}. {label}"
            if index < current_step:
                st.badge(step_label, icon=":material/check:", color="green")
            elif index == current_step:
                st.badge(step_label, icon=":material/arrow_forward:",
                         color="blue")
            else:
                st.badge(step_label, color="gray")


def render_introduction():
    """Render step 0 and its answer checks."""

    st.title("Hypothesis Testing Activity")

    st.subheader(
        "Can you determine whether a mysterious coin is actually fair?"
    )

    st.write(
        """
        In this activity, you'll investigate a coin using statistical evidence.

        Along the way, you'll learn how **hypothesis testing** works —
        from understanding samples all the way to making a statistical decision.
        """
    )

    st.divider()

    st.header("🪙 The Fair Coin Investigation")

    st.write(
        """
        A manufacturer claims that their coin is perfectly fair.

        A fair coin should have a **50% chance of landing Heads**.

        You decide to investigate.
        """
    )

    st.info(
        """
        You flip the coin **100 times**.

        🟢 Heads: **60**

        ⚪ Tails: **40**
        """
    )

    answer = st.radio(
        "What is your first intuition?",
        [
            "The coin is probably fair",
            "The coin might be biased",
            "I'm not sure yet"
        ],
        index=None,
        key="intuition_question"
    )

    if answer is not None:

        st.success(
            "Good! But intuition alone isn't enough. "
            "Let's investigate using statistics."
        )

        st.divider()

        if st.button(
            "Start Investigation →",
            type="primary",
            width="stretch"
        ):
            st.session_state.step = 1
            st.rerun()


def render_data_step():
    """Render step 1 and its answer checks."""

    st.title("Step 1 — Understanding the Data")

    st.write(
        """
        Before we test whether the coin is fair,
        let's understand what our observations represent.
        """
    )

    # --------------------------------------------------------
    # Population vs Sample
    # --------------------------------------------------------

    with learning_section('Population vs sample', 'correct_step1', 'The 100 flips are a sample.'):
        st.subheader("Population vs Sample")

        st.info(
            "We flipped the coin **100 times** and observed "
            "**60 Heads and 40 Tails**."
        )

        def reset_sample_answer():
            st.session_state.sample_checked = False
            st.session_state.correct_step1 = False
            st.session_state.step1_complete = False

        question = st.radio(
            "Those 100 flips are:",
            [
                "The population",
                "A sample",
                "A parameter",
                "A hypothesis"
            ],
            index=None,
            key="sample_question",
            on_change=reset_sample_answer
        )

        if st.button("Check Answer", key="check_sample"):

            if question is not None:

                st.session_state.sample_checked = True

                st.session_state.correct_step1 = (
                    question == "A sample"
                )

        if st.session_state.get("sample_checked"):

            if st.session_state.get("correct_step1"):

                st.success(
                    "✅ Correct! The 100 flips are a **sample** from all "
                    "the possible coin flips we could observe."
                )

            else:

                st.error(
                    "Not quite. Think about whether these 100 flips "
                    "represent every possible outcome or only the "
                    "observations we collected."
                )

    # --------------------------------------------------------
    # Reveal remaining material only after correct answer
    # --------------------------------------------------------

    if st.session_state.get("correct_step1"):

        # ----------------------------------------------------
        # Parameter vs Statistic
        # ----------------------------------------------------

        with learning_section('Parameter vs statistic', 'random_n', 'p is the parameter; the observed proportion is a statistic.'):
            st.divider()

            st.subheader("Parameter vs Statistic")

            st.write(
                """
                The true probability that the coin lands on Heads is a
                **population parameter**:
                """
            )

            st.latex(
                r"p = \text{true probability of Heads}"
            )

            st.write(
                """
                But we do not know the true value of \(p\).

                What we *do* know is what happened in our sample:
                **60 Heads out of 100 flips**.
                """
            )

            st.latex(r"\hat{p} = \frac{60}{100} = 0.60")

            st.markdown(
                r"""
                The observed proportion $\hat{p}$ is a **statistic**.

                We use a statistic calculated from our sample to learn
                something about an unknown population parameter.
                """
            )

        # ----------------------------------------------------
        # Interactive sampling experiment
        # ----------------------------------------------------

        with learning_section('Run a sampling experiment', 'random_n', 'A fair coin can produce different sample proportions.'):
            st.divider()

            st.subheader("🪙 Random samples naturally vary")

            st.write(
                """
                Even if the coin is perfectly fair, we would **not**
                expect every experiment to produce exactly half Heads
                and half Tails.

                Try it yourself.
                """
            )

            n_flips = st.slider(
                "Choose the number of times to flip the fair coin:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="sample_size"
            )

            if (
                "random_n" in st.session_state
                and st.session_state.random_n != n_flips
            ):
                st.caption(
                    "👆 You changed the sample size. "
                    "Flip again to run the new experiment."
                )

            if st.button(
                "🪙 Flip the coin",
                key="flip_coin"
            ):

                heads = np.random.binomial(
                    n_flips,
                    0.5
                )

                tails = n_flips - heads

                st.session_state.random_heads = heads
                st.session_state.random_tails = tails
                st.session_state.random_n = n_flips

        if "random_heads" in st.session_state:

            heads = st.session_state.random_heads
            tails = st.session_state.random_tails
            n = st.session_state.random_n

            p_hat = heads / n

            se = np.sqrt(
                0.5 * (1 - 0.5) / n
            )

            with learning_section('Your random sample', 'step1_complete', 'Random samples naturally vary.'):
                st.write("### Your random sample")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Heads",
                        heads
                    )

                with col2:
                    st.metric(
                        "Tails",
                        tails
                    )

                with col3:
                    st.metric(
                        "Observed proportion",
                        f"{p_hat:.3f}"
                    )

                st.write(
                    f"""
                    This experiment produced **{heads} Heads**
                    and **{tails} Tails**.

                    The observed proportion was **{p_hat:.3f}**
                    instead of exactly **0.500**.
                    """
                )

                st.info(
                    """
                    Even though we simulated a perfectly fair coin,
                    the sample usually does not produce exactly 50% Heads.

                    **Random samples naturally vary.**
                    """
                )

            # ------------------------------------------------
            # Sampling distribution and SE
            # ------------------------------------------------

            with learning_section('Sampling distribution and standard error', 'step1_complete', 'Larger samples have less sampling variability.'):
                st.divider()

                st.subheader(
                    "Sampling Distribution & Standard Error"
                )

                st.write(
                    """
                    Imagine repeating this experiment many times.

                    Each experiment would produce a slightly different
                    observed proportion \(\hat{p}\).

                    The distribution of all these sample proportions
                    is called the **sampling distribution**.
                    """
                )

                st.write(
                    """
                    The **standard error** tells us how much these sample
                    proportions typically vary.
                    """
                )

                st.write(
                    "For a proportion:"
                )

                st.latex(
                    r"SE = \sqrt{\frac{p(1-p)}{n}}"
                )

                st.write(
                    """
                    Since we are simulating a **fair coin**, we use:
                    """
                )

                st.latex(
                    r"p = 0.5"
                )

                st.write(
                    f"For your chosen sample size, \(n={n}\):"
                )

                st.latex(
                    rf"SE = "
                    rf"\sqrt{{\frac{{0.5(1-0.5)}}{{{n}}}}}"
                    rf" = {se:.4f}"
                )

                st.metric(
                    "Standard Error",
                    f"{se:.4f}"
                )

                st.caption(
                    "Try changing the sample size and flipping again. "
                    "Pay attention to what happens to the standard error."
                )

            # ------------------------------------------------
            # Quick check
            # ------------------------------------------------

            with learning_section('Check your understanding', 'step1_complete', 'Larger n means smaller standard error.'):
                st.divider()

                st.subheader("Quick check")

                def reset_se_question():
                    st.session_state.se_checked = False
                    st.session_state.step1_complete = False

                se_question = st.radio(
                    "As the sample size increases, what happens to the standard error?",
                    [
                        "It increases",
                        "It decreases",
                        "It stays the same"
                    ],
                    index=None,
                    key="se_question",
                    on_change=reset_se_question
                )

                if st.button(
                    "Check",
                    key="check_se"
                ):

                    if se_question is not None:

                        st.session_state.se_checked = True

                        st.session_state.step1_complete = (
                            se_question == "It decreases"
                        )

                if st.session_state.get("se_checked"):

                    if st.session_state.get(
                        "step1_complete"
                    ):

                        st.success(
                            "✅ Correct! As the sample size increases, "
                            "the standard error decreases."
                        )

                        st.info(
                            """
                            Larger samples tend to give more stable estimates.

                            **Larger n → smaller SE → less sampling variability**
                            """
                        )

                    else:

                        st.error(
                            "Not quite. Try comparing the standard error "
                            "for a small sample with the standard error "
                            "for a much larger sample."
                        )

    navigation_buttons(
        show_continue=st.session_state.get(
            "step1_complete",
            False
        )
    )


def render_hypotheses_step():
    """Render step 2 and its answer checks."""

    st.title("Step 2 — Form Your Hypotheses")

    st.write(
        """
        The manufacturer claims that the coin is fair.

        A fair coin has a probability of Heads equal to:
        """
    )

    st.latex(
        r"p = 0.5"
    )

    # --------------------------------------------------------
    # H0
    # --------------------------------------------------------

    with learning_section('Null hypothesis', 'h0_correct', 'H₀: p = 0.5.'):
        st.subheader("Choose the null hypothesis")

        def reset_h0():
            reset_keys(
                "h0_checked",
                "h0_correct",
                "h1_checked",
                "h1_correct",
                "coffee_h1_checked",
                "coffee_h1_correct"
            )

        h0 = st.radio(
            "What should H₀ be?",
            [
                "p = 0.5",
                "p ≠ 0.5",
                "p > 0.5",
                "p < 0.5"
            ],
            index=None,
            key="h0_question",
            on_change=reset_h0
        )

        if st.button(
            "Check H₀",
            key="check_h0"
        ):

            if h0 is not None:

                st.session_state.h0_checked = True
                st.session_state.h0_correct = (
                    h0 == "p = 0.5"
                )

        if st.session_state.get("h0_checked"):

            if st.session_state.get("h0_correct"):

                st.success(
                    "✅ Correct! H₀ represents the default assumption: "
                    "the coin is fair."
                )

            else:

                st.error(
                    "Not quite. H₀ should represent the manufacturer's "
                    "claim that the coin is fair."
                )

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    if st.session_state.get("h0_correct"):

        with learning_section('Alternative hypothesis', 'h1_correct', 'H₁: p ≠ 0.5.'):
            st.subheader(
                "Now choose the alternative hypothesis"
            )

            def reset_h1():
                reset_keys(
                    "h1_checked",
                    "h1_correct",
                    "coffee_h1_checked",
                    "coffee_h1_correct"
                )

            h1 = st.radio(
                "What should H₁ be?",
                [
                    "p = 0.5",
                    "p ≠ 0.5",
                    "p > 0.5",
                    "p < 0.5"
                ],
                index=None,
                key="h1_question",
                on_change=reset_h1
            )

            if st.button(
                "Check H₁",
                key="check_h1"
            ):

                if h1 is not None:

                    st.session_state.h1_checked = True

                    st.session_state.h1_correct = (
                        h1 == "p ≠ 0.5"
                    )

            if st.session_state.get("h1_checked"):

                if st.session_state.get(
                    "h1_correct"
                ):

                    st.success(
                        "✅ Correct! We want to detect bias "
                        "in either direction."
                    )

                else:

                    st.error(
                        "Not quite. We are asking whether the coin "
                        "is biased toward Heads **or** Tails."
                    )

    # --------------------------------------------------------
    # Alternative directions
    # --------------------------------------------------------

    if (
        st.session_state.get("h0_correct")
        and st.session_state.get("h1_correct")
    ):

        with learning_section('Apply hypotheses to a new example', 'coffee_h1_correct', 'Suspecting less coffee means H₁: μ < 250.'):
            st.info(
                """
                Our hypotheses are:

                **H₀: p = 0.5**

                **H₁: p ≠ 0.5**

                H₀ says the coin is fair.

                H₁ says the coin is biased in either direction.
                """
            )

            st.divider()

            st.subheader(
                "The direction of H₁ depends on the question"
            )

            st.markdown(
                """
                **What are you trying to detect?**

                - Increase → $>$
                - Decrease → $<$
                - Any difference → $\\neq$
                """
            )

            st.subheader("Try another example")

            st.info(
                """
                A coffee machine should pour **250 ml on average**.

                You suspect that it actually pours **less**.
                """
            )

            def reset_coffee_h1():
                reset_keys(
                    "coffee_h1_checked",
                    "coffee_h1_correct"
                )

            coffee_h1 = st.radio(
                "Which alternative hypothesis is correct?",
                [
                    "H₁: μ = 250",
                    "H₁: μ ≠ 250",
                    "H₁: μ < 250",
                    "H₁: μ > 250"
                ],
                index=None,
                key="coffee_h1",
                on_change=reset_coffee_h1
            )

            if st.button(
                "Check coffee example",
                key="check_coffee_h1"
            ):

                if coffee_h1 is not None:

                    st.session_state.coffee_h1_checked = True

                    st.session_state.coffee_h1_correct = (
                        coffee_h1 == "H₁: μ < 250"
                    )

            if st.session_state.get(
                "coffee_h1_checked"
            ):

                if st.session_state.get(
                    "coffee_h1_correct"
                ):

                    st.success(
                        "✅ Correct! The question asks whether "
                        "the machine pours **less**, so H₁ uses <."
                    )

                    st.info(
                        """
                        **Takeaway**

                        H₀ = default/reference assumption

                        H₁ = effect or difference we are looking for
                        """
                    )

                else:

                    st.error(
                        "Not quite. Focus on the word **less** "
                        "in the question."
                    )

    navigation_buttons(
        show_continue=st.session_state.get(
            "coffee_h1_correct",
            False
        )
    )


def render_test_statistic_step():
    """Render Step 3: Explore and understand the Z-score."""

    st.title("Step 3 — How far is our result from H₀?")

    # --------------------------------------------------------
    # Introduction: Why do we need a Z-score?
    # --------------------------------------------------------

    st.write(
        """
        We flipped our coin **100 times** and observed **60 Heads**.

        Under our null hypothesis, the coin is fair:
        """
    )

    st.latex(r"H_0: p = 0.5")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Expected proportion", "0.50")

    with col2:
        st.metric("Observed proportion", "0.60")

    st.write(
        """
        The difference is **0.10 (10 percentage points)**.

        But is this a large difference, or could it simply
        happen because of random sampling variability?

        To answer this, we need to consider how much
        our samples naturally vary.
        """
    )

    # --------------------------------------------------------
    # 1. Interactive exploration
    # --------------------------------------------------------

    with learning_section(
        "Explore the Z-score",
        "z_exploration_correct",
        "The farther the observation is from H₀, the larger |Z| becomes."
    ):

        st.subheader("Explore what happens")

        st.write(
            """
            Move the slider to change the number of Heads.

            Try **50, 55, 60, and 65 Heads**.

            Watch how the Z-score and its position on the
            distribution change.
            """
        )

        heads = st.slider(
            "Number of Heads out of 100",
            min_value=35,
            max_value=65,
            value=60,
            key="z_heads"
        )

        p0 = 0.5
        n = 100
        p_hat = heads / n
        se = np.sqrt(p0 * (1 - p0) / n)
        z = (p_hat - p0) / se

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Heads", heads)

        with col2:
            st.metric("Observed proportion", f"{p_hat:.2f}")

        with col3:
            st.metric("Z-score", f"{z:.2f}")

        # Normal distribution visualization

        x = np.linspace(-4, 4, 500)
        y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

        fig, ax = plt.subplots()

        ax.plot(
            x, y,
            color="steelblue",
            linewidth=2,
            label="Distribution under H₀"
        )

        ax.axvline(
            0,
            color="gray",
            linestyle="--",
            linewidth=2,
            label="H₀ prediction (Z = 0)"
        )

        ax.axvline(
            z,
            color="darkorange",
            linewidth=2,
            label=f"Your result (Z = {z:.2f})"
        )

        ax.scatter(
            [z],
            [(1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)],
            color="darkorange",
            s=70,
            zorder=5
        )

        ax.set_xlabel("Z-score")
        ax.set_ylabel("Density")
        ax.set_xlim(-4, 4)
        ax.set_title("Where does your result fall?")
        ax.legend(fontsize=8)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "The dashed line represents what H₀ predicts. "
            "The orange line represents your observed result."
        )

        # ----------------------------------------------------
        # 2. Prediction question
        # ----------------------------------------------------

        st.divider()
        st.subheader("What did you notice?")

        def reset_z_exploration():
            reset_keys(
                "z_exploration_checked",
                "z_exploration_correct"
            )

        exploration_answer = st.radio(
            "What happens as the observed proportion moves farther from 0.50?",
            [
                "The Z-score always becomes positive",
                "The absolute Z-score increases",
                "The Z-score moves closer to zero"
            ],
            index=None,
            key="z_exploration_answer",
            on_change=reset_z_exploration
        )

        if st.button(
            "Check observation",
            key="check_z_exploration"
        ):
            if exploration_answer is not None:
                st.session_state.z_exploration_checked = True
                st.session_state.z_exploration_correct = (
                    exploration_answer == "The absolute Z-score increases"
                )

        if st.session_state.get("z_exploration_checked"):

            if st.session_state.get("z_exploration_correct"):

                st.success(
                    "✅ Correct! The farther the observation is "
                    "from what H₀ predicts, the larger its "
                    "absolute Z-score becomes."
                )

                st.info(
                    "The sign tells us the **direction**, "
                    "while the absolute value tells us the **distance**."
                )

            else:
                st.error(
                    "Not quite. Try moving the slider both "
                    "above and below 50 Heads. "
                    "Pay attention to the distance from zero."
                )

    # --------------------------------------------------------
    # 3. Explain the Z-score
    # --------------------------------------------------------

    if st.session_state.get("z_exploration_correct"):

        with learning_section(
            "Understand the Z-score",
            "z_explanation_complete",
            "Z measures distance from H₀ in standard errors."
        ):

            st.subheader("What is a Z-score?")

            st.write(
                """
                The **Z-score** measures how far our observed
                result is from what H₀ predicts, taking normal
                sampling variability into account.

                It expresses this distance in units of
                **standard errors**.
                """
            )

            st.latex(
                r"Z = \frac{\text{Observed} - "
                r"\text{Expected under }H_0}"
                r"{\text{Standard Error}}"
            )

            st.write(
                """
                For our original experiment, we observed
                **60 Heads out of 100**.
                """
            )

            st.latex(
                r"SE = \sqrt{\frac{0.5(1-0.5)}{100}} = 0.05"
            )

            st.latex(
                r"Z = \frac{0.60-0.50}{0.05} = 2"
            )

            st.info(
                """
                **Z = 2** means our observed proportion
                is **2 standard errors above** what H₀ predicts.

                The Z-score has **no unit**.
                """
            )

            st.subheader("Understanding the Z-score scale")

            st.markdown("""
- **Z = 0** → exactly what H₀ predicts
- **|Z| ≈ 1** → common
- **|Z| ≈ 2** → unusual
- **|Z| ≈ 3** → very unusual
""")

            st.caption(
                "These are intuition guidelines, not formal decision thresholds."
            )

            st.write("### Positive vs Negative Z")

            st.write(
                """
                - **Positive Z:** the observation is above
                  the value predicted by H₀.
                - **Negative Z:** the observation is below
                  the value predicted by H₀.

                The larger **|Z|** is, the farther our
                observation is from the null hypothesis.
                """
            )

            if st.button(
                "I understand — continue to practice →",
                key="z_explanation_done"
            ):
                st.session_state.z_explanation_complete = True

    # --------------------------------------------------------
    # 4. Independent transfer exercise
    # --------------------------------------------------------

    if st.session_state.get("z_explanation_complete"):

        with learning_section(
            "Try a Z-score calculation",
            "z_transfer_correct",
            "(490 − 500) / 5 = −2."
        ):

            st.subheader("Quick transfer check")

            st.info(
                """
                Suppose:

                **H₀: μ = 500**

                Observed sample mean: **490**

                Standard error: **5**
                """
            )

            st.write(
                "Try calculating the Z-score on your own."
            )

            with st.expander("Need a hint?"):
                st.write(
                    "Subtract the value predicted by H₀ "
                    "from the observed value, then divide "
                    "by the standard error."
                )

                st.latex(
                    r"Z = \frac{\text{Observed} - "
                    r"\text{Expected}}{SE}"
                )

            def reset_z_transfer():
                reset_keys(
                    "z_transfer_checked",
                    "z_transfer_correct"
                )

            z_transfer = st.radio(
                "What is the Z-score?",
                [
                    "Z = -2",
                    "Z = -1",
                    "Z = 1",
                    "Z = 2"
                ],
                index=None,
                key="z_transfer",
                on_change=reset_z_transfer
            )

            if st.button(
                "Check Z-score",
                key="check_z_transfer"
            ):

                if z_transfer is not None:

                    st.session_state.z_transfer_checked = True

                    st.session_state.z_transfer_correct = (
                        z_transfer == "Z = -2"
                    )

            if st.session_state.get("z_transfer_checked"):

                if st.session_state.get("z_transfer_correct"):

                    st.success(
                        "✅ Correct! The observed mean is "
                        "2 standard errors below H₀."
                    )

                    st.latex(
                        r"Z = \frac{490-500}{5} = -2"
                    )

                else:

                    st.error(
                        "Not quite. Think about the direction "
                        "of the difference, then calculate "
                        "(490 - 500) / 5."
                    )

    # --------------------------------------------------------
    # 5. Transition to the p-value
    # --------------------------------------------------------

    if st.session_state.get("z_transfer_correct"):

        st.divider()

        st.subheader("What's next?")

        st.write(
            """
            We now know that our original coin experiment
            gave **Z = 2**.

            This means our result is 2 standard errors
            away from what H₀ predicts.

            But **how unlikely is a result this extreme
            if the coin is actually fair?**

            That's what we'll investigate next using
            the **p-value**.
            """
        )

    navigation_buttons(
        show_continue=st.session_state.get(
            "z_transfer_correct",
            False
        )
    )


def render_pvalue_step():
    """Render step 4 and its answer checks."""

    st.title(
        "Step 4 — How surprising is this result under H₀?"
    )

    st.write(
        """
        In the previous step, we measured how far our result
        was from H₀ using the Z-score.

        Now we ask:

        **If H₀ were true, how likely would it be to observe
        a result this extreme or even more extreme?**
        """
    )

    st.latex(
        r"H_0: p = 0.5"
    )

    st.latex(
        r"H_1: p \neq 0.5"
    )

    st.info(
        """
        Because H₁ uses **≠**, we care about unusual results
        in **both directions**.

        This means we have a **two-sided test**.
        """
    )

    heads = 60
    p_hat = 0.60
    p0 = 0.5
    n = 100

    se = np.sqrt(
        p0 * (1 - p0) / n
    )

    z = (
        p_hat - p0
    ) / se

    st.subheader("Our observed result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Z-score",
            f"{z:.2f}"
        )

    with col2:
        st.metric(
            "Observed proportion",
            f"{p_hat:.2f}"
        )

    # --------------------------------------------------------
    # Misconception check
    # --------------------------------------------------------

    with learning_section('What does a p-value mean?', 'pvalue_interpretation_correct', 'The p-value assumes H₀ is true.'):
        st.subheader(
            "Before we calculate it..."
        )

        def reset_p_interpretation():
            reset_keys(
                "p_interpretation_checked",
                "pvalue_interpretation_correct",
                "coffee_z_checked",
                "coffee_z_correct",
                "coffee_tail_checked",
                "coffee_tail_correct",
                "coffee_p_checked",
                "coffee_p_correct",
                "pvalue_reflection_checked",
                "pvalue_reflection_correct"
            )

        interpretation = st.radio(
            "What does a p-value tell us?",
            [
                "The probability that H₀ is true",
                "The probability that the result happened by chance",
                (
                    "Assuming H₀ is true, the probability of observing "
                    "a result this extreme or more extreme"
                ),
                "The probability that H₁ is true"
            ],
            index=None,
            key="pvalue_interpretation",
            on_change=reset_p_interpretation
        )

        if st.button(
            "Check interpretation",
            key="check_p_interpretation"
        ):

            if interpretation is not None:

                st.session_state.p_interpretation_checked = True

                st.session_state.pvalue_interpretation_correct = (
                    interpretation.startswith(
                        "Assuming H₀ is true"
                    )
                )

        if st.session_state.get(
            "p_interpretation_checked"
        ):

            if st.session_state.get(
                "pvalue_interpretation_correct"
            ):

                st.success(
                    "✅ Correct! The p-value assumes H₀ is true "
                    "and measures how unusual our result would be."
                )

            else:

                st.error(
                    "Not quite. The p-value does not tell us "
                    "the probability that H₀ or H₁ is true."
                )

    # --------------------------------------------------------
    # Coin p-value visualization
    # --------------------------------------------------------

    if st.session_state.get(
        "pvalue_interpretation_correct"
    ):

        with learning_section('Visualize the two-sided p-value', 'coffee_z_correct', 'For Z = 2, the two tails give p ≈ 0.0455.'):
            st.subheader(
                "Visualizing the p-value"
            )

            x = np.linspace(
                -4,
                4,
                500
            )

            y = (
                1 / np.sqrt(2 * np.pi)
            ) * np.exp(
                -0.5 * x ** 2
            )

            fig, ax = plt.subplots()

            ax.plot(
                x,
                y
            )

            ax.axvline(
                z,
                linestyle="--"
            )

            ax.axvline(
                -z,
                linestyle="--"
            )

            left_tail = (
                x <= -abs(z)
            )

            right_tail = (
                x >= abs(z)
            )

            ax.fill_between(
                x[left_tail],
                y[left_tail],
                alpha=0.4
            )

            ax.fill_between(
                x[right_tail],
                y[right_tail],
                alpha=0.4
            )

            ax.set_xlim(
                -4,
                4
            )

            ax.set_xlabel(
                "Z-score"
            )

            ax.set_ylabel(
                "Density"
            )

            ax.set_title(
                "Results this extreme or more extreme"
            )

            st.pyplot(fig)

            plt.close(fig)

            st.write(
                """
                The shaded regions represent outcomes that are at least
                as extreme as our observed result.

                Since this is a two-sided test, we include **both tails**.
                """
            )

            p_value = (
                2 * (
                    1 - normal_cdf(abs(z))
                )
            )

            st.metric(
                "p-value",
                f"{p_value:.4f}"
            )

            st.write(
                f"""
                If the coin were truly fair, a result this extreme
                or more extreme would occur about
                **{p_value * 100:.1f}% of the time**.
                """
            )

        # ----------------------------------------------------
        # Coffee machine exercise
        # ----------------------------------------------------

        with learning_section('Calculate the coffee machine Z-score', 'coffee_z_correct', '(244 − 250) / 2 = −3.'):
            st.divider()

            st.subheader(
                "Now try a different example"
            )

            st.info(
                """
                A coffee machine should pour **250 ml on average**.

                You suspect it pours **less**.

                From a sample:

                **x̄ = 244 ml**

                **SE = 2 ml**
                """
            )

            def reset_coffee_z():
                reset_keys(
                    "coffee_z_checked",
                    "coffee_z_correct",
                    "coffee_tail_checked",
                    "coffee_tail_correct",
                    "coffee_p_checked",
                    "coffee_p_correct",
                    "pvalue_reflection_checked",
                    "pvalue_reflection_correct"
                )

            coffee_z = st.radio(
                "What is the test statistic?",
                [
                    "Z = -1",
                    "Z = -2",
                    "Z = -3",
                    "Z = 3"
                ],
                index=None,
                key="coffee_z",
                on_change=reset_coffee_z
            )

            if st.button(
                "Check Z",
                key="check_coffee_z"
            ):

                if coffee_z is not None:

                    st.session_state.coffee_z_checked = True

                    st.session_state.coffee_z_correct = (
                        coffee_z == "Z = -3"
                    )

            if st.session_state.get(
                "coffee_z_checked"
            ):

                if st.session_state.get(
                    "coffee_z_correct"
                ):

                    st.success(
                        "✅ Correct!"
                    )

                    st.latex(
                        r"Z = \frac{244-250}{2} = -3"
                    )

                else:

                    st.error(
                        "Not quite. Calculate (244 - 250) / 2."
                    )

        # ----------------------------------------------------
        # Tail
        # ----------------------------------------------------

        if st.session_state.get(
            "coffee_z_correct"
        ):

            with learning_section('Choose the correct tail', 'coffee_tail_correct', 'H₁: μ < 250 uses the left tail.'):
                def reset_coffee_tail():
                    reset_keys(
                        "coffee_tail_checked",
                        "coffee_tail_correct",
                        "coffee_p_checked",
                        "coffee_p_correct",
                        "pvalue_reflection_checked",
                        "pvalue_reflection_correct"
                    )

                tail_answer = st.radio(
                    "Since H₁: μ < 250, which probability gives the p-value?",
                    [
                        "P(Z ≥ -3)",
                        "P(Z ≤ -3)",
                        "2P(Z ≤ -3)",
                        "P(-3 ≤ Z ≤ 3)"
                    ],
                    index=None,
                    key="coffee_tail",
                    on_change=reset_coffee_tail
                )

                if st.button(
                    "Check tail",
                    key="check_coffee_tail"
                ):

                    if tail_answer is not None:

                        st.session_state.coffee_tail_checked = True

                        st.session_state.coffee_tail_correct = (
                            tail_answer == "P(Z ≤ -3)"
                        )

                if st.session_state.get(
                    "coffee_tail_checked"
                ):

                    if st.session_state.get(
                        "coffee_tail_correct"
                    ):

                        st.success(
                            "✅ Correct! Because H₁ uses <, "
                            "this is a left-tailed test."
                        )

                    else:

                        st.error(
                            "Not quite. H₁ uses <, so we look "
                            "toward the left tail."
                        )

        # ----------------------------------------------------
        # Coffee p-value interpretation
        # ----------------------------------------------------

        if st.session_state.get(
            "coffee_tail_correct"
        ):

            with learning_section('Interpret the coffee machine p-value', 'coffee_p_correct', 'Under H₀, a mean this low or lower is very unusual.'):
                st.latex(
                    r"p = P(Z \leq -3) \approx 0.00135"
                )

                def reset_coffee_p():
                    reset_keys(
                        "coffee_p_checked",
                        "coffee_p_correct",
                        "pvalue_reflection_checked",
                        "pvalue_reflection_correct"
                    )

                coffee_interpretation = st.radio(
                    "What does p ≈ 0.00135 mean?",
                    [
                        (
                            "There is a 0.135% probability that "
                            "the machine really pours 250 ml"
                        ),
                        (
                            "There is a 99.865% probability that "
                            "the machine pours less than 250 ml"
                        ),
                        (
                            "Assuming H₀ is true, there is about a 0.135% "
                            "chance of observing a sample mean of 244 ml "
                            "or lower"
                        ),
                        (
                            "There is a 0.135% probability that the "
                            "experiment was caused by chance"
                        )
                    ],
                    index=None,
                    key="coffee_p_interpretation",
                    on_change=reset_coffee_p
                )

                if st.button(
                    "Check p-value interpretation",
                    key="check_coffee_p"
                ):

                    if coffee_interpretation is not None:

                        st.session_state.coffee_p_checked = True

                        st.session_state.coffee_p_correct = (
                            coffee_interpretation.startswith(
                                "Assuming H₀"
                            )
                        )

                if st.session_state.get(
                    "coffee_p_checked"
                ):

                    if st.session_state.get(
                        "coffee_p_correct"
                    ):

                        st.success(
                            "✅ Correct! Assuming H₀ is true, "
                            "observing a result this low or lower "
                            "would be extremely unusual."
                        )

                    else:

                        st.error(
                            "Not quite. Remember that p-values are "
                            "interpreted assuming H₀ is true."
                        )

        # ----------------------------------------------------
        # Final reflection
        # ----------------------------------------------------

        if st.session_state.get(
            "coffee_p_correct"
        ):

            with learning_section('Final reflection', 'pvalue_reflection_correct', 'Farther from H₀ means a smaller p-value.'):
                st.subheader("Final reflection")

                def reset_p_reflection():
                    reset_keys(
                        "pvalue_reflection_checked",
                        "pvalue_reflection_correct"
                    )

                reflection = st.radio(
                    "As the observed result moves farther away from H₀, what happens to the p-value?",
                    [
                        "It increases",
                        "It decreases",
                        "It stays the same"
                    ],
                    index=None,
                    key="pvalue_reflection",
                    on_change=reset_p_reflection
                )

                if st.button(
                    "Check reflection",
                    key="check_pvalue_reflection"
                ):

                    if reflection is not None:

                        st.session_state.pvalue_reflection_checked = True

                        st.session_state.pvalue_reflection_correct = (
                            reflection == "It decreases"
                        )

                if st.session_state.get(
                    "pvalue_reflection_checked"
                ):

                    if st.session_state.get(
                        "pvalue_reflection_correct"
                    ):

                        st.success(
                            "✅ Correct!"
                        )

                        st.latex(
                            r"|Z| \uparrow "
                            r"\Rightarrow "
                            r"p\text{-value} \downarrow"
                        )

                    else:

                        st.error(
                            "Not quite. Think about what happens when "
                            "an observation becomes increasingly unusual."
                        )

    navigation_buttons(
        show_continue=st.session_state.get(
            "pvalue_reflection_correct",
            False
        )
    )


def render_significance_step():
    """Render step 5 and its answer checks."""

    st.title(
        "Step 5 — Is the p-value small enough?"
    )

    st.write(
        """
        We obtained:
        """
    )

    p_value = 0.046
    alpha = 0.05

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "p-value",
            f"{p_value:.3f}"
        )

    with col2:
        st.metric(
            "α",
            f"{alpha:.2f}"
        )

    st.write(
        """
        The significance level **α** is the threshold we use
        to decide whether the evidence against H₀ is strong enough.
        """
    )

    st.latex(
        r"p \leq \alpha "
        r"\Rightarrow "
        r"\text{Reject } H_0"
    )

    st.latex(
        r"p > \alpha "
        r"\Rightarrow "
        r"\text{Fail to reject } H_0"
    )

    # --------------------------------------------------------
    # Coin decision
    # --------------------------------------------------------

    with learning_section('Make the coin decision', 'decision_correct', '0.046 < 0.05: reject H₀.'):
        def reset_decision():
            reset_keys(
                "decision_checked",
                "decision_correct",
                "decision2_checked",
                "decision2_correct"
            )

        decision = st.radio(
            "What should we conclude?",
            [
                "Accept H₀",
                "Reject H₀",
                "Fail to reject H₀",
                "There is not enough information"
            ],
            index=None,
            key="decision_question",
            on_change=reset_decision
        )

        if st.button(
            "Check decision",
            key="check_decision"
        ):

            if decision is not None:

                st.session_state.decision_checked = True

                st.session_state.decision_correct = (
                    decision == "Reject H₀"
                )

        if st.session_state.get(
            "decision_checked"
        ):

            if st.session_state.get(
                "decision_correct"
            ):

                st.success(
                    "✅ Correct! Since p = 0.046 is smaller "
                    "than α = 0.05, we reject H₀."
                )

            else:

                st.error(
                    "Not quite. Compare the p-value directly with α."
                )

    if st.session_state.get(
        "decision_correct"
    ):

        with learning_section('What rejection means', 'decision2_correct', 'Significant evidence is not absolute certainty.'):
            st.latex(
                r"0.046 < 0.05"
            )

            st.success(
                "Reject H₀"
            )

            st.write(
                """
                The data provide **statistically significant evidence**
                against the assumption that the coin is fair.
                """
            )

            st.warning(
                """
                Be careful:

                We should **not** say:

                **"The coin is definitely biased."**

                A better conclusion is:

                **"The data provide sufficient evidence against the
                assumption that the coin is fair."**
                """
            )

        # ----------------------------------------------------
        # Fail-to-reject example
        # ----------------------------------------------------

        with learning_section('Try the opposite case', 'decision2_correct', '0.12 > 0.05: fail to reject H₀.'):
            st.divider()

            st.subheader(
                "What about the opposite case?"
            )

            st.write(
                "Suppose another experiment gives:"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "p-value",
                    "0.12"
                )

            with col2:
                st.metric(
                    "α",
                    "0.05"
                )

            def reset_decision2():
                reset_keys(
                    "decision2_checked",
                    "decision2_correct"
                )

            decision2 = st.radio(
                "What should we conclude now?",
                [
                    "Reject H₀",
                    "Fail to reject H₀",
                    "Accept H₀"
                ],
                index=None,
                key="decision2",
                on_change=reset_decision2
            )

            if st.button(
                "Check second decision",
                key="check_decision2"
            ):

                if decision2 is not None:

                    st.session_state.decision2_checked = True

                    st.session_state.decision2_correct = (
                        decision2 == "Fail to reject H₀"
                    )

            if st.session_state.get(
                "decision2_checked"
            ):

                if st.session_state.get(
                    "decision2_correct"
                ):

                    st.success(
                        "✅ Correct! Since 0.12 > 0.05, "
                        "we fail to reject H₀."
                    )

                    st.warning(
                        """
                        **Failing to reject H₀ does NOT mean that H₀
                        has been proven true.**

                        It only means we do not have strong enough
                        evidence against it.
                        """
                    )

                else:

                    st.error(
                        "Not quite. Compare 0.12 with 0.05."
                    )

    navigation_buttons(
        show_continue=st.session_state.get(
            "decision2_correct",
            False
        )
    )


def render_errors_step():
    """Render step 6 and its answer checks."""

    st.title(
        "Step 6 — What if our decision is wrong?"
    )

    st.write(
        """
        Hypothesis testing gives us a decision based on evidence.

        But sometimes that decision can be wrong.
        """
    )

    # --------------------------------------------------------
    # Scenario 1
    # --------------------------------------------------------

    with learning_section('Scenario 1: a false positive', 'type1_correct', 'Rejecting a true H₀ is a Type I error.'):
        st.subheader("Scenario 1")

        st.info(
            """
            Suppose the coin is actually **fair**.

            But our hypothesis test concludes that the coin is **biased**.
            """
        )

        def reset_type1():
            reset_keys(
                "type1_checked",
                "type1_correct",
                "type2_checked",
                "type2_correct",
                "med_error_checked",
                "med_error_correct"
            )

        error1 = st.radio(
            "What happened?",
            [
                "Type I error",
                "Type II error",
                "No error"
            ],
            index=None,
            key="type1_question",
            on_change=reset_type1
        )

        if st.button(
            "Check Scenario 1",
            key="check_type1"
        ):

            if error1 is not None:

                st.session_state.type1_checked = True

                st.session_state.type1_correct = (
                    error1 == "Type I error"
                )

        if st.session_state.get(
            "type1_checked"
        ):

            if st.session_state.get(
                "type1_correct"
            ):

                st.success(
                    "✅ Correct! We rejected H₀ even though "
                    "H₀ was actually true."
                )

            else:

                st.error(
                    "Not quite. Remember that H₀ says the coin is fair."
                )

    if st.session_state.get(
        "type1_correct"
    ):

        with learning_section('Type I error explained', 'type2_correct', 'α controls the probability of a Type I error.'):
            st.write("### Type I Error")

            st.latex(
                r"\text{Type I error} = "
                r"\text{Reject a true } H_0"
            )

            st.write(
                """
                This is also called a **false positive**.
                """
            )

            st.info(
                """
                The probability of making a Type I error
                is controlled by **α**.
                """
            )

        # ----------------------------------------------------
        # Scenario 2
        # ----------------------------------------------------

        with learning_section('Scenario 2: a missed effect', 'type2_correct', 'Failing to reject a false H₀ is a Type II error.'):
            st.divider()

            st.subheader("Scenario 2")

            st.info(
                """
                Now suppose the coin is actually **biased**.

                But our experiment does **not** provide enough evidence
                to reject H₀.
                """
            )

            def reset_type2():
                reset_keys(
                    "type2_checked",
                    "type2_correct",
                    "med_error_checked",
                    "med_error_correct"
                )

            error2 = st.radio(
                "What happened this time?",
                [
                    "Type I error",
                    "Type II error",
                    "No error"
                ],
                index=None,
                key="type2_question",
                on_change=reset_type2
            )

            if st.button(
                "Check Scenario 2",
                key="check_type2"
            ):

                if error2 is not None:

                    st.session_state.type2_checked = True

                    st.session_state.type2_correct = (
                        error2 == "Type II error"
                    )

            if st.session_state.get(
                "type2_checked"
            ):

                if st.session_state.get(
                    "type2_correct"
                ):

                    st.success(
                        "✅ Correct! H₀ was false, "
                        "but we failed to reject it."
                    )

                else:

                    st.error(
                        "Not quite. The coin really is biased, "
                        "but the test failed to detect it."
                    )

    if st.session_state.get(
        "type2_correct"
    ):

        with learning_section('Type II error and summary', 'med_error_correct', 'A false negative misses a real effect.'):
            st.write("### Type II Error")

            st.latex(
                r"\text{Type II error} = "
                r"\text{Fail to reject a false } H_0"
            )

            st.write(
                """
                This is also called a **false negative**.

                A real effect exists, but our test fails to detect it.
                """
            )

            st.subheader("Quick summary")

            st.markdown(
                """
                | Reality | Decision | Result |
                |---|---|---|
                | H₀ is true | Reject H₀ | **Type I error** |
                | H₀ is true | Fail to reject H₀ | Correct decision |
                | H₀ is false | Reject H₀ | Correct decision |
                | H₀ is false | Fail to reject H₀ | **Type II error** |
                """
            )

        # ----------------------------------------------------
        # Transfer example
        # ----------------------------------------------------

        with learning_section('Apply errors to a medication test', 'med_error_correct', 'Detecting an effect that does not exist is a Type I error.'):
            st.divider()

            st.subheader("Transfer example")

            st.info(
                """
                Suppose we test whether a new medication works.

                The medication actually **does not work**,
                but the test concludes that it **does work**.
                """
            )

            def reset_med():
                reset_keys(
                    "med_error_checked",
                    "med_error_correct"
                )

            med_error = st.radio(
                "What type of error is this?",
                [
                    "Type I error",
                    "Type II error"
                ],
                index=None,
                key="med_error",
                on_change=reset_med
            )

            if st.button(
                "Check medication example",
                key="check_med_error"
            ):

                if med_error is not None:

                    st.session_state.med_error_checked = True

                    st.session_state.med_error_correct = (
                        med_error == "Type I error"
                    )

            if st.session_state.get(
                "med_error_checked"
            ):

                if st.session_state.get(
                    "med_error_correct"
                ):

                    st.success(
                        "✅ Correct! We concluded that an effect "
                        "exists when it actually does not."
                    )

                else:

                    st.error(
                        "Not quite. Think about whether this is "
                        "a false positive or a false negative."
                    )

    navigation_buttons(
        show_continue=st.session_state.get(
            "med_error_correct",
            False
        )
    )


def render_power_step():
    """Render step 7 and its answer checks."""

    st.title(
        "Step 7 — Can our test detect a real effect?"
    )

    st.write(
        """
        Suppose the coin is actually biased:

        **p = 0.60**

        But our experiment fails to detect that bias.
        """
    )

    with learning_section('Check the missed-effect scenario', 'power_warmup_correct', 'Missing a real effect is a Type II error.'):
        def reset_power_warmup():
            reset_keys(
                "power_warmup_checked",
                "power_warmup_correct"
            )

        warmup = st.radio(
            "What type of error occurred?",
            [
                "Type I error",
                "Type II error"
            ],
            index=None,
            key="power_warmup",
            on_change=reset_power_warmup
        )

        if st.button(
            "Check",
            key="check_power_warmup"
        ):

            if warmup is not None:

                st.session_state.power_warmup_checked = True

                st.session_state.power_warmup_correct = (
                    warmup == "Type II error"
                )

        if st.session_state.get(
            "power_warmup_checked"
        ):

            if st.session_state.get(
                "power_warmup_correct"
            ):

                st.success(
                    "✅ Correct! Failing to detect a real effect "
                    "is a Type II error."
                )

            else:

                st.error(
                    "Not quite. A real effect exists, but we failed to detect it."
                )

    if st.session_state.get(
        "power_warmup_correct"
    ):

        st.subheader("Statistical Power")

        st.latex(
            r"Power = 1 - \beta"
        )

        st.write(
            """
            where \(\beta\) is the probability of making a Type II error.

            **High power means a high probability of detecting
            a real effect.**
            """
        )

        st.divider()

        st.subheader("Explore the effect of sample size")

        st.write(
            """
            Suppose the real probability of Heads is **0.60**,
            while H₀ predicts **0.50**.

            Change the sample size below.
            """
        )

        n_power = st.slider(
            "Sample size n:",
            min_value=10,
            max_value=1000,
            value=50,
            step=10,
            key="power_n"
        )

        se_power = np.sqrt(
            0.5 * 0.5 / n_power
        )

        z_effect = (
            0.60 - 0.50
        ) / se_power

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Standard Error",
                f"{se_power:.4f}"
            )

        with col2:
            st.metric(
                "Difference in SE units",
                f"{z_effect:.2f}"
            )

        if n_power < 50:

            st.warning(
                "With a small sample, random variation is large. "
                "The bias may be difficult to detect."
            )

        elif n_power < 300:

            st.info(
                "The difference is becoming easier to distinguish "
                "from random variation."
            )

        else:

            st.success(
                "With a large sample, the difference between "
                "p = 0.60 and p = 0.50 is much easier to detect."
            )

        st.latex(
            r"n \uparrow \Rightarrow Power \uparrow"
        )

        st.write(
            """
            Power also generally increases when:

            - **Effect size increases**
            - **Variability decreases**
            """
        )

        st.info(
            """
            **Main idea**

            High statistical power means that if a real effect exists,
            the test has a good chance of detecting it.
            """
        )

    navigation_buttons(
        show_continue=st.session_state.get(
            "power_warmup_correct",
            False
        )
    )


def render_final_challenge():
    """Render step 8 and its answer checks."""

    st.title(
        "Step 8 — Final Challenge"
    )

    st.write(
        """
        You've now seen all the pieces of a hypothesis test.

        First, remember the reasoning process:
        """
    )

    st.info(
        """
        **Question → H₀/H₁ → Standard Error → Z-score →
        p-value → compare with α → decision → interpretation**
        """
    )

    st.write(
        """
        Now the scaffolding is removed.

        Solve the following new problem from start to finish.
        """
    )

    st.divider()

    st.header(
        "🚴 Delivery Time Investigation"
    )

    st.info(
        """
        A food delivery company claims that its average delivery time
        is **30 minutes**.

        Some customers believe that deliveries are actually taking
        **longer**.

        A random sample of **64 deliveries** has:

        **x̄ = 32 minutes**

        The population standard deviation is known to be:

        **σ = 8 minutes**

        Use:

        **α = 0.05**

        Determine whether the data provide sufficient evidence
        that the true average delivery time is greater than
        30 minutes.

        Explain your reasoning completely.
        """
    )

    st.subheader("Your workspace")

    student_work = st.text_area(
        "Solve the problem from start to finish:",
        height=300,
        key="final_work",
        placeholder=(
            "Write your hypotheses, calculations, decision, "
            "and interpretation here..."
        )
    )

    # --------------------------------------------------------
    # Progressive hints
    # --------------------------------------------------------

    if "hint_number" not in st.session_state:
        st.session_state.hint_number = 0

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "💡 Need a hint",
            width="stretch"
        ):

            if st.session_state.hint_number < 4:
                st.session_state.hint_number += 1

    hints = [
        (
            "What claim should be treated as the "
            "default assumption?"
        ),
        (
            "Think about whether the alternative "
            "should use >, <, or ≠."
        ),
        (
            "Once you have the hypotheses, determine "
            "how far the observed sample mean is from H₀."
        ),
        (
            "For a sample mean with known population "
            "standard deviation: SE = σ / √n"
        )
    ]

    for i in range(
        st.session_state.hint_number
    ):
        st.info(
            f"💡 Hint {i + 1}: {hints[i]}"
        )

    # --------------------------------------------------------
    # Prototype "Check my work"
    # --------------------------------------------------------

    with col2:

        check_work = st.button(
            "✓ Check my work",
            type="primary",
            width="stretch"
        )

    if check_work:

        if not student_work.strip():

            st.warning(
                "Write some reasoning first so that your work can be checked."
            )

        else:

            work = (
                student_work
                .lower()
                .replace(" ", "")
                .replace("μ", "mu")
                .replace("α", "alpha")
            )

            st.subheader("Feedback")

            # H0
            if (
                "h0:mu=30" in work
                or "h0=mu=30" in work
                or "h0:mean=30" in work
            ):
                st.success(
                    "✅ Your null hypothesis looks correct."
                )
            else:
                st.info(
                    "🔎 Check your null hypothesis. "
                    "What value represents the company's original claim?"
                )

            # H1
            if (
                "h1:mu>30" in work
                or "h1=mu>30" in work
                or "h1:mean>30" in work
            ):
                st.success(
                    "✅ Your alternative hypothesis has "
                    "the correct direction."
                )

            elif (
                "mu!=30" in work
                or "mu≠30" in work
                or "mu<30" in work
            ):
                st.info(
                    "🔎 Reconsider the direction of H₁. "
                    "The question specifically asks whether delivery "
                    "times are **greater than 30 minutes**."
                )

            else:
                st.info(
                    "🔎 Make sure you state an alternative hypothesis."
                )

            # SE
            if (
                "se=1" in work
                or "standarderror=1" in work
            ):
                st.success(
                    "✅ Your standard error appears to be correct."
                )
            else:
                st.info(
                    "🔎 Check the standard error using "
                    "**SE = σ / √n**."
                )

            # Z
            if (
                "z=2" in work
                or "z-score=2" in work
                or "zscore=2" in work
            ):
                st.success(
                    "✅ Your test statistic appears to be correct."
                )
            else:
                st.info(
                    "🔎 Once you have the standard error, calculate "
                    "how many standard errors 32 is above 30."
                )

            # p
            if (
                "0.0228" in work
                or "0.023" in work
                or "p=.0228" in work
            ):
                st.success(
                    "✅ Your p-value appears to be correct."
                )
            else:
                st.info(
                    "🔎 Check whether this is a left-tailed, "
                    "right-tailed, or two-sided test."
                )

            # decision
            if (
                "rejecth0" in work
                or "rejectthenull" in work
            ):
                st.success(
                    "✅ Your statistical decision appears correct."
                )
            else:
                st.info(
                    "🔎 Compare the p-value with α = 0.05."
                )

            # contextual conclusion
            if (
                "greaterthan30" in work
                or "longerthan30" in work
            ):
                st.success(
                    "✅ You appear to have interpreted the result "
                    "in the context of delivery times."
                )
            else:
                st.info(
                    "🔎 Finish with a conclusion about the "
                    "**true average delivery time**, not just "
                    "'reject H₀'."
                )

            st.caption(
                "For this prototype, the feedback is rule-based. "
                "This can later be replaced with AI-generated feedback."
            )

    # --------------------------------------------------------
    # Full solution
    # --------------------------------------------------------

    st.divider()

    with st.expander(
        "Show full solution"
    ):

        st.warning(
            "Try solving the problem yourself before opening the solution."
        )

        st.write("### 1. Hypotheses")

        st.latex(
            r"H_0: \mu = 30"
        )

        st.latex(
            r"H_1: \mu > 30"
        )

        st.write(
            "### 2. Standard Error"
        )

        st.latex(
            r"SE = \frac{\sigma}{\sqrt{n}}"
        )

        st.latex(
            r"SE = \frac{8}{\sqrt{64}} = 1"
        )

        st.write(
            "### 3. Test Statistic"
        )

        st.latex(
            r"Z = \frac{32-30}{1} = 2"
        )

        st.write(
            "### 4. p-value"
        )

        st.write(
            "Because \(H_1: \mu > 30\), this is a right-tailed test."
        )

        st.latex(
            r"p = P(Z \geq 2) \approx 0.0228"
        )

        st.write(
            "### 5. Decision"
        )

        st.latex(
            r"0.0228 < 0.05"
        )

        st.success(
            "Reject H₀"
        )

        st.write(
            "### 6. Real-world conclusion"
        )

        st.write(
            """
            There is statistically significant evidence that
            the true average delivery time is greater than
            30 minutes.
            """
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "← Previous",
            key="final_previous",
            width="stretch"
        ):
            st.session_state.step = 7
            st.rerun()

    with col2:

        if st.button(
            "↺ Start Over",
            key="final_restart",
            width="stretch"
        ):
            st.session_state.clear()
            st.session_state.step = 0
            st.rerun()


# Add a new step function above, then append (label, function) here.
STEP_RENDERERS = (
    ("Introduction", render_introduction),
    ("Data", render_data_step),
    ("Hypotheses", render_hypotheses_step),
    ("Z-score", render_test_statistic_step),
    ("p-value", render_pvalue_step),
    ("Decision", render_significance_step),
    ("Errors", render_errors_step),
    ("Power", render_power_step),
    ("Final challenge", render_final_challenge),
)


# Streamlit reruns this short entry point after every interaction.
st.set_page_config(
    page_title="Hypothesis Testing Activity",
    page_icon="🧪",
    layout="centered"
)


initialize_session_state()
render_progress()
_, render_current_step = STEP_RENDERERS[st.session_state.step]
st.session_state.learning_sections = []
render_current_step()

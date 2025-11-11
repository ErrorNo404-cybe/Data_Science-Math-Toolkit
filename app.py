import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, integrate, series, lambdify
import networkx as nx
from itertools import combinations, permutations
import math

# Page config
st.set_page_config(page_title="Avash's Data Science Math Toolkit", layout="wide")
st.title("🧮 Avash's Data Science Math Toolkit")
st.markdown("An interactive educational tool covering core mathematical foundations of data science.")

# Sidebar navigation
topic = st.sidebar.selectbox("Choose a Topic", [
    "1. Linear Algebra",
    "2. Probability & Statistics",
    "3. Calculus & Optimization",
    "4. Discrete Math & Set Theory"
])

# 1. LINEAR ALGEBRA
if topic == "1. Linear Algebra":
    st.header("1. Linear Algebra")
    op = st.selectbox("Operation", [
        "Vectors and Scalars",
        "Dot Product",
        "Matrices and Operations",
        "Determinant",
        "Matrix Inverse",
        "Eigenvalues & Eigenvectors",
        "Singular Value Decomposition (SVD)",
        "Norms (L1, L2, Lp)"
    ])

    if op == "Vectors and Scalars":
        st.subheader("Vectors and Scalars")
        st.write("A **vector** is an ordered list of numbers; a **scalar** is a single number.")
        dim = st.slider("Vector dimension", 2, 5, 3)
        vec = st.text_input(f"Enter {dim} comma-separated numbers", "1,2,3")
        try:
            v = np.array([float(x.strip()) for x in vec.split(",")])
            if len(v) != dim:
                st.error("Dimension mismatch!")
            else:
                scalar = st.number_input("Scalar multiplier", value=2.0)
                st.write(f"Vector: {v}")
                st.write(f"Scalar × Vector = {scalar * v}")
        except:
            st.error("Invalid input. Use numbers separated by commas.")

    elif op == "Dot Product":
        st.subheader("Dot Product (Inner Product)")
        dim = st.slider("Vector dimension", 2, 5, 3)
        v1_input = st.text_input("Vector 1", "1,2,3")
        v2_input = st.text_input("Vector 2", "4,5,6")
        try:
            v1 = np.array([float(x) for x in v1_input.split(",")])
            v2 = np.array([float(x) for x in v2_input.split(",")])
            if len(v1) != len(v2):
                st.error("Vectors must have same dimension!")
            else:
                dot = np.dot(v1, v2)
                st.write(f"**v₁ · v₂ =** {dot}")
                st.latex(r"\mathbf{v_1} \cdot \mathbf{v_2} = \sum_{i=1}^n v_{1i} v_{2i}")
        except:
            st.error("Invalid vectors.")

    elif op == "Matrices and Operations":
        st.subheader("Matrix Operations")
        rows = st.slider("Rows", 2, 4, 2)
        cols = st.slider("Columns", 2, 4, 2)
        st.write("Enter Matrix A (row-wise, comma-separated):")
        mat_input = []
        for i in range(rows):
            row = st.text_input(f"Row {i+1}", "1,2")
            mat_input.append(row)
        try:
            A = np.array([[float(x) for x in r.split(",")] for r in mat_input])
            st.write("**Matrix A:**")
            st.write(A)

            # Transpose
            st.write("**Transpose Aᵀ:**")
            st.write(A.T)

            # Add scalar
            scalar = st.number_input("Add scalar to A", value=0.0)
            st.write(f"**A + {scalar}:**")
            st.write(A + scalar)

            # Multiply by scalar
            scalar2 = st.number_input("Multiply A by scalar", value=1.0)
            st.write(f"**{scalar2} × A:**")
            st.write(scalar2 * A)

        except Exception as e:
            st.error(f"Invalid matrix: {e}")

    elif op == "Determinant":
        st.subheader("Determinant of a Square Matrix")
        n = st.slider("Matrix size (n×n)", 2, 4, 2)
        st.write("Enter square matrix (row-wise):")
        mat_input = []
        for i in range(n):
            row = st.text_input(f"Row {i+1}", "1,2")
            mat_input.append(row)
        try:
            A = np.array([[float(x) for x in r.split(",")] for r in mat_input])
            if A.shape != (n, n):
                st.error("Matrix must be square.")
            else:
                det = np.linalg.det(A)
                st.write(f"**det(A) =** {det:.4f}")
                st.latex(r"\det(A) = |A|")
        except:
            st.error("Invalid matrix.")

    elif op == "Matrix Inverse":
        st.subheader("Matrix Inverse (A⁻¹)")
        n = st.slider("Matrix size (n×n)", 2, 4, 2)
        st.write("Enter invertible square matrix:")
        mat_input = []
        for i in range(n):
            row = st.text_input(f"Row {i+1}", "2,1")
            mat_input.append(row)
        try:
            A = np.array([[float(x) for x in r.split(",")] for r in mat_input])
            if np.linalg.det(A) == 0:
                st.error("Matrix is singular (not invertible)!")
            else:
                inv = np.linalg.inv(A)
                st.write("**A⁻¹ =**")
                st.write(inv)
                st.write("**Check: A × A⁻¹ ≈ I**")
                st.write(np.dot(A, inv).round(6))
        except:
            st.error("Invalid or non-invertible matrix.")

    elif op == "Eigenvalues & Eigenvectors":
        st.subheader("Eigenvalues and Eigenvectors")
        n = st.slider("Matrix size (n×n)", 2, 4, 2)
        st.write("Enter square matrix:")
        mat_input = []
        for i in range(n):
            row = st.text_input(f"Row {i+1}", "4,1")
            mat_input.append(row)
        try:
            A = np.array([[float(x) for x in r.split(",")] for r in mat_input])
            if A.shape[0] != A.shape[1]:
                st.error("Matrix must be square.")
            else:
                w, v = np.linalg.eig(A)
                st.write("**Eigenvalues:**", w)
                st.write("**Eigenvectors (columns):**")
                st.write(v)
                st.latex(r"A \mathbf{v} = \lambda \mathbf{v}")
        except Exception as e:
            st.error(f"Error: {e}")

    elif op == "Singular Value Decomposition (SVD)":
        st.subheader("Singular Value Decomposition")
        st.write("For any matrix A (m×n): A = U Σ Vᵀ")
        rows = st.slider("Rows (m)", 2, 4, 3)
        cols = st.slider("Columns (n)", 2, 4, 2)
        st.write("Enter matrix A:")
        mat_input = []
        for i in range(rows):
            row = st.text_input(f"Row {i+1}", "1,2")
            mat_input.append(row)
        try:
            A = np.array([[float(x) for x in r.split(",")] for r in mat_input])
            U, s, VT = np.linalg.svd(A)
            st.write("**A =**")
            st.write(A)
            st.write("**U =**")
            st.write(U)
            st.write("**Σ (singular values) =**")
            st.write(s)
            st.write("**Vᵀ =**")
            st.write(VT)
        except:
            st.error("Invalid matrix.")

    elif op == "Norms (L1, L2, Lp)":
        st.subheader("Vector Norms")
        vec_input = st.text_input("Enter vector (comma-separated)", "3,4")
        p = st.slider("p for Lp norm (p≥1)", 1, 10, 2)
        try:
            v = np.array([float(x) for x in vec_input.split(",")])
            l1 = np.linalg.norm(v, ord=1)
            l2 = np.linalg.norm(v, ord=2)
            lp = np.linalg.norm(v, ord=p)
            st.write(f"**L1 norm =** {l1}")
            st.write(f"**L2 norm =** {l2}")
            st.write(f"**L{p} norm =** {lp:.4f}")
            st.latex(r"\| \mathbf{v} \|_p = \left( \sum_i |v_i|^p \right)^{1/p}")
        except:
            st.error("Invalid vector.")

# 2. PROBABILITY & STATISTICS
elif topic == "2. Probability & Statistics":
    st.header("2. Probability & Statistics")
    op = st.selectbox("Operation", [
        "Descriptive Statistics",
        "Covariance and Correlation",
        "Probability Distributions",
        "Bayes' Theorem",
        "Central Limit Theorem (CLT)",
        "Hypothesis Testing",
        "Statistical Inference",
        "Sampling Methods"
    ])

    if op == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        data_input = st.text_area("Enter data (comma-separated)", "1,2,3,4,5,6,7,8,9,10")
        try:
            data = np.array([float(x.strip()) for x in data_input.split(",")])
            st.write("**Data:**", data)
            st.write(f"Mean: {np.mean(data):.4f}")
            st.write(f"Median: {np.median(data):.4f}")
            st.write(f"Std Dev: {np.std(data, ddof=1):.4f}")
            st.write(f"Variance: {np.var(data, ddof=1):.4f}")
            st.write(f"Min: {np.min(data)}, Max: {np.max(data)}")
            st.write(f"Q1: {np.percentile(data,25):.2f}, Q3: {np.percentile(data,75):.2f}")
        except:
            st.error("Invalid data.")

    elif op == "Covariance and Correlation":
        st.subheader("Covariance & Correlation")
        x_input = st.text_area("X values", "1,2,3,4,5")
        y_input = st.text_area("Y values", "2,4,6,8,10")
        try:
            X = np.array([float(x) for x in x_input.split(",")])
            Y = np.array([float(y) for y in y_input.split(",")])
            if len(X) != len(Y):
                st.error("X and Y must have same length.")
            else:
                cov = np.cov(X, Y, ddof=1)[0,1]
                corr = np.corrcoef(X, Y)[0,1]
                st.write(f"**Cov(X,Y) =** {cov:.4f}")
                st.write(f"**Corr(X,Y) =** {corr:.4f}")
                st.latex(r"\text{Cov}(X,Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]")
                st.latex(r"\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}")
        except:
            st.error("Invalid input.")

    elif op == "Probability Distributions":
        st.subheader("Common Probability Distributions")
        dist = st.selectbox("Distribution", ["Normal", "Binomial", "Poisson", "Exponential"])
        if dist == "Normal":
            mu = st.number_input("Mean (μ)", value=0.0)
            sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.1)
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
            y = stats.norm.pdf(x, mu, sigma)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title(f"Normal(μ={mu}, σ={sigma})")
            st.pyplot(fig)

        elif dist == "Binomial":
            n = st.slider("Trials (n)", 1, 50, 10)
            p = st.slider("Success prob (p)", 0.0, 1.0, 0.5)
            k = np.arange(0, n+1)
            pmf = stats.binom.pmf(k, n, p)
            fig, ax = plt.subplots()
            ax.bar(k, pmf)
            ax.set_title(f"Binomial(n={n}, p={p})")
            st.pyplot(fig)

        elif dist == "Poisson":
            lam = st.slider("λ (rate)", 0.1, 10.0, 2.0)
            k = np.arange(0, 15)
            pmf = stats.poisson.pmf(k, lam)
            fig, ax = plt.subplots()
            ax.bar(k, pmf)
            ax.set_title(f"Poisson(λ={lam})")
            st.pyplot(fig)

        elif dist == "Exponential":
            lam = st.slider("λ (rate)", 0.1, 5.0, 1.0)
            x = np.linspace(0, 5/lam, 200)
            y = stats.expon.pdf(x, scale=1/lam)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title(f"Exponential(λ={lam})")
            st.pyplot(fig)

    elif op == "Bayes' Theorem":
        st.subheader("Bayes' Theorem")
        st.latex(r"P(A|B) = \frac{P(B|A) P(A)}{P(B)}")
        st.write("Example: Medical testing")
        p_disease = st.slider("P(Disease)", 0.0, 1.0, 0.01)
        sens = st.slider("Sensitivity P(Test⁺|Disease)", 0.0, 1.0, 0.95)
        spec = st.slider("Specificity P(Test⁻|No Disease)", 0.0, 1.0, 0.90)
        p_test_pos = p_disease * sens + (1 - p_disease) * (1 - spec)
        p_disease_given_pos = (sens * p_disease) / p_test_pos
        st.write(f"**P(Disease | Test⁺) =** {p_disease_given_pos:.4f}")
        st.write("Even with high sensitivity/specificity, rare diseases yield many false positives!")

    elif op == "Central Limit Theorem (CLT)":
        st.subheader("Central Limit Theorem Demo")
        dist_type = st.selectbox("Population Distribution", ["Uniform", "Exponential", "Skewed"])
        sample_size = st.slider("Sample Size (n)", 2, 100, 30)
        num_samples = st.slider("Number of Samples", 100, 5000, 1000)

        if dist_type == "Uniform":
            population = np.random.uniform(0, 10, 100000)
        elif dist_type == "Exponential":
            population = np.random.exponential(2, 100000)
        else:  # Skewed
            population = np.random.gamma(2, 2, 100000)

        means = [np.mean(np.random.choice(population, sample_size)) for _ in range(num_samples)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(population, bins=50, alpha=0.7, color='lightblue')
        ax1.set_title("Population Distribution")
        ax2.hist(means, bins=50, alpha=0.7, color='salmon')
        ax2.set_title(f"Sampling Distribution of Mean (n={sample_size})")
        st.pyplot(fig)
        st.write("→ Sampling distribution becomes normal as n increases!")

    elif op == "Hypothesis Testing":
        st.subheader("One-Sample t-Test")
        st.write("Test if sample mean = hypothesized mean (μ₀)")
        data_input = st.text_area("Sample data", "5.1,4.9,5.2,5.0,4.8,5.3")
        mu0 = st.number_input("Hypothesized mean (μ₀)", value=5.0)
        try:
            data = np.array([float(x) for x in data_input.split(",")])
            t_stat, p_val = stats.ttest_1samp(data, mu0)
            st.write(f"t-statistic = {t_stat:.4f}")
            st.write(f"p-value = {p_val:.4f}")
            alpha = 0.05
            if p_val < alpha:
                st.success(f"Reject H₀: mean ≠ {mu0} (p < {alpha})")
            else:
                st.info(f"Fail to reject H₀: mean = {mu0} (p ≥ {alpha})")
        except:
            st.error("Invalid data.")

    elif op == "Statistical Inference":
        st.subheader("Confidence Interval for Mean")
        data_input = st.text_area("Data", "10,12,11,13,14,9,15")
        confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
        try:
            data = np.array([float(x) for x in data_input.split(",")])
            n = len(data)
            mean = np.mean(data)
            std_err = np.std(data, ddof=1) / np.sqrt(n)
            h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
            st.write(f"Mean = {mean:.4f}")
            st.write(f"{int(confidence*100)}% CI: [{mean - h:.4f}, {mean + h:.4f}]")
        except:
            st.error("Invalid data.")

    elif op == "Sampling Methods":
        st.subheader("Sampling Methods")
        st.write("Population: integers 1 to 100")
        method = st.radio("Sampling Method", ["Simple Random", "Stratified", "Systematic"])
        sample_size = st.slider("Sample size", 5, 20, 10)
        np.random.seed(42)  # reproducible
        population = np.arange(1, 101)
        if method == "Simple Random":
            sample = np.random.choice(population, sample_size, replace=False)
        elif method == "Stratified":
            # 2 strata: 1-50, 51-100
            n1 = sample_size // 2
            n2 = sample_size - n1
            stratum1 = np.random.choice(np.arange(1,51), n1, replace=False)
            stratum2 = np.random.choice(np.arange(51,101), n2, replace=False)
            sample = np.concatenate([stratum1, stratum2])
        else:  # Systematic
            k = len(population) // sample_size
            start = np.random.randint(0, k)
            sample = population[start::k][:sample_size]
        st.write("Sample:", sorted(sample))

# 3. CALCULUS & OPTIMIZATION
elif topic == "3. Calculus & Optimization":
    st.header("3. Calculus & Optimization")
    op = st.selectbox("Operation", [
        "Derivatives",
        "Partial Derivatives & Gradients",
        "Chain Rule",
        "Maxima and Minima",
        "Integrals",
        "Taylor Series"
    ])

    x = symbols('x')
    y = symbols('y')

    if op == "Derivatives":
        st.subheader("Symbolic Derivatives")
        expr_str = st.text_input("Enter function f(x)", "x**2 + 3*x + 2")
        try:
            expr = sp.sympify(expr_str)
            deriv = diff(expr, x)
            st.latex(f"f(x) = {sp.latex(expr)}")
            st.latex(f"f'(x) = {sp.latex(deriv)}")
            # Plot
            f_lamb = lambdify(x, expr, 'numpy')
            d_lamb = lambdify(x, deriv, 'numpy')
            xs = np.linspace(-5, 5, 400)
            fig, ax = plt.subplots()
            ax.plot(xs, f_lamb(xs), label='f(x)')
            ax.plot(xs, d_lamb(xs), label="f'(x)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Invalid expression: {e}")

    elif op == "Partial Derivatives & Gradients":
        st.subheader("Partial Derivatives")
        expr_str = st.text_input("Enter f(x, y)", "x**2 + x*y + y**2")
        try:
            expr = sp.sympify(expr_str)
            dx = diff(expr, x)
            dy = diff(expr, y)
            st.latex(f"f(x,y) = {sp.latex(expr)}")
            st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(dx)}")
            st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(dy)}")
            st.latex(f"\\nabla f = \\left( {sp.latex(dx)}, {sp.latex(dy)} \\right)")
        except:
            st.error("Invalid expression.")

    elif op == "Chain Rule":
        st.subheader("Chain Rule Example")
        st.write("Let y = f(g(x))")
        f_str = st.text_input("f(u)", "sin(u)")
        g_str = st.text_input("g(x)", "x**2")
        try:
            u = symbols('u')
            f = sp.sympify(f_str)
            g = sp.sympify(g_str)
            y = f.subs(u, g)
            dy_dx = diff(y, x)
            st.latex(f"f(u) = {sp.latex(f)}")
            st.latex(f"g(x) = {sp.latex(g)}")
            st.latex(f"y = f(g(x)) = {sp.latex(y)}")
            st.latex(f"\\frac{{dy}}{{dx}} = {sp.latex(dy_dx)}")
            st.latex(r"\text{Chain Rule: } \frac{dy}{dx} = \frac{df}{du} \cdot \frac{dg}{dx}")
        except:
            st.error("Invalid functions.")

    elif op == "Maxima and Minima":
        st.subheader("Find Local Extrema")
        expr_str = st.text_input("f(x)", "x**3 - 3*x**2 + 2")
        try:
            expr = sp.sympify(expr_str)
            deriv = diff(expr, x)
            critical_points = sp.solve(deriv, x)
            st.latex(f"f(x) = {sp.latex(expr)}")
            st.latex(f"f'(x) = {sp.latex(deriv)}")
            st.write("**Critical points:**", critical_points)
            # Classify using second derivative
            d2 = diff(deriv, x)
            for cp in critical_points:
                val = d2.subs(x, cp)
                if val > 0:
                    st.write(f"x = {cp} → local **minimum**")
                elif val < 0:
                    st.write(f"x = {cp} → local **maximum**")
                else:
                    st.write(f"x = {cp} → inflection point")
            # Plot
            f_lamb = lambdify(x, expr, 'numpy')
            xs = np.linspace(-2, 4, 400)
            fig, ax = plt.subplots()
            ax.plot(xs, f_lamb(xs))
            ax.axhline(0, color='k', linewidth=0.5)
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

    elif op == "Integrals":
        st.subheader("Symbolic Integration")
        expr_str = st.text_input("f(x)", "2*x + 3")
        try:
            expr = sp.sympify(expr_str)
            integral = integrate(expr, x)
            definite = integrate(expr, (x, 0, 1))
            st.latex(f"f(x) = {sp.latex(expr)}")
            st.latex(f"\\int f(x) dx = {sp.latex(integral)} + C")
            st.latex(f"\\int_0^1 f(x) dx = {sp.latex(definite)}")
        except:
            st.error("Invalid expression.")

    elif op == "Taylor Series":
        st.subheader("Taylor Series Expansion")
        expr_str = st.text_input("f(x)", "sin(x)")
        order = st.slider("Order", 1, 10, 5)
        try:
            expr = sp.sympify(expr_str)
            taylor = series(expr, x, 0, order+1).removeO()
            st.latex(f"f(x) = {sp.latex(expr)}")
            st.latex(f"\\text{{Taylor series (order {order})}} = {sp.latex(taylor)}")
            # Plot
            f_lamb = lambdify(x, expr, 'numpy')
            t_lamb = lambdify(x, taylor, 'numpy')
            xs = np.linspace(-2*np.pi, 2*np.pi, 400)
            fig, ax = plt.subplots()
            ax.plot(xs, f_lamb(xs), label='f(x)')
            ax.plot(xs, t_lamb(xs), '--', label=f'Taylor (order {order})')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        except:
            st.error("Invalid function.")

# 4. DISCRETE MATH & SET THEORY
elif topic == "4. Discrete Math & Set Theory":
    st.header("4. Discrete Mathematics & Set Theory")
    op = st.selectbox("Operation", [
        "Set Theory and Logic",
        "Combinatorics",
        "Graph Theory"
    ])

    if op == "Set Theory and Logic":
        st.subheader("Set Operations")
        A_input = st.text_input("Set A (comma-separated)", "1,2,3,4")
        B_input = st.text_input("Set B (comma-separated)", "3,4,5,6")
        try:
            A = set([x.strip() for x in A_input.split(",")])
            B = set([x.strip() for x in B_input.split(",")])
            st.write(f"A = {A}")
            st.write(f"B = {B}")
            st.write(f"A ∪ B = {A | B}")
            st.write(f"A ∩ B = {A & B}")
            st.write(f"A - B = {A - B}")
            st.write(f"B - A = {B - A}")
            st.write(f"A Δ B = {A ^ B} (symmetric difference)")
            st.write(f"Is A ⊆ B? {A.issubset(B)}")
        except:
            st.error("Invalid sets.")

    elif op == "Combinatorics":
        st.subheader("Combinatorics")
        n = st.number_input("n (total items)", min_value=1, value=5)
        k = st.number_input("k (choose/select)", min_value=1, max_value=int(n), value=2)
        st.write(f"**Permutations P(n,k) = n!/(n−k)! =** {math.perm(int(n), int(k))}")
        st.write(f"**Combinations C(n,k) = n!/(k!(n−k)!) =** {math.comb(int(n), int(k))}")
        st.latex(r"P(n,k) = \frac{n!}{(n-k)!}, \quad C(n,k) = \binom{n}{k}")

    elif op == "Graph Theory":
        st.subheader("Graph Visualization")
        nodes = st.slider("Number of nodes", 3, 8, 5)
        edge_prob = st.slider("Edge probability", 0.2, 1.0, 0.5)
        G = nx.erdos_renyi_graph(nodes, edge_prob, seed=42)
        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
        st.pyplot(fig)
        st.write(f"Number of edges: {G.number_of_edges()}")
        if nx.is_connected(G):
            st.write("Graph is connected.")
        else:
            st.write("Graph is disconnected.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
st.caption("© 2025 Avash's Data Science Math Toolkit. All rights reserved.")
st.caption(
    "🔗 Data Trainer available at: [avash-data-trainer.streamlit.app](https://avash-data-trainer.streamlit.app)"
)
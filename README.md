# PID Auto-Tuner Using Machine Learning

## 1. Introduction

A PID controller regulates the speed of a motor using three gains: Kp, Ki and Kd. The correct values depend on the physical properties of each motor, and finding them usually requires trial and error or a separate optimisation for every motor.

In this project I built a machine learning model that predicts suitable PID gains directly from a motor's parameters. No ready-made dataset exists for this problem, so I created one by simulating many DC motors in Python, searching for the best PID gains for each motor, and recording them as training labels. I then trained several regression models on this dataset, tested the best one on motors it had never seen, and compared it with other tuning approaches.

An interactive interface built with Streamlit allows any motor to be described with sliders and the resulting step responses to be compared immediately.

## 2. Project Structure

```
pid_autotuner/
    core.py             Motor model, PID simulation, metrics, classical tuning rules, search tuner
    run_pipeline.py     Dataset generation, model training, benchmark, robustness test, plots
    app.py              Streamlit interface
    requirements.txt    Python dependencies
    data/dataset.csv    Simulated motors with their best-found PID gains
    models/             Trained models
    results/            Score tables and plots
```

## 3. Method

### 3.1 Motor model

I modelled a DC motor with two states, armature current and shaft speed:

- di/dt = (V - R·i - Ke·w) / L
- dw/dt = (Kt·i - b·w) / J

The parameters are resistance R, inductance L, rotor inertia J, viscous friction b, torque constant Kt and back-EMF constant Ke. These six values describe a motor and are the inputs of the machine learning model.

The system is discretised exactly with a zero-order hold at a sample time of 2 ms. The controller output is limited to a 12 V supply, and the target speed is set to 50% of the motor's no-load speed so that the step is large but reachable. I included anti-windup (the integrator stops accumulating while the output is saturated) and applied the derivative term to the measured speed to avoid a spike at the step.

### 3.2 Generating motors

I sampled motors randomly within realistic ranges:

| Parameter | Range |
|---|---|
| R | 0.8 to 4 ohm (uniform) |
| L | 1 to 8 mH (log-uniform) |
| J | 5e-5 to 5e-4 kg m^2 (log-uniform) |
| b | 1e-5 to 5e-4 N m s (log-uniform) |
| Kt | 0.03 to 0.12 N m/A (log-uniform) |
| Ke | Kt multiplied by a random factor between 0.9 and 1.1 |

Motors whose mechanical time constant fell outside 30 to 500 ms were rejected, so that every simulation has meaningful dynamics within the 2.5 s window.

### 3.3 Finding the best PID for each motor

For every motor I tested 600 PID combinations in closed loop:

1. 400 random combinations, sampled on a logarithmic scale.
2. 200 further combinations concentrated around the best result of the first stage.

The search ranges are scaled by the motor's own dynamics (DC gain and time constants), so the same search settings work for fast and slow motors. All combinations of one motor are simulated together as vectors, which keeps the dataset generation fast.

### 3.4 Performance measure

For each step response I measured:

- rise time (time to reach 90% of the target),
- settling time (time after which the speed stays within 2% of the target),
- overshoot,
- steady-state error (average error over the last 10% of the run).

I combined them into one cost, where lower is better:

> cost = settling time / T + rise time / T + 2 × overshoot + 5 × steady-state error

Here T is the simulation length, and overshoot and error are fractions of the target. The combination with the lowest cost became the label for that motor. The weights are my own design choice, and the results depend on them.

### 3.5 Dataset

The dataset has one row per motor. The inputs are the six motor parameters plus three derived quantities (DC gain, mechanical time constant and electrical time constant), all on a log10 scale. The outputs are log10(Kp), log10(Ki) and log10(Kd). I used the logarithm because the gains span several orders of magnitude.

I generated 1000 motors and split them 80% for training, 10% for validation and 10% for testing.

### 3.6 Models

I trained four regressors, each preceded by feature standardisation:

- Ridge regression (baseline)
- Random Forest
- Gradient Boosting
- Multi-layer perceptron (two hidden layers of 64 units)

The model with the lowest validation error was selected for the benchmark.

### 3.7 Comparison methods

I tested the selected model on 100 new motors generated with a different random seed, so none of them appeared in training. The following methods were compared on the same motors:

- **ML model:** gains predicted from the motor parameters.
- **Ziegler-Nichols:** reaction-curve rules.
- **Cohen-Coon:** reaction-curve rules.
- **Search optimiser:** the same search used to create the labels, run again on each test motor.

Ziegler-Nichols and Cohen-Coon both need a first-order-plus-delay model of the motor. I obtained it from a simulated open-loop step response using the two-point (28.3% and 63.2%) method.

The search optimiser should be read as a reference rather than an independent competitor, because it produced the training labels.

## 4. Results

### 4.1 Prediction accuracy (log10 scale, held-out test motors)

| Model | Validation MAE | Test MAE | Test R2 |
|---|---|---|---|
| Ridge | 0.436 | 0.429 | 0.368 |
| Random Forest | 0.434 | 0.429 | 0.381 |
| Gradient Boosting | 0.436 | 0.428 | 0.381 |
| MLP | 0.441 | 0.441 | 0.325 |

Random Forest had the lowest validation error and was used in the benchmark. The R2 values are modest. This is expected, because many different PID combinations give almost identical responses, so the best gains for a motor are not uniquely defined. For this reason I consider the control performance below to be the more meaningful measure.

### 4.2 Control performance on unseen motors

| Method | Mean cost | Median cost | Overshoot (%) | Settling (s) | Tuning time (ms) |
|---|---|---|---|---|---|
| Search optimiser | 0.0725 | 0.0528 | 0.0005 | 0.096 | 118.7 |
| ML (Random Forest) | 0.0743 | 0.0548 | 0.003 | 0.101 | 21.4 |
| Cohen-Coon | 0.0858 | 0.0657 | 0.65 | 0.097 | 3.0 |
| Ziegler-Nichols | 0.1038 | 0.0869 | 1.50 | 0.100 | 2.9 |

All four methods reached the target with zero steady-state error on every test motor.

### 4.3 Robustness (median cost on 40 test motors)

| Method | Plant mismatch (J and R changed by up to 20%) | Noise 1% | Noise 3% |
|---|---|---|---|
| Search optimiser | 0.0599 | 1.0780 | 1.2219 |
| ML (Random Forest) | 0.0652 | 1.0373 | 1.2230 |
| Cohen-Coon | 0.0798 | 1.0995 | 1.2399 |
| Ziegler-Nichols | 0.0934 | 1.1024 | 1.2473 |

In the noise columns the cost is above 1 for every method. Sensor noise keeps the speed outside the 2% settling band, so the settling term reaches its maximum. These columns are therefore useful only for comparing methods with each other.

## 5. Analysis

- The ML model reaches a mean cost within about 3% of the search optimiser and clearly below both classical methods.
- Its tuning time is about 6 times shorter than the optimiser's. Most of its 21 ms is the overhead of evaluating a 300-tree forest, and a smaller model would be faster.
- The classical rules are the fastest (about 3 ms), but they produce more overshoot and a higher cost.
- When the motor differs from the one used for tuning, the ML gains degrade less than the classical gains.
- The speed advantage of the ML model over the optimiser is large only because the optimiser needs 600 simulations per motor. For a single motor the difference is a fraction of a second, so the benefit grows with the number of motors to be tuned.

## 6. Limitations

- All data come from a simulated linear DC motor. No physical hardware was used.
- The model only covers speed control, and the training range is limited. Motors outside it (for example, mechanical time constants beyond 30 to 500 ms) are predicted poorly, and the interface shows a warning in that case.
- The labels depend on my cost function and on a random search, so they are good solutions rather than proven optima.
- Gain predictions have limited R2 because the optimum is flat.

## 7. Possible Extensions

- Add load disturbances, nonlinear friction and quantised sensors to the simulator.
- Use a better optimiser, such as differential evolution, to create the labels.
- Predict a distribution of good gains instead of a single value.
- Validate on a real motor.

## 8. How to Run

Python 3.9 or newer is required. Python 3.11 or 3.12 is recommended.

```
pip install -r requirements.txt
python run_pipeline.py --motors 1000 --test 100 --jobs -1
streamlit run app.py
```

The first command installs the dependencies. The second generates the dataset, trains the models, runs the benchmark and writes the results to `data/`, `models/` and `results/`. The third starts the interface at http://localhost:8501.

The trained model file depends on the installed scikit-learn version, so it must be regenerated with the second command on a new machine before starting the interface.

All random seeds are fixed, so the dataset and the accuracy figures are reproducible. Timing figures vary with the computer used.

## 9. Interface

- **Tuner:** sliders for the six motor parameters and for sensor noise. The step response and a metrics table are shown for the ML model and the comparison methods.
- **Benchmark:** the tables and plots from section 4.
- **Dataset:** the generated data, with a download option.
- **How it works:** a short description of the method.

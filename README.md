# PID Auto-Tuner Using Machine Learning

## 1. Introduction

A PID controller regulates the speed of a motor using three gains: Kp, Ki and Kd. The correct values depend on the physical properties of each motor, and finding them usually requires trial and error or a separate optimisation for every motor.

In this project I built a machine learning model that predicts suitable PID gains directly from a motor's parameters. No ready-made dataset exists for this problem, so I created one by simulating many DC motors in Python, searching for the best PID gains for each motor, and recording them as training labels. I then trained several regression models on this dataset, tested the best one on motors it had never seen, and compared it with other tuning approaches.

An interactive interface built with Streamlit allows any motor to be described with sliders, the resulting step responses to be compared immediately, and the accuracy figures of the model to be viewed in one place.

## 2. Project Structure

```
pid_autotuner/
    core.py                  Motor model, PID simulation, metrics, classical tuning rules, search tuner
    run_pipeline.py          Dataset generation, model training, benchmark, robustness test, plots
    app.py                   Streamlit interface
    accuracy_report.py       Prints the exact accuracy figures from the benchmark results
    requirements.txt         Python dependencies
    data/dataset.csv         Simulated motors with their best-found PID gains
    models/                  Trained models
    results/                 Score tables, plots and reported_accuracy.json
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

The model with the lowest validation error was selected for the benchmark. In my final run on my own computer this was Gradient Boosting. The selected model can differ between runs on different machines, because the library versions differ, so the file `results/model_scores.csv` always shows which model was chosen.

### 3.7 Comparison methods

I tested the selected model on 100 new motors generated with a different random seed, so none of them appeared in training. The following methods were compared on the same motors:

- **ML model:** gains predicted from the motor parameters.
- **Ziegler-Nichols:** reaction-curve rules.
- **Cohen-Coon:** reaction-curve rules.
- **Search optimiser:** the same search used to create the labels, run again on each test motor.

Ziegler-Nichols and Cohen-Coon both need a first-order-plus-delay model of the motor. I obtained it from a simulated open-loop step response using the two-point (28.3% and 63.2%) method.

The search optimiser should be read as a reference rather than an independent competitor, because it produced the training labels.

## 4. Results

The full tables are written to the `results/` folder each time the pipeline runs (`model_scores.csv`, `comparison_summary.csv`, `comparison_per_motor.csv` and `robustness.csv`), together with the plots. The figures below are from my final run.

### 4.1 Prediction accuracy

The model scores are measured on held-out test motors, on the log10 scale of the gains. The selected model, Gradient Boosting, reached a test R2 of 0.359 (35.9%). This value is modest, and this is expected, because many different PID combinations give almost identical responses, so the best gains for a motor are not uniquely defined. For this reason I consider the control performance in section 4.2 to be the more meaningful measure, and R2 should always be reported as the R2 of the predicted gains, not as the accuracy of the controller.

### 4.2 Control accuracy on unseen motors

The predicted gains were used in the closed-loop simulation of 100 unseen motors, and the result was compared with the search optimiser. The figures are produced by `accuracy_report.py` and shown in the Accuracy tab of the interface.

| Measure | Result |
|---|---|
| Control success rate (overshoot below 10%, steady-state error below 2%) | 100.0% |
| Motors where the ML cost is within 10% of the optimiser cost | 79.0% |
| Average performance relative to the optimiser (optimiser cost divided by ML cost) | 95.5% |

Every test motor reached its target. The ML model therefore gives roughly 95% of the performance of the optimiser, while needing a single prediction instead of 600 simulations. Results for the two classical methods on the same motors, including overshoot, settling time and tuning time, are listed in `results/comparison_summary.csv`.

### 4.3 Robustness

I tested the gains of every method under 1% and 3% sensor noise and with the motor's J and R changed by up to 20% after tuning. The medians over 40 test motors are stored in `results/robustness.csv`. In the noise columns the cost is above 1 for every method. Sensor noise keeps the speed outside the 2% settling band, so the settling term reaches its maximum. These columns are therefore useful only for comparing methods with each other.

### 4.4 Separately reported evaluation

In a separate evaluation on 25 held-out synthetic motors, I counted a motor as valid when the gains passed the closed-loop checks. The valid rates were as follows:

| Method | Valid rate on 25 held-out motors |
|---|---|
| Direct ML prediction | 92% (23/25) |
| ML-assisted local search | 96% (24/25) |
| Random search | 100% (25/25) |
| Ziegler-Nichols | 76% (19/25) |
| IMC PI | 64% (16/25) |

In that evaluation, random search was the more reliable method, and ML-assisted local search had the lower score on 21 of the 24 motors where both succeeded. ML-assisted local search and IMC PI are not part of `run_pipeline.py`. These figures use a different test set and different pass criteria from section 4.2, so the two sets of numbers must not be compared directly. They are stored in `results/reported_accuracy.json` and displayed in the Accuracy tab of the interface.

## 5. Analysis

- The ML model reaches about 95% of the optimiser's performance on unseen motors, and all of them were controlled successfully.
- Prediction takes a fraction of the time that the optimiser needs, because the optimiser runs 600 simulations per motor. The benefit grows with the number of motors to be tuned. For a single motor the difference is small.
- The classical rules need no training and only one open-loop simulation, so their tuning time is the shortest, but they are not designed for a saturating drive, and the benchmark table shows how this affects their overshoot and cost.
- The ML model does not beat random search on reliability in the separately reported evaluation, so I do not claim that it replaces a search. Its advantage is speed.
- The lower R2 of the predicted gains does not prevent good control, because several gain combinations produce nearly the same response.

## 6. Limitations

- All data come from a simulated linear DC motor. No physical hardware was used.
- The model only covers speed control, and the training range is limited. Motors outside it (for example, mechanical time constants beyond 30 to 500 ms) are predicted poorly, and the interface shows a warning in that case.
- The labels depend on my cost function and on a random search, so they are good solutions rather than proven optima.
- Gain predictions have limited R2 because the optimum is flat.
- The accuracy figures in sections 4.2 and 4.4 come from two different test sets and pass criteria.

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
python accuracy_report.py
streamlit run app.py
```

The first command installs the dependencies. The second generates the dataset, trains the models, runs the benchmark and writes the results to `data/`, `models/` and `results/`. The third prints the exact accuracy figures from those results. The fourth starts the interface at http://localhost:8501.

The trained model file depends on the installed scikit-learn version, so it must be regenerated with the second command on a new machine before starting the interface. The file `results/reported_accuracy.json` must be present in the `results/` folder for the reported table to appear in the Accuracy tab.

All random seeds are fixed, so the dataset is reproducible on the same library versions. Timing figures vary with the computer used.

## 9. Interface

- **Tuner:** sliders for the six motor parameters and for sensor noise. The step response and a metrics table are shown for the ML model and the comparison methods.
- **Accuracy:** the reported valid rates with a bar chart (section 4.4), and the live accuracy of the trained model computed from the benchmark results (section 4.2).
- **Benchmark:** the tables and plots from the pipeline run.
- **Dataset:** the generated data, with a download option.
- **How it works:** a short description of the method.

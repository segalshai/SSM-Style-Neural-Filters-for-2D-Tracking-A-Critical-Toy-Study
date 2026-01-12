# SSM-Style-Neural-Filters-for-2D-Tracking-A-Critical-Toy-Study
SSM-Style Neural Filters for 2D Tracking: A  Critical Toy Study
https://github.com/segalshai/SSM-Style-Neural-Filters-for-2D-Tracking-A-Critical-Toy-Study/blob/main/main_EKF.py
<a href="Fast_Survey_Mamba_SSMs_for_Target_Tracking_with_Toy_Study.pdf" target="_blank">Read summery here (PDF)</a>

The interacting multiple model (IMM) filter is a standard baseline for maneuvering tracking: it formalizes maneuvering as switching dynamics and runs a bank of motion models. Its main limitations are strong dependence on the model bank (if the true maneuver is not represented, performance degrades sharply), sensitivity to transition/noise tuning (e.g., mode probabilities, 𝑄 / 𝑅 Q/R), and fragility under long measurement gaps where prediction error accumulates with limited opportunity for correction.

Two recent learning directions are relevant. First, KalmanNet-style methods retain the Kalman predict/update flow but learn mismatch-sensitive components (gain-like corrections), aiming to improve estimation under model mismatch. Second, selective state-space sequence models (SSMs) such as Mamba provide causal, linear-time streaming inference, motivating their use as learned motion priors that can propagate state through missing measurements more effectively than a fixed hand-designed model.

The attached report follows the second direction: it tests whether a small SSM-style (“Mamba-like”) streaming block can learn a turn prior and remain robust during missing-measurement intervals. The constant-velocity model is used only as a minimal baseline for a quick, controlled comparison.

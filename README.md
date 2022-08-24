# jax-mppi
Model Predictive Path Integral (MPPI) control with JAX.

This repo contains two implementations of the MPPI algorithm.
The [pendulum](https://gym.openai.com/envs/Pendulum-v0/) environment from the OpenAI gym works as a simple test task to illustrate the techniques.
The two algorithm versions are:

- A standard MPPI by [Williams et al. 2017](https://ieeexplore.ieee.org/document/7989202). Some parts of the code follow the control library [by Shunichi09](https://github.com/Shunichi09/PythonLinearNonlinearControl). Some additional modifications are based on [this](https://github.com/google-research/pddm) paper by Google.

- The [MPOPIS](https://github.com/sisl/MPOPIS) algorithm with adaptive importance sampling. This version uses a cross entropy method to iteratively update the control samling distribution, often achieving higher performance with fewer samples than the standard MPPI.

Both algorithms heavily rely on [JAX](https://github.com/google/jax) to implement the model rollout functions. JAX makes an excellent tool for sampling-based model predictive control (MPC) as it allows to replace for-loops over the system dynamics with the efficient [scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html). In addition, JAX allows vectorizing and jit-compiling the sampling step to achieve very high efficiency.

## Example use

```python
python main.py --algo cemppi --horizon 30 --opt_iters 20
```

![alt text](video/pendulum.gif)

## References

Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M.
Rehg, Byron Boots, and Evangelos A. Theodorou. Information theoretic MPC
for model-based reinforcement learning. In 2017 IEEE International Con-
ference on Robotics and Automation (ICRA), pages 1714–1721, 2017. doi:
10.1109/ICRA.2017.7989202.

Anusha Nagabandi, Kurt Konolige, Sergey Levine, and Vikash Kumar. Deep
dynamics models for learning dexterous manipulation. In Leslie Pack Kael-
bling, Danica Kragic, and Komei Sugiura, editors, Proceedings of the Con-
ference on Robot Learning, volume 100 of Proceedings of Machine Learning
Research, pages 1101–1112. PMLR, 30 Oct–01 Nov 2020.

Dylan M. Asmar, Ransalu Senanayake, Shawn Manuel, and Mykel J. Kochenderfer. Model predictive optimized path integral strategies, 2022.

The [control library by Shunichi09](https://github.com/Shunichi09/PythonLinearNonlinearControl).
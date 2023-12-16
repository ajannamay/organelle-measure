import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.plot(
    np.array([    0,  149,  247,  353,  479]),
    np.array([0.213,0.214,0.203,0.210,0.213])/0.213,
    label="glu-0"
)
plt.plot(
    np.array([    0,  132,  210,  306,  481]),
    np.array([0.017,0.023,0.036,0.064,0.150])/0.017,
    label="glu-0.5"
)
plt.plot(
    np.array([    0,  123,  240,  480]),
    np.array([0.006,0.012,0.021,0.118])/0.006,
    label="glu-5"
)
plt.plot(
    np.array([    0,   96,  212,  338,  454,  480]),
    np.array([0.017,0.026,0.061,0.152,0.384,0.481])/0.017,
    label="glu-50"
)
plt.plot(
    np.array([    0,   97,  216,  384,  480]),
    np.array([0.008,0.015,0.034,0.118,0.233])/0.008,
    label="glu-100"
)
plt.plot(
    np.array([    0,  158,  256,  389,  479]),
    np.array([0.012,0.034,0.074,0.192,0.440])/0.012,
    label="glu-200"
)
plt.legend()
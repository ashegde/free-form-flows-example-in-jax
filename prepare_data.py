import numpy as np

def generate_dataset(n: int = 1000, r_inner = 0.5, r_outer = 1.5):
    #theta = np.linspace(0, 2 * np.pi, n)
    theta = np.random.rand(n) * 2 * np.pi
    r = r_inner + (r_outer - r_inner) / 2 * (1 + np.sin(2 * theta))
    noise = 0.1 * (2.0 * np.random.rand(n) - 1.0)
    x = r * np.cos(theta) * (1 + noise)
    y = r * np.sin(theta) * (1 + noise)

    return np.stack([x, y], axis=1)

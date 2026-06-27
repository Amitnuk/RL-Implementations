import torch
import numpy as np
from function_approximation.value_based.fqn.network import FCQNetwork


if __name__ == "__main__" :

    torch.manual_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = np.random.randn(5)
    
    QNet = FCQNetwork(input_shape=state.shape[0],
                   output_shape=3,
                   hidden_units=(5,5))

    Q = QNet(state)

    print(f"Q={Q, Q.shape}")

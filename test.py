from pypile import PileManager
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    pile = PileManager()
    pile.read_dat(Path("./tests/Test-1-2.dat"))
    
    np.set_printoptions(linewidth=200, precision=2, suppress=True)
    print(f"Pile stiffness matrix K:\n{pile.K}")
    

    np.testing.assert_allclose(pile.K, [
        [ 3.75361337e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.65098740e+07,  0.00000000e+00],
        [ 0.00000000e+00,  3.68517766e+06,  0.00000000e+00, -1.63086648e+07, 0.00000000e+00,  6.98491931e-10],
        [ 0.00000000e+00,  0.00000000e+00,  3.40590554e+07,  0.00000000e+00, 1.86264515e-09,  0.00000000e+00],
        [ 0.00000000e+00, -1.62731362e+07,  0.00000000e+00,  4.64474149e+09, 0.00000000e+00, -2.79396772e-09],
        [ 1.64737208e+07,  0.00000000e+00,  1.86264515e-09,  0.00000000e+00, 5.76996816e+08,  0.00000000e+00],
        [ 0.00000000e+00,  6.98491931e-10,  0.00000000e+00, -9.31322575e-10, 0.00000000e+00,  5.72172846e+08]
    ], rtol=1e-5, atol=1e-8)
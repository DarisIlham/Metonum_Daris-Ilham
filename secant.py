import numpy as np

def broyden_method(x0, y0, epsilon=0.000001, max_iter=100):
    def f(xy):
        x, y = xy
        return np.array([x**2 + x * y - 10, y + 3 * x * y**2 - 57])
    
    xy = np.array([x0, y0])
    print(f"{'r':<3} {'x':<12} {'y':<12} {'deltaX':<12} {'deltaY':<12}")
    print(f"{0:<3} {xy[0]:<12.6f} {xy[1]:<12.6f} {0.000000:<12.6f} {0.000000:<12.6f}")
    
    # Jacobian awal (dari Newton)
    x, y = xy
    J = np.array([[2*x + y, x], [3*y**2, 1 + 6*x*y]])
    
    F = f(xy)
    for r in range(1, max_iter + 1):
        if np.linalg.det(J) == 0:
            print("Error: Determinan Jacobian mendekati nol.")
            break
        
        s = np.linalg.solve(-J, F)  # s = -J^{-1} F
        xy_new = xy + s
        delta_x, delta_y = abs(s)
        
        print(f"{r:<3} {xy_new[0]:<12.6f} {xy_new[1]:<12.6f} {delta_x:<12.6f} {delta_y:<12.6f}")
        
        if delta_x < epsilon and delta_y < epsilon:
            print("Konvergen.")
            return xy_new
        
        F_new = f(xy_new)
        y_diff = F_new - F
        J = J + np.outer(y_diff - J @ s, s) / np.dot(s, s)
        
        xy = xy_new
        F = F_new
    
    print("Tidak konvergen dalam max_iter.")
    return xy

# Jalankan
broyden_method(1.5, 3.5)
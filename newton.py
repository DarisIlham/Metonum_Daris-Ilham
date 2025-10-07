import math
# Implementasi metode Newton-Raphson untuk sistem persamaan non-linear
# Daris Muhammad Ilham - 21120123130054

def newton_raphson(x0, y0, epsilon=0.000001, max_iter=100):
    print(f"{'r':<3} {'x':<12} {'y':<12} {'deltaX':<12} {'deltaY':<12}")
    x = x0
    y = y0
    print(f"{0:<3} {x:<12.6f} {y:<12.6f} {0.000000:<12.6f} {0.000000:<12.6f}")
    
    for r in range(1, max_iter + 1):
        u = x**2 + x * y - 10
        v = y + 3 * x * y**2 - 57
        
        du_dx = 2 * x + y
        du_dy = x
        dv_dx = 3 * y**2
        dv_dy = 1 + 6 * x * y
        
        det = du_dx * dv_dy - du_dy * dv_dx
        if abs(det) < 1e-10:
            print("Error: Determinan Jacobian mendekati nol.")
            break
        
        delta_x = - (u * dv_dy - v * du_dy) / det
        delta_y = (u * dv_dx - v * du_dx) / det  # Note: + untuk delta_y berdasarkan formula
        
        x_new = x + delta_x
        y_new = y + delta_y
        
        print(f"{r:<3} {x_new:<12.6f} {y_new:<12.6f} {abs(delta_x):<12.6f} {abs(delta_y):<12.6f}")
        
        if abs(delta_x) < epsilon and abs(delta_y) < epsilon:
            print("Konvergen.")
            return x_new, y_new
        x = x_new
        y = y_new
    
    print("Tidak konvergen dalam max_iter.")
    return x, y

# Jalankan
newton_raphson(1.5, 3.5)
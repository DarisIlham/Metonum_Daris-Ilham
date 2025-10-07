import math

def fixed_point_jacobi(x0, y0, epsilon=0.000001, max_iter=100):
    print(f"{'r':<3} {'x':<12} {'y':<12} {'deltaX':<12} {'deltaY':<12}")
    x = x0
    y = y0
    print(f"{0:<3} {x:<12.6f} {y:<12.6f} {0.000000:<12.6f} {0.000000:<12.6f}")
    
    for r in range(1, max_iter + 1):
        try:
            x_new = math.sqrt(10 - x * y)  # g1B
            y_new = 57 - 3 * x * y**2     # g2A
        except ValueError:
            print("Error: Domain negatif pada sqrt.")
            break
        
        delta_x = abs(x_new - x)
        delta_y = abs(y_new - y)
        
        print(f"{r:<3} {x_new:<12.6f} {y_new:<12.6f} {delta_x:<12.6f} {delta_y:<12.6f}")
        
        if delta_x < epsilon and delta_y < epsilon:
            print("Konvergen.")
            return x_new, y_new
        x = x_new
        y = y_new
    
    print("Tidak konvergen dalam max_iter.")
    return x, y

# Jalankan
fixed_point_jacobi(1.5, 3.5)
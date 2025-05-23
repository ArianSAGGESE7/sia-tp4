import numpy as np
from Hopfield import HopfieldNetwork, show_pattern, add_noise, is_equal

# Definir patrones de letras (matrices 5x5, valores -1 y 1)
J = np.array([
    [-1, -1, 1, 1, -1],
    [-1,-1, -1, 1, -1],
    [-1,-1, -1, 1, -1],
    [ 1, -1,-1, 1, -1],
    [-1, 1,  1,-1, -1]
]).flatten()

L = np.array([
    [ 1, -1, -1, -1, -1],
    [ 1, -1, -1, -1, -1],
    [ 1, -1, -1, -1, -1],
    [ 1, -1, -1, -1, -1],
    [ 1,  1,  1,  1,  1]
]).flatten()

C = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1, -1, -1, -1, -1],
    [ 1, -1, -1, -1, -1],
    [ 1, -1, -1, -1, -1],
    [ 1,  1,  1,  1,  1]
]).flatten()

T = np.array([
    [ 1,  1,  1,  1,  1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1]
]).flatten()

# Stack de patrones
patterns = np.stack([J, L, C, T])
letters = ["J", "L", "C", "T"]

# Entrenar red
net = HopfieldNetwork()
net.train(patterns)

# --- Parte (a): patrón con ruido moderado ---
print("Prueba con ruido moderado (30%)")
test_index = 0  # Letra J
original = patterns[test_index]
noisy = add_noise(original, noise_level=0.8)

show_pattern(original, "Original (Letra J)")
show_pattern(noisy, "Entrada con Ruido (30%)")

recovered, steps = net.recall(noisy, steps=10)
for i, s in enumerate(steps):
    show_pattern(s, f"Paso {i}")

# Verificar si recuperó algún patrón conocido
for i, p in enumerate(patterns):
    if is_equal(recovered, p):
        print(f"Recuperó la letra {letters[i]}")
        break
else:
    print("Estado espurio (no coincide con letras conocidas)")

# # --- Parte (b): patrón MUY ruidoso ---
# print("\nPrueba con ruido alto (60%)")
# very_noisy = add_noise(original, noise_level=0.6)
# show_pattern(very_noisy, "Entrada con Ruido Alto (60%)")

# recovered2, steps2 = net.recall(very_noisy, steps=10)
# for i, s in enumerate(steps2):
#     show_pattern(s, f"Paso {i} (Posible estado espurio)")

# # Verificar si es espurio
# for i, p in enumerate(patterns):
#     if is_equal(recovered2, p):
#         print(f" Recuperó la letra {letters[i]}")
#         break
# else:
#     print("Estado espurio detectado")


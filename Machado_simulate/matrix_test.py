from colour.blindness import matrix_cvd_Machado2009

for severity in [i / 10 for i in range(11)]:
    M = matrix_cvd_Machado2009("Deuteranomaly", severity)
    print(M)

# M1 = matrix_cvd_Machado2009("Protanomaly", 0.1)
# M2 = matrix_cvd_Machado2009("Protanomaly", 0.5)
# M3 = matrix_cvd_Machado2009("Protanomaly", 0.9)


# print(M1)
# print(M2)
# print(M3)